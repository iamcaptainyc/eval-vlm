"""离线 vLLM(pyvllm)本地推理后端。

不起 HTTP 服务:直接用 vllm 的 offline ``LLM`` 引擎在本机加载**合并后的全精度权重**推理
(与 openai/vllm 那种连远端 OpenAI 兼容 API 的后端互补)。主要给自动调参闭环用——
训练 → 合并 → 本后端离线加载评测,结果落 eval-vlm 的 run_dir。

关键事实:
  - ``LLM(model=..., mm_processor_kwargs={"min_pixels":..,"max_pixels":..}, ...)`` 一次性
    加载权重;图片的像素上下限由 mm_processor_kwargs 交给 Qwen-VL processor 内部缩放,
    **应与训练时的 image_max_pixels 对齐**,否则视觉 token 分布偏离训练。
  - ``llm.chat(messages, sampling_params)`` 接受 OpenAI 风格多模态 messages(图片用
    ``{"type":"image_url","image_url":{"url": <data URI>}}``),内部套用模型 chat 模板;
    返回 RequestOutput 列表,取 ``[0].outputs[0].text``。
  - 图片用 **base64 data URI** 传入(免去 vllm 的本地文件白名单 allowed_local_media_path 配置)。

约束(对齐 mnn/hf 后端):单图、由含 <image> 的 user 轮驱动;单个 LLM 对象有状态、
不可并发(thread_safe=False,编排层降为串行,本类再加锁兜底)。
"""
from __future__ import annotations

import base64
import os
import threading
import time
from pathlib import Path
from typing import Optional

from ..config import Config
from ..data.loader import resolve_image_path
from ..data.schema import Prediction, Turn
from .base import InferenceBackend

_INTERNAL_PLACEHOLDER = "<image>"

# 常见图片扩展名 -> data URI 的 mime;未知回落 image/jpeg。
_MIME = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
         ".webp": "image/webp", ".bmp": "image/bmp", ".gif": "image/gif"}


class VLLMOfflineBackend(InferenceBackend):
    # 单个有状态 LLM 对象 + 单 GPU,不并发(与 mnn/hf 一致);编排层据此串行。
    thread_safe = False

    def __init__(self, cfg: Config) -> None:
        super().__init__(cfg)
        vc = cfg.inference.vllm_offline
        # 必须在 import vllm 之前写环境变量(部分 vllm/flashinfer 变量在 import 期读取)。
        if vc.env:
            os.environ.update({str(k): str(v) for k, v in vc.env.items()})
        try:
            from vllm import LLM, SamplingParams
        except ImportError as e:  # pragma: no cover - 取决于运行环境是否装了 vllm
            raise ImportError(
                "backend=vllm_offline 需要安装 vllm(离线 LLM 引擎)。"
                "请在评测用的 conda 环境 `pip install vllm`。"
            ) from e

        if not vc.model_path:
            raise ValueError(
                "backend=vllm_offline 需要 inference.vllm_offline.model_path(合并后全精度权重目录);"
                "请在数据集 config.yaml 设置,或用 `eval-vlm pred --vllm-model <目录>` 指定。"
            )

        self._vc = vc
        self._lock = threading.Lock()
        llm_kwargs = dict(
            model=str(vc.model_path),
            gpu_memory_utilization=vc.gpu_memory_utilization,
            max_model_len=vc.max_model_len,
            max_num_seqs=vc.max_num_seqs,
            max_num_batched_tokens=vc.max_num_batched_tokens,
            trust_remote_code=vc.trust_remote_code,
            dtype=vc.dtype,
            limit_mm_per_prompt={"image": 1},
            mm_processor_kwargs={
                "min_pixels": vc.image_min_pixels,
                "max_pixels": vc.image_max_pixels,
            },
        )
        if vc.vllm_kwargs:                       # 逃生口:原样合并 vllm 原生键(如 enforce_eager)
            llm_kwargs.update(vc.vllm_kwargs)
        self.llm = LLM(**llm_kwargs)
        self.sampling = SamplingParams(
            max_tokens=vc.max_tokens,
            temperature=vc.temperature,
            top_p=vc.top_p,
            top_k=vc.top_k,
            repetition_penalty=vc.repetition_penalty,
        )

    # ------------------------------------------------------------------
    def _data_uri(self, img_path: Path) -> str:
        """把图片读成 base64 data URI(免 vllm 本地文件白名单配置)。"""
        mime = _MIME.get(img_path.suffix.lower(), "image/jpeg")
        b64 = base64.b64encode(img_path.read_bytes()).decode("ascii")
        return f"data:{mime};base64,{b64}"

    def _build_messages(self, context: list[Turn], sample_id: str, data_uri: str) -> list[dict]:
        """把对话上下文转成 vllm.chat 认的多模态 messages(仿 hf_backend._build_messages)。

        含 <image> 的 user 轮拆成 text/image_url 块(单图 -> 一个 image_url 块,用 data URI);
        其它轮纯文本原样带上(支持 few-shot/多轮)。system_prompt(若配置)作首个 system 轮。
        """
        img_turns = [t for t in context if _INTERNAL_PLACEHOLDER in t.content]
        if len(img_turns) != 1:
            raise ValueError(
                f"样本 {sample_id}:vllm_offline 后端要求对话恰好含 1 个 {_INTERNAL_PLACEHOLDER} "
                f"占位符,当前 {len(img_turns)} 个。"
            )
        messages: list[dict] = []
        if self._vc.system_prompt:
            messages.append({"role": "system",
                             "content": [{"type": "text", "text": self._vc.system_prompt}]})
        for turn in context:
            if _INTERNAL_PLACEHOLDER not in turn.content:
                messages.append({"role": turn.role,
                                 "content": [{"type": "text", "text": turn.content}]})
                continue
            parts: list[dict] = []
            segments = turn.content.split(_INTERNAL_PLACEHOLDER)
            for si, seg in enumerate(segments):
                if seg:
                    parts.append({"type": "text", "text": seg})
                if si < len(segments) - 1:
                    parts.append({"type": "image_url", "image_url": {"url": data_uri}})
            messages.append({"role": turn.role, "content": parts})
        return messages

    def complete(
        self,
        context: list[Turn],
        images: list[str],
        sample_id: str,
        expected: Optional[str] = None,
    ) -> Prediction:
        start = time.time()
        try:
            if len(images) != 1:
                raise ValueError(
                    f"样本 {sample_id}:vllm_offline 后端仅支持单图推理,当前 {len(images)} 张。"
                )
            img_path = resolve_image_path(images[0], self.cfg)
            if not img_path.exists():
                raise FileNotFoundError(f"图片不存在: {img_path}(原始引用: {images[0]})")
            messages = self._build_messages(context, sample_id, self._data_uri(img_path))
        except Exception as e:  # 构造阶段失败(缺图/占位符异常等)
            self._raise_if_fail_fast()
            return Prediction(id=sample_id, error=f"build_prompt: {e}")

        with self._lock:
            try:
                # v1 per-sample(0.8B 很快);后续可批量 self.llm.chat([m1,m2,...]) 提速。
                outs = self.llm.chat(messages, self.sampling, use_tqdm=False)
                out = outs[0]
                text_out = out.outputs[0].text if out.outputs else ""
                raw = {"backend": "vllm_offline"}
                try:
                    raw["prompt_len"] = len(out.prompt_token_ids)
                    raw["gen_seq_len"] = len(out.outputs[0].token_ids)
                    raw["prompt_token_count"] = raw["prompt_len"]
                except Exception:  # noqa: BLE001 - 统计可选
                    pass
                return Prediction(
                    id=sample_id,
                    prediction=text_out or "",
                    latency=round(time.time() - start, 3),
                    raw=raw,
                )
            except Exception as e:  # noqa: BLE001 - 记录而非中断整批
                self._raise_if_fail_fast()
                return Prediction(
                    id=sample_id,
                    latency=round(time.time() - start, 3),
                    error=f"{type(e).__name__}: {e}",
                )
