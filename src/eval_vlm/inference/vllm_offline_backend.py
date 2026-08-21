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

支持多图多轮:按 context 中 ``<image>`` 出现顺序从 images 列表头部取对应数量的图片,
与 OpenAI 后端的队列消费模式一致。``limit_mm_per_prompt`` 由配置项
``max_images_per_prompt``(默认 4)控制。

**supports_batch=True**:runner 走批量路径,把每一轮所有样本一次性交给 ``llm.chat([多对话])``,
由 vLLM 引擎内部 continuous batching 提速(而非逐条串行)——这是相对 openai/mnn/hf 后端的关键差异。
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
from .base import BatchItem, InferenceBackend

_INTERNAL_PLACEHOLDER = "<image>"

# 常见图片扩展名 -> data URI 的 mime;未知回落 image/jpeg。
_MIME = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
         ".webp": "image/webp", ".bmp": "image/bmp", ".gif": "image/gif"}


class VLLMOfflineBackend(InferenceBackend):
    # 单个有状态 LLM 对象 + 单 GPU,不并发(与 mnn/hf 一致);编排层据此串行(但见 supports_batch)。
    thread_safe = False
    # 支持真·批处理:runner 会把整轮所有样本一次交给 complete_batch,由 vLLM 内部 continuous batching。
    supports_batch = True

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
            limit_mm_per_prompt={"image": vc.max_images_per_prompt},
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

    def _build_messages(self, context: list[Turn], sample_id: str, data_uris: list[str]) -> list[dict]:
        """把对话上下文转成 vllm.chat 认的多模态 messages。

        含 <image> 的 user 轮拆成 text/image_url 块,按 <image> 出现顺序从 data_uris
        队列中消费 data URI(支持多图多轮)。其它轮纯文本原样带上(支持 few-shot/多轮)。
        system_prompt(若配置)作首个 system 轮。
        """
        img_queue = list(data_uris)
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
                    if not img_queue:
                        raise ValueError(
                            f"样本 {sample_id}：<image> 占位符多于可用图片数"
                        )
                    parts.append({"type": "image_url", "image_url": {"url": img_queue.pop(0)}})
            messages.append({"role": turn.role, "content": parts})
        return messages

    def _messages_for(self, context: list[Turn], images: list[str], sample_id: str) -> list[dict]:
        """按 context 中 <image> 占位符数量取图 + 解析路径 + 组多模态 messages。

        统计 context 里的 <image> 数量,从 images 列表头部取对应数量的图片
        (LlamaFactory 约定:images 按 <image> 出现顺序排列;runner 逐轮 rollout
        时 context 逐渐变长,占位符数也随之增加,每次从头取即可)。
        构造失败抛异常,由调用方兜。
        """
        n_placeholders = sum(t.content.count(_INTERNAL_PLACEHOLDER) for t in context)
        if n_placeholders == 0:
            raise ValueError(
                f"样本 {sample_id}：context 中没有 {_INTERNAL_PLACEHOLDER} 占位符"
            )
        if n_placeholders > len(images):
            raise ValueError(
                f"样本 {sample_id}：<image> 占位符({n_placeholders})多于图片数({len(images)})"
            )
        needed = images[:n_placeholders]
        data_uris: list[str] = []
        for img in needed:
            img_path = resolve_image_path(img, self.cfg)
            if not img_path.exists():
                raise FileNotFoundError(f"图片不存在: {img_path}(原始引用: {img})")
            data_uris.append(self._data_uri(img_path))
        return self._build_messages(context, sample_id, data_uris)

    @staticmethod
    def _out_to_pred(out, sample_id: str, latency: Optional[float]) -> Prediction:
        """把一个 vLLM RequestOutput 转成 Prediction。"""
        text = out.outputs[0].text if out.outputs else ""
        raw = {"backend": "vllm_offline"}
        try:
            raw["prompt_len"] = len(out.prompt_token_ids)
            raw["gen_seq_len"] = len(out.outputs[0].token_ids)
            raw["prompt_token_count"] = raw["prompt_len"]
        except Exception:  # noqa: BLE001 - 统计可选
            pass
        return Prediction(id=sample_id, prediction=text or "", latency=latency, raw=raw)

    def complete(
        self,
        context: list[Turn],
        images: list[str],
        sample_id: str,
        expected: Optional[str] = None,
    ) -> Prediction:
        start = time.time()
        try:
            messages = self._messages_for(context, images, sample_id)
        except Exception as e:  # 构造阶段失败(缺图/占位符异常等)
            self._raise_if_fail_fast()
            return Prediction(id=sample_id, error=f"build_prompt: {e}")
        with self._lock:
            try:
                outs = self.llm.chat(messages, self.sampling, use_tqdm=False)
                return self._out_to_pred(outs[0], sample_id, round(time.time() - start, 3))
            except Exception as e:  # noqa: BLE001 - 记录而非中断整批
                self._raise_if_fail_fast()
                return Prediction(id=sample_id, latency=round(time.time() - start, 3),
                                  error=f"{type(e).__name__}: {e}")

    def complete_batch(self, items: list[BatchItem]) -> list[Prediction]:
        """真·批处理:一次把整批 prompt 交给 vLLM(内部 continuous batching)。

        构造失败的单条记 error、不进批;其余组成一次 llm.chat(多对话) 调用,按序映射回结果。
        返回与 items 等长同序的 Prediction。
        """
        start = time.time()
        preds: list[Optional[Prediction]] = [None] * len(items)
        msgs: dict[int, list[dict]] = {}
        for i, it in enumerate(items):
            try:
                msgs[i] = self._messages_for(it.context, it.images, it.sample_id)
            except Exception as e:  # 构造失败:记 error,不进批
                self._raise_if_fail_fast()
                preds[i] = Prediction(id=it.sample_id, error=f"build_prompt: {e}")

        if msgs:
            idxs = list(msgs.keys())
            conversations = [msgs[i] for i in idxs]
            with self._lock:
                try:
                    outs = self.llm.chat(conversations, self.sampling, use_tqdm=True)
                except Exception as e:  # noqa: BLE001 - 整批生成失败:每条记 error
                    self._raise_if_fail_fast()
                    for i in idxs:
                        preds[i] = Prediction(id=items[i].sample_id,
                                              error=f"{type(e).__name__}: {e}")
                    outs = None
            if outs is not None:
                lat = round((time.time() - start) / max(len(idxs), 1), 3)  # 均摊延迟(近似)
                for i, out in zip(idxs, outs):
                    preds[i] = self._out_to_pred(out, items[i].sample_id, lat)

        return [p if p is not None else Prediction(id=items[i].sample_id, error="unknown")
                for i, p in enumerate(preds)]
