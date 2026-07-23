"""HuggingFace(transformers)本地参考后端。

作为「转换前·训练态」的**参考基准**:用 transformers 直接加载本地权重推理,
与 mnn(转换后)后端做行为级精度对比(见 precision 命令)。设计要点:

  - **复刻训练预处理**:把配置的 image_min/max_pixels 传给 AutoProcessor
    (与 LlamaFactory 训练时在 processor 上设 image_max_pixels 等价),使参考端与
    候选端(mnn_backend 复刻的同一套规则)喂给视觉编码器的 token 分布一致。
  - **贪心解码**(默认):确定、可复现,便于与 mnn 的 argmax 逐条对齐。
  - **对齐元数据落 raw**:prompt_token_count / image_pixels / image_grid_thw /
    output_token_ids,供 precision 的「输入对齐审计」与「首发散点」使用。

约束(对齐 mnn_backend 能力,保证两端可比):单图、由含 <image> 的 user 轮驱动。
可选依赖:未装 transformers/torch 时构造后端才报错,不影响其它后端。
"""
from __future__ import annotations

import threading
import time
from typing import Any, Optional

from ..config import Config
from ..data.loader import resolve_image_path
from ..data.schema import Prediction, Turn
from .base import InferenceBackend

_INTERNAL_PLACEHOLDER = "<image>"


class HFBackend(InferenceBackend):
    # 单个有状态大模型对象 + 单 GPU,不并发(与 mnn 一致);编排层据此串行。
    thread_safe = False

    def __init__(self, cfg: Config) -> None:
        super().__init__(cfg)
        try:
            import torch  # noqa: F401
            from transformers import AutoModelForVision2Seq, AutoProcessor
        except ImportError as e:  # pragma: no cover - 取决于运行环境
            raise ImportError(
                "backend=hf 需要安装 transformers 与 torch(以及 Qwen2-VL 所需的 "
                "qwen-vl-utils);请 `pip install \"transformers>=4.45\" torch "
                "qwen-vl-utils`。"
            ) from e

        hc = cfg.inference.hf
        if not hc.model_path:
            raise ValueError(
                "backend=hf 需要 inference.hf.model_path(本地 HF 权重目录);"
                "请在数据集 config.yaml 设置,或用 `eval-vlm pred --hf-model <目录>` 指定。"
            )

        self._torch = torch
        self._lock = threading.Lock()

        # processor:把训练时的 image_max/min_pixels 规则下发,复刻 LlamaFactory 预处理。
        proc_kwargs: dict[str, Any] = {}
        if hc.image_max_pixels and hc.image_max_pixels > 0:
            proc_kwargs["max_pixels"] = int(hc.image_max_pixels)
        if hc.image_min_pixels and hc.image_min_pixels > 0:
            proc_kwargs["min_pixels"] = int(hc.image_min_pixels)
        self.processor = AutoProcessor.from_pretrained(hc.model_path, **proc_kwargs)

        dtype = self._resolve_dtype(hc.dtype)
        if hc.device == "auto":
            self.model = AutoModelForVision2Seq.from_pretrained(
                hc.model_path, torch_dtype=dtype, device_map="auto"
            )
        else:
            self.model = AutoModelForVision2Seq.from_pretrained(
                hc.model_path, torch_dtype=dtype
            ).to(hc.device)
        self.model.eval()

    def _resolve_dtype(self, name: str):
        if not name or name == "auto":
            return "auto"
        return getattr(self._torch, name)

    # ------------------------------------------------------------------
    def _build_messages(self, context: list[Turn], sample_id: str) -> list[dict]:
        """把对话上下文转成 processor.apply_chat_template 认的多模态 messages。

        含 <image> 的 user 轮拆成 text/image 块(单图 -> 一个 image 块);其它轮纯文本
        原样带上(支持 few-shot / 多轮)。system_prompt(若配置)作为首个 system 轮。
        """
        hc = self.cfg.inference.hf
        img_turns = [t for t in context if _INTERNAL_PLACEHOLDER in t.content]
        if len(img_turns) != 1:
            raise ValueError(
                f"样本 {sample_id}:hf 后端要求对话恰好含 1 个 {_INTERNAL_PLACEHOLDER} "
                f"占位符,当前 {len(img_turns)} 个。"
            )

        messages: list[dict] = []
        if hc.system_prompt:
            messages.append({"role": "system",
                             "content": [{"type": "text", "text": hc.system_prompt}]})
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
                    parts.append({"type": "image"})
            messages.append({"role": turn.role, "content": parts})
        return messages

    def _image_meta(self, inputs) -> dict:
        """从 processor 输出提取图片对齐元数据(grid_thw -> 像素/视觉 token 数)。

        Qwen2-VL 的 image_grid_thw = [[t, h, w]](patch 单位,patch=14px)。
        resized 像素 ≈ (h*14)*(w*14);视觉 token 数 = t*h*w / merge^2(merge 默认 2)。
        取不到则返回空 dict(不影响推理)。
        """
        meta: dict = {}
        grid = getattr(inputs, "image_grid_thw", None)
        if grid is None and isinstance(inputs, dict):
            grid = inputs.get("image_grid_thw")
        if grid is None:
            return meta
        try:
            thw = [int(x) for x in list(grid[0])]
            t, h, w = thw[0], thw[1], thw[2]
            merge = int(getattr(self.processor.image_processor, "merge_size", 2) or 2)
            patch = int(getattr(self.processor.image_processor, "patch_size", 14) or 14)
            meta["image_grid_thw"] = thw
            meta["image_pixels"] = (h * patch) * (w * patch)
            meta["image_tokens"] = (t * h * w) // (merge * merge)
        except Exception:  # noqa: BLE001 - 元数据尽力而为
            pass
        return meta

    def complete(
        self,
        context: list[Turn],
        images: list[str],
        sample_id: str,
        expected: Optional[str] = None,
    ) -> Prediction:
        hc = self.cfg.inference.hf
        start = time.time()
        try:
            if len(images) != 1:
                raise ValueError(
                    f"样本 {sample_id}:hf 参考后端仅支持单图推理,当前 {len(images)} 张。"
                )
            from PIL import Image

            messages = self._build_messages(context, sample_id)
            img_path = resolve_image_path(images[0], self.cfg)
            if not img_path.exists():
                raise FileNotFoundError(f"图片不存在: {img_path}(原始引用: {images[0]})")
            pil_img = Image.open(img_path).convert("RGB")
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.processor(
                text=[text], images=[pil_img], padding=True, return_tensors="pt"
            )
        except Exception as e:  # 构造阶段失败(缺图/占位符不符等)
            return Prediction(id=sample_id, error=f"build_prompt: {e}")

        with self._lock:
            try:
                inputs = inputs.to(self.model.device)
                prompt_len = int(inputs.input_ids.shape[1])
                with self._torch.inference_mode():
                    gen = self.model.generate(
                        **inputs,
                        max_new_tokens=hc.max_tokens,
                        do_sample=not hc.greedy,
                    )
                trimmed = gen[:, prompt_len:]
                out_text = self.processor.batch_decode(
                    trimmed, skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0]
                raw: dict = {
                    "backend": "hf",
                    "prompt_token_count": prompt_len,
                    "output_token_ids": [int(x) for x in trimmed[0].tolist()],
                }
                raw.update(self._image_meta(inputs))
                return Prediction(
                    id=sample_id,
                    prediction=out_text or "",
                    latency=round(time.time() - start, 3),
                    raw=raw,
                )
            except Exception as e:  # noqa: BLE001 - 捕获以记录而非中断整批
                return Prediction(
                    id=sample_id,
                    latency=round(time.time() - start, 3),
                    error=f"{type(e).__name__}: {e}",
                )
