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

import os
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
            from transformers import AutoProcessor
            # transformers 4.49+ 用 AutoModelForImageTextToText 取代旧的
            # AutoModelForVision2Seq(后者在更新版本里已弃用甚至移除)。优先用新名,
            # 回退旧名,兼容不同 transformers 版本(否则新版会报 cannot import name)。
            try:
                from transformers import AutoModelForImageTextToText as _AutoVLM
            except ImportError:
                from transformers import AutoModelForVision2Seq as _AutoVLM
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
            self.model = _AutoVLM.from_pretrained(
                hc.model_path, torch_dtype=dtype, device_map="auto"
            )
        else:
            self.model = _AutoVLM.from_pretrained(
                hc.model_path, torch_dtype=dtype
            ).to(hc.device)
        self.model.eval()
        # 模型架构标识:MiniCPM-V-4.6(NaViT 切片视觉)不能走通用 processor(text,images)——
        # 那会把按子图分组的 pixel_values 打平、丢掉 grids/num_patches_per_image、也不传
        # downsample_mode,导致其 vit_merger 的统一 reshape 崩(见 _build_minicpmv46_inputs)。
        # 故对它走一条复刻 LlamaFactory MiniCPMV4_6Plugin 的专属预处理;其它模型不受影响。
        self._model_type = str(getattr(getattr(self.model, "config", None), "model_type", "") or "")

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

    def _maybe_resize_max_side(self, pil_img):
        """按最长边上限做**纯等比缩放**(不对齐 patch),再交给 processor。

        与 mnn 的 image_max_side 同义:仅当 max(宽,高) 超过上限时等比缩小,patch/切片
        对齐仍交给模型自己的 processor。适合 MiniCPM-V 这类非 Qwen 的切片模型——把
        Qwen 专属的 image_max/min_pixels 关掉(设 0)、改用本项控制输入尺寸,避免那些
        旋钮污染切片网格。<=0 或未超限则原样返回。缩放用 BICUBIC(同 LlamaFactory/mnn)。
        """
        from PIL import Image
        side = self.cfg.inference.hf.image_max_side
        if not side or side <= 0:
            return pil_img
        w, h = pil_img.size
        longest = max(w, h)
        if longest <= side:
            return pil_img
        factor = side / longest
        dst = (max(int(w * factor), 1), max(int(h * factor), 1))
        return pil_img.resize(dst, Image.BICUBIC)

    def _minicpmv46_downsample_mode(self) -> str:
        """取 MiniCPM-V-4.6 的 downsample_mode(4x/16x 视觉 token 压缩比)。

        与 plugin 一致:优先环境变量 DOWNSAMPLE_MODE,否则读 image_processor.downsample_mode,
        缺省 "16x"。它既决定占位符里每 patch 的 token 数,也要透传给 forward(否则占位符
        token 数与模型内部对不上)。
        """
        ds = os.getenv("DOWNSAMPLE_MODE")
        if ds is None:
            ds = getattr(getattr(self.processor, "image_processor", None), "downsample_mode", "16x")
        return ds or "16x"

    def _minicpmv46_placeholder(self, mm_inputs: dict, downsample_mode: str) -> str:
        """复刻 MiniCPMV4_6Plugin._build_v4_6_placeholder(单图,image_idx=0)。

        按 NaViT token 计数展开图像占位符:<image> + image_token*N + </image>,再按切片网格
        (grids)拼 <slice>...</slice> 行。N = target_sizes.prod(-1)//token_divisor,
        token_divisor = 4(4x)/16(16x)。
        """
        proc = self.processor
        grids = mm_inputs.get("grids", [[0, 0]])
        num_patches_per_image = mm_inputs.get("num_patches_per_image", [1])
        target_sizes = mm_inputs.get("target_sizes")
        token_divisor = 4 if downsample_mode == "4x" else 16

        n_patches = int(num_patches_per_image[0])                 # 单图:flat_index=0
        num_tokens_per_patch = target_sizes[0:n_patches].prod(-1) // token_divisor
        num_rows, num_cols = int(grids[0][0]), int(grids[0][1])

        image_start = getattr(proc, "image_start_token", "<image>")
        image_end = getattr(proc, "image_end_token", "</image>")
        slice_start = getattr(proc, "slice_start_token", "<slice>")
        slice_end = getattr(proc, "slice_end_token", "</slice>")
        image_id_start = getattr(proc, "image_id_start_token", "<image_id>")
        image_id_end = getattr(proc, "image_id_end_token", "</image_id>")
        image_token = (
            getattr(proc, "image_token", None)
            or getattr(getattr(proc, "tokenizer", None), "image_token", None)
            or "<image>"
        )

        placeholder = image_start + image_token * int(num_tokens_per_patch[0]) + image_end
        if getattr(proc, "default_use_image_id", True):
            placeholder = f"{image_id_start}0{image_id_end}" + placeholder

        if getattr(proc, "slice_mode", True) and num_rows > 0 and num_cols > 0:
            per_slice = int(num_tokens_per_patch[1]) if num_tokens_per_patch.numel() > 1 else 0
            slice_ph = slice_start + image_token * per_slice + slice_end
            placeholder += "\n".join(slice_ph * num_cols for _ in range(num_rows))
        return placeholder

    def _build_minicpmv46_inputs(self, context: list[Turn], sample_id: str, pil_img) -> dict:
        """为 MiniCPM-V-4.6 构造与 LlamaFactory 同源的 generate 输入。

        视觉张量走 **image_processor 直调**(保留按子图分组的 pixel_values + grids +
        num_patches_per_image,喂给它自己的 vit_merger 才不崩);文本按 _minicpmv46_placeholder
        手工展开图像占位符后 tokenize;downsample_mode 透传给 forward;grids/num_patches_per_image
        forward 不收,不放进返回值。
        """
        hc = self.cfg.inference.hf
        # 校验恰好 1 个 <image>(与通用路径一致)。
        img_turns = [t for t in context if _INTERNAL_PLACEHOLDER in t.content]
        if len(img_turns) != 1:
            raise ValueError(
                f"样本 {sample_id}:hf 后端要求对话恰好含 1 个 {_INTERNAL_PLACEHOLDER} "
                f"占位符,当前 {len(img_turns)} 个。"
            )

        mm_inputs = self.processor.image_processor([pil_img], return_tensors="pt")
        downsample_mode = self._minicpmv46_downsample_mode()
        placeholder = self._minicpmv46_placeholder(mm_inputs, downsample_mode)

        # 把内部 <image> 占位替换成展开后的图像占位串,再套对话模板 + tokenize(纯文本)。
        messages: list[dict] = []
        if hc.system_prompt:
            messages.append({"role": "system", "content": hc.system_prompt})
        for turn in context:
            content = turn.content
            if _INTERNAL_PLACEHOLDER in content:
                content = content.replace(_INTERNAL_PLACEHOLDER, placeholder, 1)
            messages.append({"role": turn.role, "content": content})
        prompt = self.processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        text_inputs = self.processor.tokenizer(prompt, return_tensors="pt")

        return {
            "input_ids": text_inputs["input_ids"],
            "attention_mask": text_inputs["attention_mask"],
            "pixel_values": mm_inputs["pixel_values"],
            "target_sizes": mm_inputs["target_sizes"],
            "downsample_mode": downsample_mode,  # str:forward 显式形参,透传;非张量不 .to()
        }

    def build_inputs(self, context: list[Turn], images: list[str], sample_id: str):
        """通用 Qwen 路径:加载单图 + 复刻 LlamaFactory 预处理,返回 (processor 输入, 预处理后 PIL 图)。

        由 complete 与 onnx_precision 共用,保证 torch(safetensors)与 ONNX 两端喂**同源输入**
        (否则逐层数值对比无意义)。返回的 BatchFeature 仍在 CPU 上(未 .to(device)),
        含 input_ids / attention_mask / pixel_values / image_grid_thw。
        MiniCPM-V-4.6 有专属预处理(_build_minicpmv46_inputs),不走这里。
        """
        from PIL import Image

        if len(images) != 1:
            raise ValueError(
                f"样本 {sample_id}:hf 参考后端仅支持单图推理,当前 {len(images)} 张。"
            )
        img_path = resolve_image_path(images[0], self.cfg)
        if not img_path.exists():
            raise FileNotFoundError(f"图片不存在: {img_path}(原始引用: {images[0]})")
        pil_img = self._maybe_resize_max_side(Image.open(img_path).convert("RGB"))
        messages = self._build_messages(context, sample_id)
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text], images=[pil_img], padding=True, return_tensors="pt"
        )
        return inputs, pil_img

    def _to_device(self, obj, device):
        """把(可能嵌套 list/tuple 的)张量搬到 device;非张量(如 downsample_mode 字符串)原样返回。"""
        if self._torch.is_tensor(obj):
            return obj.to(device)
        if isinstance(obj, (list, tuple)):
            return type(obj)(self._to_device(x, device) for x in obj)
        return obj

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
            if self._model_type == "minicpmv4_6":
                # MiniCPM-V-4.6:走复刻 LlamaFactory 的专属预处理(image_processor 直调 +
                # 手工占位符 + downsample_mode);其自身 image_processor 负责切片/尺寸,故不预缩放。
                from PIL import Image

                if len(images) != 1:
                    raise ValueError(
                        f"样本 {sample_id}:hf 参考后端仅支持单图推理,当前 {len(images)} 张。"
                    )
                img_path = resolve_image_path(images[0], self.cfg)
                if not img_path.exists():
                    raise FileNotFoundError(f"图片不存在: {img_path}(原始引用: {images[0]})")
                pil_img = Image.open(img_path).convert("RGB")
                inputs = self._build_minicpmv46_inputs(context, sample_id, pil_img)
            else:
                # 通用 Qwen 路径:抽出的 build_inputs(与 onnx_precision 共用),含图片加载 + 等比缩放。
                inputs, pil_img = self.build_inputs(context, images, sample_id)
        except Exception as e:  # 构造阶段失败(缺图/占位符不符等)
            self._raise_if_fail_fast()  # fail-fast:直接抛出带完整 traceback
            return Prediction(id=sample_id, error=f"build_prompt: {e}")

        with self._lock:
            try:
                # minicpmv4_6 路径返回普通 dict(含 str 型 downsample_mode);通用路径返回
                # BatchFeature。分别搬到设备并取 prompt 长度,再统一 **inputs 送 generate。
                if isinstance(inputs, dict):
                    inputs = {k: self._to_device(v, self.model.device) for k, v in inputs.items()}
                    prompt_len = int(inputs["input_ids"].shape[1])
                else:
                    inputs = inputs.to(self.model.device)
                    prompt_len = int(inputs.input_ids.shape[1])
                with self._torch.inference_mode():
                    gen = self.model.generate(
                        **inputs,
                        max_new_tokens=hc.max_tokens,
                        do_sample=not hc.greedy,
                        # 显式给 pad_token_id(回落 eos),否则 generate 每条都打印
                        # "Setting pad_token_id to eos_token_id" 警告(纯噪音,不影响结果)。
                        pad_token_id=self.processor.tokenizer.pad_token_id
                        or self.processor.tokenizer.eos_token_id,
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
                    # 喂给 processor 前的实际图片尺寸(经 image_max_side 等比缩放后),供对齐审计。
                    "hf_image_input_size": [pil_img.width, pil_img.height],
                }
                raw.update(self._image_meta(inputs))
                return Prediction(
                    id=sample_id,
                    prediction=out_text or "",
                    latency=round(time.time() - start, 3),
                    raw=raw,
                )
            except Exception as e:  # noqa: BLE001 - 捕获以记录而非中断整批
                self._raise_if_fail_fast()  # fail-fast:直接抛出带完整 traceback
                return Prediction(
                    id=sample_id,
                    latency=round(time.time() - start, 3),
                    error=f"{type(e).__name__}: {e}",
                )
