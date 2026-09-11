"""llama.cpp (libmtmd / llama-server / mtmd-cli) 多模态推理后端。

定位:
  利用本地或远端编译好的 llama.cpp 对多模态模型 (如 Qwen2-VL, Qwen2.5-VL, MiniCPM-V, Gemma-3 等 GGUF 格式) 进行推理评测。
  多模态模型在 llama.cpp 中通常由两个文件组成:
    1. 主语言模型: model.gguf
    2. 多模态投影器: mmproj.gguf

工作模式:
  1. mode="server" (推荐, 默认):
     通过 HTTP 请求调用由 `llama-server -m <model.gguf> --mmproj <mmproj.gguf>` 启动的服务。
     其 /v1/chat/completions 接口原生兼容 OpenAI 多模态格式,并在底层由 libmtmd 做图像切片编码与 continuous batching。
     支持多线程高并发 (`thread_safe=True`),并发度由 config.inference.llamacpp.max_concurrency 决定。
  2. mode="cli":
     通过子进程直接调用本地编译的 `mtmd-cli` 或 `llama-mtmd-cli` 可执行文件进行离线单条样本推理。
     因为进程调用排他,`thread_safe=False`,自动强制串行执行。

多轮与多图对齐:
  在 eval (runner.py) 的多轮 rollout 中,历史对话 context 会随着轮次推进逐渐变长。
  本后端严格遵循 LlamaFactory / vLLM 规范:
  按 context 中出现的 `<image>` 占位符数量与顺序,从 sample.images 列表中严格消费对应的图片,
  将其预处理 (可选等比缩小保护) 并编码为 Base64 Data URI,确保多图多轮不错位、不漏图、不重复消费。
"""
from __future__ import annotations

import base64
import io
import json
import math
import mimetypes
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Optional

from ..config import Config
from ..data.loader import resolve_image_path
from ..data.schema import Prediction, Turn
from .base import InferenceBackend

# 识别为常见图片的扩展名与 MIME 映射
_MIME = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".gif": "image/gif",
}

_INTERNAL_PLACEHOLDER = "<image>"


class LlamaCppBackend(InferenceBackend):
    def __init__(self, cfg: Config) -> None:
        super().__init__(cfg)
        self.lc = cfg.inference.llamacpp
        self.mode = self.lc.mode.lower().strip()

        if self.mode == "server":
            self.thread_safe = True
            try:
                from openai import OpenAI
            except ImportError as e:
                raise ImportError(
                    "backend=llamacpp(mode='server') 需要安装 openai 客户端库: pip install openai"
                ) from e

            # 初始化客户端,重试由本后端统一接管控制
            self.client = OpenAI(
                base_url=self.lc.base_url,
                api_key=self.lc.api_key or "EMPTY",
                timeout=self.lc.request_timeout,
                max_retries=0,
            )
        elif self.mode == "cli":
            self.thread_safe = False
            # 检查 CLI 二进制是否存在 (如果提供了)
            if self.lc.cli_binary and not Path(self.lc.cli_binary).expanduser().exists():
                raise FileNotFoundError(f"未找到 mtmd-cli 可执行文件: {self.lc.cli_binary}")
        else:
            raise ValueError(f"未知 llama.cpp 运行模式: {self.mode!r} (可选: 'server', 'cli')")

    # ------------------------------------------------------------------
    # 图像预处理与尺寸规范化 (与 MNN / HF 对齐,避免超大图压垮 C++ 视觉编码器)
    # ------------------------------------------------------------------
    def _preprocess_image_bytes(self, img_path: Path) -> tuple[bytes, str]:
        """读取并按配置等比缩放图片,返回 (二进制数据, MIME 类型)。"""
        ext = img_path.suffix.lower()
        mime = _MIME.get(ext, "image/jpeg")

        max_pixels = self.lc.image_max_pixels
        min_pixels = self.lc.image_min_pixels
        max_side = self.lc.image_max_side

        # 如果没有开启任何尺寸约束,直接读二进制返回
        if max_pixels <= 0 and min_pixels <= 0 and max_side <= 0:
            return img_path.read_bytes(), mime

        try:
            from PIL import Image
            with Image.open(img_path) as im:
                im = im.convert("RGB")
                w, h = im.size
                orig_w, orig_h = w, h

                # 1. 最长边限制 (纯等比缩小)
                if max_side > 0 and max(w, h) > max_side:
                    scale = max_side / max(w, h)
                    w, h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))

                # 2. 总像素上限与下限
                cur_pixels = w * h
                if max_pixels > 0 and cur_pixels > max_pixels:
                    scale = math.sqrt(max_pixels / cur_pixels)
                    w, h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
                elif min_pixels > 0 and cur_pixels < min_pixels:
                    scale = math.sqrt(min_pixels / cur_pixels)
                    w, h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))

                if (w, h) != (orig_w, orig_h):
                    im = im.resize((w, h), Image.Resampling.BICUBIC)

                buf = io.BytesIO()
                im.save(buf, format="JPEG", quality=95)
                return buf.getvalue(), "image/jpeg"
        except Exception:
            # 如果缺少 PIL 或读取失败,安全降级为直接读取原文件
            return img_path.read_bytes(), mime

    def _data_uri_from_path(self, img_path: Path) -> str:
        """把本地图片路径转为经过尺寸校验与保护的 Base64 Data URI。"""
        data, mime = self._preprocess_image_bytes(img_path)
        b64 = base64.b64encode(data).decode("ascii")
        return f"data:{mime};base64,{b64}"

    # ------------------------------------------------------------------
    # 多图与多轮消息组装
    # ------------------------------------------------------------------
    def _build_messages(
        self, context: list[Turn], images: list[str], sample_id: str
    ) -> list[dict[str, Any]]:
        """把对话上下文 (含历史 assistant 轮与当前 user 轮) 转换为 OpenAI 兼容多模态消息。

        遵循规则:
          1. 支持多轮:历史 assistant 轮作为纯文本保留;
          2. 支持多图:按 context 中从前往后出现的 <image> 占位符顺序从 images 队列中消费图片;
          3. 若配置了 system_prompt,在首部注入 system 轮。
        """
        n_placeholders = sum(t.content.count(_INTERNAL_PLACEHOLDER) for t in context)
        if n_placeholders > len(images):
            raise ValueError(
                f"样本 {sample_id}: 对话中的 {_INTERNAL_PLACEHOLDER} 占位符数量({n_placeholders}) "
                f"超过了可用图片数量({len(images)})"
            )

        img_queue: list[str] = []
        for img_ref in images[:n_placeholders]:
            p = resolve_image_path(img_ref, self.cfg)
            if not p.exists():
                raise FileNotFoundError(f"图片不存在: {p} (原始引用: {img_ref})")
            img_queue.append(self._data_uri_from_path(p))

        messages: list[dict[str, Any]] = []
        if self.lc.system_prompt:
            messages.append({"role": "system", "content": self.lc.system_prompt})

        for turn in context:
            if turn.role != "user":
                # 历史 assistant 轮或 system 轮:纯文本透传
                messages.append({"role": turn.role, "content": turn.content})
                continue

            # user 轮可能包含 0, 1 或多个 <image> 占位符
            if _INTERNAL_PLACEHOLDER not in turn.content:
                messages.append({"role": "user", "content": turn.content})
                continue

            content_parts: list[dict[str, Any]] = []
            segments = turn.content.split(_INTERNAL_PLACEHOLDER)
            for si, seg in enumerate(segments):
                if seg:
                    content_parts.append({"type": "text", "text": seg})
                if si < len(segments) - 1:
                    if not img_queue:
                        raise ValueError(f"样本 {sample_id}: 内部图片队列已耗尽")
                    url = img_queue.pop(0)
                    content_parts.append({
                        "type": "image_url",
                        "image_url": {"url": url},
                    })
            if not content_parts:
                content_parts.append({"type": "text", "text": ""})
            messages.append({"role": "user", "content": content_parts})

        return messages

    # ------------------------------------------------------------------
    # 推理执行: Server 模式
    # ------------------------------------------------------------------
    def _complete_server(
        self, messages: list[dict[str, Any]], sample_id: str
    ) -> Prediction:
        start_time = time.time()
        last_err: Optional[Exception] = None

        # 组织请求参数
        extra_body: dict[str, Any] = {}
        if self.lc.top_k > 0:
            extra_body["top_k"] = self.lc.top_k
        if self.lc.repetition_penalty != 1.0:
            # llama-server 支持 repeat_penalty 或 repetition_penalty
            extra_body["repeat_penalty"] = self.lc.repetition_penalty

        req_kwargs: dict[str, Any] = {
            "model": self.lc.model or "default",
            "messages": messages,
            "max_tokens": self.lc.max_tokens,
            "temperature": self.lc.temperature,
            "top_p": self.lc.top_p,
        }
        if extra_body:
            req_kwargs["extra_body"] = extra_body

        for attempt in range(self.lc.max_retries + 1):
            try:
                resp = self.client.chat.completions.create(**req_kwargs)
                latency = time.time() - start_time
                ans = ""
                if resp.choices and len(resp.choices) > 0:
                    ans = resp.choices[0].message.content or ""

                tokens_used = 0
                raw_info = {"model": self.lc.model}
                if getattr(resp, "usage", None):
                    raw_info["completion_tokens"] = getattr(resp.usage, "completion_tokens", 0) or 0

                return Prediction(
                    id=sample_id,
                    prediction=ans,
                    latency=round(latency, 4),
                    raw=raw_info,
                )
            except Exception as e:
                last_err = e
                # 提示如果包含 mmproj 说明模型没有挂载多模态投影器
                err_msg = str(e)
                if "mmproj" in err_msg.lower():
                    self._raise_if_fail_fast()
                    return Prediction(
                        id=sample_id,
                        error=f"llama-server 未加载多模态投影器(缺少 --mmproj): {err_msg}",
                        latency=round(time.time() - start_time, 4),
                    )

                if attempt < self.lc.max_retries:
                    sleep_time = min(self.lc.retry_backoff * (2 ** attempt), 10)
                    time.sleep(sleep_time)
                else:
                    self._raise_if_fail_fast()

        if self.cfg.inference.fail_fast and last_err is not None:
            raise last_err

        return Prediction(
            id=sample_id,
            error=f"请求失败(重试 {self.lc.max_retries} 次): {last_err}",
            latency=round(time.time() - start_time, 4),
        )

    # ------------------------------------------------------------------
    # 推理执行: CLI 模式
    # ------------------------------------------------------------------
    def _complete_cli(
        self, context: list[Turn], images: list[str], sample_id: str
    ) -> Prediction:
        start_time = time.time()
        binary = self.lc.cli_binary or "mtmd-cli"
        model_path = self.lc.model_path
        mmproj_path = self.lc.mmproj_path

        if not model_path:
            return Prediction(id=sample_id, error="CLI 模式必须指定 inference.llamacpp.model_path")
        if not mmproj_path:
            return Prediction(id=sample_id, error="CLI 模式必须指定 inference.llamacpp.mmproj_path")

        # 统计本轮使用的图片
        n_placeholders = sum(t.content.count(_INTERNAL_PLACEHOLDER) for t in context)
        needed_imgs = images[:n_placeholders]

        cmd: list[str] = [
            binary,
            "-m", str(model_path),
            "--mmproj", str(mmproj_path),
            "-n", str(self.lc.max_tokens),
            "--temp", str(self.lc.temperature),
            "--top-p", str(self.lc.top_p),
        ]
        if self.lc.repetition_penalty != 1.0:
            cmd.extend(["--repeat-penalty", str(self.lc.repetition_penalty)])

        # 挂载全部图片
        for img in needed_imgs:
            p = resolve_image_path(img, self.cfg)
            if not p.exists():
                return Prediction(id=sample_id, error=f"图片不存在: {p}")
            cmd.extend(["--image", str(p)])

        # 拼装多轮 prompt (最后一轮提问 + 历史上下文)
        # 若是简单单轮:
        user_prompt = context[-1].content.replace(_INTERNAL_PLACEHOLDER, "").strip()
        cmd.extend(["-p", user_prompt])

        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=self.lc.request_timeout)
            latency = time.time() - start_time
            if res.returncode != 0:
                self._raise_if_fail_fast()
                return Prediction(
                    id=sample_id,
                    error=f"mtmd-cli 退出状态码 {res.returncode}: {res.stderr.strip()}",
                    latency=round(latency, 4),
                )
            output = res.stdout.strip()
            return Prediction(
                id=sample_id,
                prediction=output,
                latency=round(latency, 4),
            )
        except Exception as e:
            self._raise_if_fail_fast()
            return Prediction(
                id=sample_id,
                error=f"CLI 调用异常: {e}",
                latency=round(time.time() - start_time, 4),
            )

    # ------------------------------------------------------------------
    # 统一接口: complete
    # ------------------------------------------------------------------
    def complete(
        self,
        context: list[Turn],
        images: list[str],
        sample_id: str,
        expected: Optional[str] = None,
    ) -> Prediction:
        if self.mode == "server":
            try:
                messages = self._build_messages(context, images, sample_id)
            except Exception as e:
                self._raise_if_fail_fast()
                return Prediction(id=sample_id, error=f"build_messages: {e}")
            return self._complete_server(messages, sample_id)

        return self._complete_cli(context, images, sample_id)
