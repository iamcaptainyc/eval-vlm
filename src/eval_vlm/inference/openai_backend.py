"""OpenAI 兼容 API 推理后端。

模型由 vLLM / SGLang / LlamaFactory `api` 等部署成 HTTP 服务,
这里只通过 OpenAI 兼容接口调用,部署与评测完全分离。
"""
from __future__ import annotations

import base64
import mimetypes
import os
import time
from pathlib import Path
from typing import Any, Optional

from ..config import Config
from ..data.loader import resolve_image_path
from ..data.schema import Prediction, Turn
from .base import InferenceBackend


def _image_to_data_url(path: Path) -> str:
    """本地图片 -> base64 data URL。"""
    mime, _ = mimetypes.guess_type(str(path))
    mime = mime or "image/jpeg"
    with path.open("rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _resolve_image_url(img: str, cfg: Config) -> str:
    """把样本中的图片引用转成可发送的 url。

    http/https/data URL 直接透传;本地路径(含可选前缀剥离)编码成 base64 data URL。
    """
    if img.startswith("http://") or img.startswith("https://") or img.startswith("data:"):
        return img
    p = resolve_image_path(img, cfg)
    if not p.exists():
        raise FileNotFoundError(f"图片不存在: {p}(原始引用: {img})")
    return _image_to_data_url(p)


class OpenAIBackend(InferenceBackend):
    def __init__(self, cfg: Config) -> None:
        super().__init__(cfg)
        try:
            from openai import OpenAI
        except ImportError as e:  # pragma: no cover
            raise ImportError("需要安装 openai: pip install openai") from e

        ic = cfg.inference
        api_key = os.environ.get(ic.api_key_env) or "EMPTY"
        self.client = OpenAI(
            base_url=ic.base_url,
            api_key=api_key,
            timeout=ic.request_timeout,
            max_retries=0,  # 重试由本类自己控制(带退避)
        )

    def _build_messages(
        self, context: list[Turn], images: list[str], sample_id: str
    ) -> list[dict[str, Any]]:
        ic = self.cfg.inference
        messages: list[dict[str, Any]] = []
        if ic.system_prompt:
            messages.append({"role": "system", "content": ic.system_prompt})

        # 图片按出现顺序消费;<image> 占位符替换为图片块。
        img_queue = list(images)
        for turn in context:
            if turn.role != "user":
                # 多轮对话里历史 assistant 轮原样带上(纯文本)。
                messages.append({"role": turn.role, "content": turn.content})
                continue
            content_parts: list[dict[str, Any]] = []
            segments = turn.content.split("<image>")
            for si, seg in enumerate(segments):
                if seg:
                    content_parts.append({"type": "text", "text": seg})
                if si < len(segments) - 1:  # 段间有一个 <image>
                    if not img_queue:
                        raise ValueError(
                            f"样本 {sample_id} 的 <image> 占位符多于图片数"
                        )
                    url = _resolve_image_url(img_queue.pop(0), self.cfg)
                    content_parts.append({
                        "type": "image_url",
                        "image_url": {"url": url, "detail": ic.image_detail},
                    })
            if not content_parts:
                content_parts.append({"type": "text", "text": ""})
            messages.append({"role": "user", "content": content_parts})
        return messages

    def complete(
        self,
        context: list[Turn],
        images: list[str],
        sample_id: str,
        expected: Optional[str] = None,
    ) -> Prediction:
        ic = self.cfg.inference
        start = time.time()
        last_err: Exception | None = None
        try:
            messages = self._build_messages(context, images, sample_id)
        except Exception as e:  # 构造阶段失败(如缺图)
            return Prediction(id=sample_id, error=f"build_messages: {e}")

        for attempt in range(ic.max_retries + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=ic.model,
                    messages=messages,
                    max_tokens=ic.max_tokens,
                    temperature=ic.temperature,
                )
                text = resp.choices[0].message.content or ""
                return Prediction(
                    id=sample_id,
                    prediction=text,
                    latency=round(time.time() - start, 3),
                    raw={"model": getattr(resp, "model", ic.model),
                         "finish_reason": resp.choices[0].finish_reason},
                )
            except Exception as e:  # noqa: BLE001 - 捕获所有以便重试/记录
                last_err = e
                if attempt < ic.max_retries:
                    time.sleep(min(2 ** attempt, 10))
        return Prediction(
            id=sample_id,
            latency=round(time.time() - start, 3),
            error=f"{type(last_err).__name__}: {last_err}",
        )
