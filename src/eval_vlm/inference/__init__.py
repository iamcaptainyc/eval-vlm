"""推理后端:把 Sample 送进模型拿到 Prediction。后端可插拔。"""
from __future__ import annotations

from ..config import Config
from .base import InferenceBackend


def build_backend(cfg: Config) -> InferenceBackend:
    """根据 config.inference.backend 构造后端实例。"""
    name = cfg.inference.backend
    if name == "openai":
        from .openai_backend import OpenAIBackend
        return OpenAIBackend(cfg)
    if name == "fake":
        from .fake_backend import FakeBackend
        return FakeBackend(cfg)
    raise ValueError(f"未知推理后端: {name!r}(可选: openai, fake)")
