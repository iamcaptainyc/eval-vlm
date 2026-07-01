"""推理后端:把 Sample 送进模型拿到 Prediction。后端可插拔。"""
from __future__ import annotations

from ..config import Config
from .base import InferenceBackend


def build_backend(cfg: Config) -> InferenceBackend:
    """根据 config.inference.backend 构造后端实例。

    可选:
      openai / vllm — OpenAI 兼容 HTTP API(vLLM/SGLang/LlamaFactory 部署);vllm 是 openai 的别名。
      mnn           — 本地 MNN(pymnn)推理:训练后转 mnn 的模型,串行,无需起服务。
      cmnn          — 本地 MNN(C++ 原生库)**批量**推理:多实例线程池并行,功能同 mnn。
      fake          — 离线回显,自检/演示用。
    """
    name = cfg.inference.backend
    if name in ("openai", "vllm"):
        from .openai_backend import OpenAIBackend
        return OpenAIBackend(cfg)
    if name == "mnn":
        from .mnn_backend import MNNBackend
        return MNNBackend(cfg)
    if name == "cmnn":
        from .cmnn_backend import CMNNBackend
        return CMNNBackend(cfg)
    if name == "fake":
        from .fake_backend import FakeBackend
        return FakeBackend(cfg)
    raise ValueError(f"未知推理后端: {name!r}(可选: openai, vllm, mnn, cmnn, fake)")


def worker_count(backend: InferenceBackend, max_concurrency: int) -> int:
    """并发编排层据后端线程安全性决定线程数。

    线程安全后端用配置的 max_concurrency(至少 1);非线程安全后端(如 MNN
    有状态单对象)强制串行返回 1,避免 KV cache / context 被并发破坏。
    """
    if not getattr(backend, "thread_safe", True):
        return 1
    return max(1, max_concurrency)
