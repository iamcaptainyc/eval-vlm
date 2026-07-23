"""scorer 名称 -> 类 的注册表。"""
from __future__ import annotations

from typing import Callable, Type

from .base import Scorer

_REGISTRY: dict[str, Type[Scorer]] = {}


def register(name: str) -> Callable[[Type[Scorer]], Type[Scorer]]:
    """类装饰器:把 Scorer 子类登记到注册表。"""
    def deco(cls: Type[Scorer]) -> Type[Scorer]:
        if name in _REGISTRY:
            raise ValueError(f"scorer 名称重复: {name!r}")
        cls.name = name
        _REGISTRY[name] = cls
        return cls
    return deco


def get_scorer(name: str, **kwargs) -> Scorer:
    """按名称构造 scorer。

    支持 "base:spec" 形式的参数后缀(如 "prefix_match:10"):冒号前查注册表,
    冒号后交给该 scorer 的 from_spec 解析。无后缀则直接实例化(向后兼容)。
    """
    base, sep, spec = name.partition(":")
    if base not in _REGISTRY:
        raise ValueError(
            f"未知 scorer: {name!r}。可用: {', '.join(sorted(_REGISTRY)) or '(空)'}"
        )
    cls = _REGISTRY[base]
    if sep:
        return cls.from_spec(spec, **kwargs)
    return cls(**kwargs)


def available_scorers() -> list[str]:
    return sorted(_REGISTRY)
