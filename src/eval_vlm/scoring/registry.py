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
    if name not in _REGISTRY:
        raise ValueError(
            f"未知 scorer: {name!r}。可用: {', '.join(sorted(_REGISTRY)) or '(空)'}"
        )
    return _REGISTRY[name](**kwargs)


def available_scorers() -> list[str]:
    return sorted(_REGISTRY)
