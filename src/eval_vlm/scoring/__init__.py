"""可插拔评分模块。

新增一个 scorer = 新建一个文件 + @register("名字"),核心无需改动。
import 时确保内置 scorer 完成注册。
"""
from __future__ import annotations

from .base import Scorer, ScoreResult
from .registry import register, get_scorer, available_scorers

# 触发内置 scorer 注册
from . import exact_match  # noqa: F401,E402
from . import token_f1     # noqa: F401,E402

__all__ = ["Scorer", "ScoreResult", "register", "get_scorer", "available_scorers"]
