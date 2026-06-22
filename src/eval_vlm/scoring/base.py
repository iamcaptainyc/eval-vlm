"""Scorer 抽象基类与结果结构。"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

from ..data.schema import Sample


@dataclass
class ScoreResult:
    """单条样本的评分结果。"""
    id: str
    score: float                       # 主指标(0~1 或任意数值)
    detail: dict[str, Any] = field(default_factory=dict)


class Scorer(ABC):
    """评分器接口。

    score_one : 对单条 (prediction, reference) 打分。
    aggregate : 把逐条结果汇总成整体指标。
    """

    name: str = "base"

    @abstractmethod
    def score_one(
        self, prediction: str, reference: Optional[str], sample: Sample
    ) -> ScoreResult:
        raise NotImplementedError

    def aggregate(self, results: list[ScoreResult]) -> dict[str, Any]:
        """默认聚合:有效样本的平均主指标。"""
        valid = [r for r in results if r.detail.get("skipped") is not True]
        n = len(valid)
        mean = sum(r.score for r in valid) / n if n else 0.0
        return {
            "scorer": self.name,
            "num_total": len(results),
            "num_scored": n,
            "num_skipped": len(results) - n,
            "mean_score": round(mean, 4),
        }
