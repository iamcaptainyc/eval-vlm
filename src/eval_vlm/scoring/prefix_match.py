"""prefix_match scorer:只比较归一化后**前 k 个字符**是否完全相同。

逻辑与 exact_match 类似(归一化 + 二值命中),但只看开头 k 个字符——
适合「轮2标签」这类答案:标准答案与模型输出往往在前几个字就分出对错,
后续多余的解释/标点不应影响判分。

用法:在 scoring.scorer / turn_scorers / --scorer 里用 "prefix_match:K" 指定 k,
例如逐轮 `turn_scorers: [token_f1, prefix_match:10]`(轮1描述用 token_f1,
轮2标签比前 10 字)。不带后缀的 "prefix_match" 用默认 k=10。

归一化沿用 exact_match.normalize(NFKC、小写、标点->空格、压缩空白),
再各取前 k 个字符比较;某端不足 k 个字符时取其全部(此时退化为整串精确匹配)。
"""
from __future__ import annotations

from typing import Any, Optional

from ..data.schema import Sample
from .base import ScoreResult, Scorer
from .exact_match import normalize
from .registry import register

DEFAULT_K = 10


@register("prefix_match")
class PrefixMatchScorer(Scorer):
    def __init__(self, k: int = DEFAULT_K):
        if not isinstance(k, int) or k <= 0:
            raise ValueError(f"prefix_match 的 k 必须为正整数,收到 {k!r}")
        self.k = k

    @classmethod
    def from_spec(cls, spec: str, **kwargs: Any) -> "PrefixMatchScorer":
        try:
            k = int(spec)
        except (TypeError, ValueError):
            raise ValueError(
                f"prefix_match 参数须为正整数 k(如 'prefix_match:10'),收到 ':{spec}'"
            )
        return cls(k=k, **kwargs)

    def score_one(
        self, prediction: str, reference: Optional[str], sample: Sample
    ) -> ScoreResult:
        if reference is None:
            return ScoreResult(id=sample.id, score=0.0,
                               detail={"skipped": True, "reason": "无 reference"})
        pred_prefix = normalize(prediction or "")[: self.k]
        ref_prefix = normalize(reference)[: self.k]
        match = 1.0 if pred_prefix == ref_prefix else 0.0
        return ScoreResult(
            id=sample.id,
            score=match,
            detail={
                "prefix_match": match,
                "k": self.k,
                "prediction_prefix": pred_prefix,
                "reference_prefix": ref_prefix,
            },
        )

    def aggregate(self, results: list[ScoreResult]) -> dict:
        base = super().aggregate(results)
        base["accuracy"] = base.pop("mean_score")   # 二值命中,主指标即 accuracy
        base["prefix_k"] = self.k
        return base
