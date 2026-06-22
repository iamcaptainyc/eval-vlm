"""默认 scorer:归一化精确匹配 + 子串命中。

适用于 VQA / 多选 / 短答案这类有标准答案的任务。
归一化:去首尾空白、小写、去标点、压缩内部空白。
"""
from __future__ import annotations

import re
import string
import unicodedata
from typing import Optional

from ..data.schema import Sample
from .base import ScoreResult, Scorer
from .registry import register

_PUNCT_TABLE = {ord(c): " " for c in string.punctuation}


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.strip().lower()
    text = text.translate(_PUNCT_TABLE)        # 标点 -> 空格
    text = re.sub(r"\s+", " ", text).strip()   # 压缩空白
    return text


@register("exact_match")
class ExactMatchScorer(Scorer):
    def score_one(
        self, prediction: str, reference: Optional[str], sample: Sample
    ) -> ScoreResult:
        if reference is None:
            return ScoreResult(id=sample.id, score=0.0,
                               detail={"skipped": True, "reason": "无 reference"})
        pred_n = normalize(prediction or "")
        ref_n = normalize(reference)
        exact = 1.0 if pred_n == ref_n else 0.0
        contains = 1.0 if ref_n and ref_n in pred_n else 0.0
        return ScoreResult(
            id=sample.id,
            score=exact,
            detail={
                "exact_match": exact,
                "contains": contains,
                "prediction_norm": pred_n,
                "reference_norm": ref_n,
            },
        )

    def aggregate(self, results: list[ScoreResult]) -> dict:
        base = super().aggregate(results)
        valid = [r for r in results if r.detail.get("skipped") is not True]
        n = len(valid)
        contains = sum(r.detail.get("contains", 0.0) for r in valid) / n if n else 0.0
        base["accuracy"] = base.pop("mean_score")  # exact_match 下主指标即 accuracy
        base["contains_rate"] = round(contains, 4)
        return base
