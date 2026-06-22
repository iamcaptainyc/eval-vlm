"""token_f1 scorer:字符级 P/R/F1,适合开放式回答(如轮1的图像描述)。

不引入额外依赖,字符级切分对中文友好(也兼容英文/数字)。
归一化:NFKC、小写、去标点、去全部空白后按字符计多重集重叠。
"""
from __future__ import annotations

import re
import string
import unicodedata
from collections import Counter
from typing import Optional

from ..data.schema import Sample
from .base import ScoreResult, Scorer
from .registry import register

_PUNCT_TABLE = {ord(c): " " for c in string.punctuation}


def _tokens(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text or "").lower()
    text = text.translate(_PUNCT_TABLE)
    text = re.sub(r"\s+", "", text)   # 去全部空白,按字符切
    return list(text)


def _prf(prediction: str, reference: str) -> tuple[float, float, float]:
    pred = Counter(_tokens(prediction))
    ref = Counter(_tokens(reference))
    if not pred and not ref:
        return 1.0, 1.0, 1.0
    overlap = sum((pred & ref).values())
    if overlap == 0:
        return 0.0, 0.0, 0.0
    precision = overlap / sum(pred.values())
    recall = overlap / sum(ref.values())
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


@register("token_f1")
class TokenF1Scorer(Scorer):
    def score_one(
        self, prediction: str, reference: Optional[str], sample: Sample
    ) -> ScoreResult:
        if reference is None:
            return ScoreResult(id=sample.id, score=0.0,
                               detail={"skipped": True, "reason": "无 reference"})
        precision, recall, f1 = _prf(prediction or "", reference)
        return ScoreResult(
            id=sample.id,
            score=f1,
            detail={"f1": round(f1, 4),
                    "precision": round(precision, 4),
                    "recall": round(recall, 4)},
        )

    def aggregate(self, results: list[ScoreResult]) -> dict:
        base = super().aggregate(results)
        valid = [r for r in results if r.detail.get("skipped") is not True]
        n = len(valid)
        mean_p = sum(r.detail.get("precision", 0.0) for r in valid) / n if n else 0.0
        mean_r = sum(r.detail.get("recall", 0.0) for r in valid) / n if n else 0.0
        base["f1"] = base.pop("mean_score")   # 主指标即平均 F1
        base["precision"] = round(mean_p, 4)
        base["recall"] = round(mean_r, 4)
        return base
