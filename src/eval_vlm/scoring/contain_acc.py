"""contain_acc scorer:归一化后 pred **包含** ref 即算命中(子串命中制)。

适合「标准答案是短标签/关键值,模型输出常带前后解释」的场景:只要模型
把标准答案作为**子串**说出来了(不论前后有多少修饰),就算对。与相邻两个
二值 scorer 的区别:
  - exact_match  : 要求整串相等(本 scorer 放宽为「包含」);
  - prefix_match : 只看开头 K 个字符(本 scorer 不限位置,标签可在任意处)。

归一化沿用 exact_match.normalize(NFKC、小写、标点->空格、压缩空白),再判子串。
注意:归一化后 ref 为空串时不算命中(空串是任何串的子串,会误判满分),按 0 计
——与 exact_match 里 `contains` 的判定一致。reference 为 None 才跳过、不计入。
"""
from __future__ import annotations

from typing import Optional

from ..data.schema import Sample
from .base import ScoreResult, Scorer
from .exact_match import normalize
from .registry import register


@register("contain_acc")
class ContainAccScorer(Scorer):
    def score_one(
        self, prediction: str, reference: Optional[str], sample: Sample
    ) -> ScoreResult:
        if reference is None:
            return ScoreResult(id=sample.id, score=0.0,
                               detail={"skipped": True, "reason": "无 reference"})
        pred_n = normalize(prediction or "")
        ref_n = normalize(reference)
        contains = 1.0 if ref_n and ref_n in pred_n else 0.0
        return ScoreResult(
            id=sample.id,
            score=contains,
            detail={
                "contain_acc": contains,
                "prediction_norm": pred_n,
                "reference_norm": ref_n,
            },
        )

    def aggregate(self, results: list[ScoreResult]) -> dict:
        base = super().aggregate(results)
        base["accuracy"] = base.pop("mean_score")   # 二值命中,主指标即 accuracy
        return base
