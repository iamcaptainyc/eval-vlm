"""跨格式质量对比:从两份 scored.jsonl 交叉出「净质量Δ」。

用途:量化门禁工具链里,把候选(如 MNN)与参考(如 HF)两个格式**各自 vs gold 的
逐轮得分**按 (id, turn) 交叉,得出「HF 对了 / MNN 错了」这类净回归统计。

**彻底解耦**:只读两份已生成的 scored.jsonl(由 `score`/`eval` 产出),不重跑 scorer、
不加载 test.json。precision 命令与 report 命令共用本模块。

「对/错」定义:二值 scorer(exact_match / prefix_match,score∈{0,1})按 score==1.0 判命中;
连续 scorer(如 token_f1)不做 ✓/✗ 交叉,改记两端 mean-score 差(标 continuous)。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

# 二值(命中/未命中)scorer 的基名;其余(如 token_f1)视为连续。
_BINARY_SCORERS = {"exact_match", "prefix_match"}
_CORRECT_EPS = 1e-9


def _scorer_base(name: Optional[str]) -> str:
    """去掉 "prefix_match:10" 这类参数后缀,取基名。"""
    return (name or "").partition(":")[0]


def _is_binary(scorer: Optional[str]) -> bool:
    return _scorer_base(scorer) in _BINARY_SCORERS


def _is_correct(score) -> bool:
    try:
        return float(score) >= 1.0 - _CORRECT_EPS
    except (TypeError, ValueError):
        return False


def load_scored(path: Path) -> dict[tuple[str, int], dict]:
    """读 scored.jsonl -> {(id, turn): row}。缺文件返回空 dict。

    row 为 evaluate.py 落盘的逐(样本,轮)记录,含 id/turn/scorer/score/detail 等。
    同 (id,turn) 多行时后写覆盖先写(与 predictions 读取语义一致)。
    """
    out: dict[tuple[str, int], dict] = {}
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "id" not in obj:
                continue
            out[(str(obj["id"]), int(obj.get("turn", -1)))] = obj
    return out


def _blank_binary() -> dict:
    return {"num": 0, "both_correct": 0, "cand_wrong_ref_correct": 0,
            "cand_correct_ref_wrong": 0, "both_wrong": 0}


def _blank_continuous() -> dict:
    return {"num": 0, "_sum_cand": 0.0, "_sum_ref": 0.0}


def quality_crosstab(cand_scored: dict[tuple[str, int], dict],
                     ref_scored: dict[tuple[str, int], dict]) -> dict:
    """交叉候选/参考两份 scored,产出净质量Δ 汇总(总体 + 逐轮)。

    候选=待评格式(如 MNN),参考=基准格式(如 HF)。net_regression_rate 指
    「参考对了但候选错了」在二值样本里的占比 —— 即转换/量化带来的净质量回归。
    """
    common = sorted(set(cand_scored) & set(ref_scored))
    binary = _blank_binary()
    continuous = _blank_continuous()
    per_turn: dict[int, dict] = {}

    for key in common:
        turn = key[1]
        cand, ref = cand_scored[key], ref_scored[key]
        # 该轮是否二值:以候选行的 scorer 为准(两端同轮通常同 scorer)。
        binary_turn = _is_binary(cand.get("scorer")) and _is_binary(ref.get("scorer"))
        pt = per_turn.setdefault(turn, {"binary": _blank_binary(),
                                        "continuous": _blank_continuous(),
                                        "scorer": _scorer_base(cand.get("scorer"))})
        if binary_turn:
            cc, rc = _is_correct(cand.get("score")), _is_correct(ref.get("score"))
            for bucket in (binary, pt["binary"]):
                bucket["num"] += 1
                if cc and rc:
                    bucket["both_correct"] += 1
                elif not cc and rc:
                    bucket["cand_wrong_ref_correct"] += 1     # 净回归(HF对 MNN错)
                elif cc and not rc:
                    bucket["cand_correct_ref_wrong"] += 1     # 净改进
                else:
                    bucket["both_wrong"] += 1
        else:
            try:
                sc, sr = float(cand.get("score", 0.0)), float(ref.get("score", 0.0))
            except (TypeError, ValueError):
                sc, sr = 0.0, 0.0
            for bucket in (continuous, pt["continuous"]):
                bucket["num"] += 1
                bucket["_sum_cand"] += sc
                bucket["_sum_ref"] += sr

    return {
        "num_common": len(common),
        "num_only_candidate": len(set(cand_scored) - set(ref_scored)),
        "num_only_reference": len(set(ref_scored) - set(cand_scored)),
        "binary": _finalize_binary(binary),
        "continuous": _finalize_continuous(continuous),
        "per_turn": {t: {"scorer": pt["scorer"],
                         "binary": _finalize_binary(pt["binary"]),
                         "continuous": _finalize_continuous(pt["continuous"])}
                     for t, pt in sorted(per_turn.items())},
    }


def _finalize_binary(b: dict) -> dict:
    n = b["num"]
    b = dict(b)
    b["net_regression_rate"] = round(b["cand_wrong_ref_correct"] / n, 4) if n else 0.0
    b["net_improvement_rate"] = round(b["cand_correct_ref_wrong"] / n, 4) if n else 0.0
    return b


def _finalize_continuous(c: dict) -> dict:
    n = c["num"]
    mean_cand = c["_sum_cand"] / n if n else 0.0
    mean_ref = c["_sum_ref"] / n if n else 0.0
    return {
        "num": n,
        "mean_score_candidate": round(mean_cand, 4),
        "mean_score_reference": round(mean_ref, 4),
        "mean_score_delta": round(mean_cand - mean_ref, 4),   # 负=候选更差
    }
