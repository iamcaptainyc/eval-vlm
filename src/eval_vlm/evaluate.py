"""评分步骤:读 predictions.jsonl + test.json -> 逐轮打分 -> 落盘。

把每个**目标轮**的预测和标准答案按 (id, turn) 对齐,套用可插拔 scorer。
不同轮可用不同 scorer(scoring.turn_scorers,按目标顺序),缺省回落 scoring.scorer。
产出 metrics.json(含 per_turn 分组指标)/ scored.jsonl / summary.md。
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from typing import Optional

from .config import Config
from .data.loader import load_samples
from .results import store
from .scoring import Scorer, get_scorer


def _scorer_for(ordinal: int, default_name: str, turn_names: list[str],
                cache: dict[str, Scorer]) -> tuple[str, Scorer]:
    """取第 ordinal 个目标轮该用的 scorer(缺省回落 default)。"""
    name = turn_names[ordinal] if ordinal < len(turn_names) else default_name
    if name not in cache:
        cache[name] = get_scorer(name)
    return name, cache[name]


def _is_exact_match_miss(row: dict) -> bool:
    """该目标轮是否为 exact_match 未命中。

    仅当该轮**用 exact_match 评分**(detail 含 exact_match)且不为满分时算未命中。
    缺失预测 / 推理报错时,exact_match 会被算成 0.0,同样计入。
    非 exact_match 评分(如 token_f1)一律不计入本清单。
    """
    detail = row["detail"]
    if "exact_match" not in detail:
        return False
    return float(detail["exact_match"]) != 1.0


def score_predictions(cfg: Config, scorer_name: Optional[str] = None) -> dict:
    """对已有预测逐轮评分,返回聚合指标。"""
    default_name = scorer_name or cfg.scoring.scorer
    turn_names = list(cfg.scoring.turn_scorers or [])
    cache: dict[str, Scorer] = {}

    if not cfg.test_path.exists():
        raise FileNotFoundError(
            f"未找到测试集 {cfg.test_path},请先运行: python -m eval_vlm split"
        )
    samples = load_samples(cfg, source=cfg.test_path)

    if not cfg.predictions_path.exists():
        raise FileNotFoundError(
            f"未找到预测文件 {cfg.predictions_path},请先运行: python -m eval_vlm run"
        )
    preds = store.load_predictions(cfg.predictions_path)
    pred_by_key = {(p.id, p.turn): p for p in preds}

    # 按目标序号(ordinal)分组,每组用各自 scorer 聚合。
    groups: dict[int, list] = defaultdict(list)
    group_scorer: dict[int, str] = {}
    scored_rows = []
    all_scores: list[float] = []

    for sample in samples:
        for ordinal, target in enumerate(sample.targets):
            name, scorer = _scorer_for(ordinal, default_name, turn_names, cache)
            group_scorer[ordinal] = name
            pred = pred_by_key.get((sample.id, target.turn_index))
            if pred is None:
                res = scorer.score_one("", target.reference, sample)
                res.detail["missing_prediction"] = True
                res.score = 0.0
            elif pred.error:
                res = scorer.score_one("", target.reference, sample)
                res.detail["inference_error"] = pred.error
                res.score = 0.0
            else:
                res = scorer.score_one(pred.prediction, target.reference, sample)
            groups[ordinal].append(res)
            all_scores.append(res.score)
            # 原图地址:优先用预测里随样本落盘的,回落到 test.json 的样本字段。
            images = list(pred.images) if (pred and pred.images) else list(sample.images)
            row = {
                "id": sample.id,
                "turn": target.turn_index,
                "ordinal": ordinal,
                "scorer": name,
                "score": res.score,
                "prediction": pred.prediction if pred else None,
                "reference": target.reference,
                "images": images,
                "detail": res.detail,
            }
            scored_rows.append(row)

    per_turn = {}
    for ordinal in sorted(groups):
        name = group_scorer[ordinal]
        per_turn[f"turn_{ordinal}"] = cache[name].aggregate(groups[ordinal])

    # 未命中以 id 为单位:某 id 任一 exact_match 轮错了,整个样本进清单(列出全部轮)。
    sample_by_id = {s.id: s for s in samples}
    rows_by_id: dict[str, list] = defaultdict(list)
    for row in scored_rows:
        rows_by_id[row["id"]].append(row)
    failed_ids = [s.id for s in samples
                  if any(_is_exact_match_miss(r) for r in rows_by_id[s.id])]
    num_failed_targets = sum(1 for r in scored_rows if _is_exact_match_miss(r))

    metrics = {
        "run_name": cfg.run_name,
        "model": cfg.inference.result_name,
        "scored_at": datetime.now(timezone.utc).isoformat(),
        "eval_targets": cfg.eval.targets,
        "eval_context": cfg.eval.context,
        "num_samples": len(samples),
        "num_targets": sum(len(s.targets) for s in samples),
        "overall_mean_score": round(sum(all_scores) / len(all_scores), 4) if all_scores else 0.0,
        "num_failed_samples": len(failed_ids),     # exact_match 未命中的样本(id)数
        "num_failed_targets": num_failed_targets,  # 其中错误的目标轮数
        "failures_path": str(cfg.failures_path),
        "per_turn": per_turn,
    }

    store.write_json(cfg.metrics_path, metrics)
    store.write_jsonl(cfg.scored_path, scored_rows)
    # 人类可读、按 id 分组的未命中清单(供人工审核);机器可读逐轮数据见 scored.jsonl。
    store.write_text(cfg.failures_path,
                     _render_failures_md(failed_ids, sample_by_id, rows_by_id, metrics))
    store.write_text(cfg.summary_path, _render_summary(metrics))
    return metrics


def _render_summary(metrics: dict) -> str:
    lines = [
        f"# 评测摘要 — {metrics.get('run_name', '')}",
        "",
        f"- 模型: `{metrics.get('model', '')}`",
        f"- 评测目标: `{metrics.get('eval_targets', '')}`  上下文: `{metrics.get('eval_context', '')}`",
        f"- 样本数: {metrics.get('num_samples', 0)}  目标轮数: {metrics.get('num_targets', 0)}",
        f"- 总体均分: {metrics.get('overall_mean_score', 0.0)}",
        f"- exact_match 未命中: {metrics.get('num_failed_samples', 0)} 个样本 / "
        f"{metrics.get('num_failed_targets', 0)} 个目标轮 -> `{metrics.get('failures_path', '')}`",
        f"- 评分时间: {metrics.get('scored_at', '')}",
        "",
        "## 逐轮指标",
    ]
    for turn_key, agg in (metrics.get("per_turn") or {}).items():
        lines.append("")
        lines.append(f"### {turn_key}  (scorer: `{agg.get('scorer', '')}`)")
        lines.append("")
        lines.append("| 指标 | 值 |")
        lines.append("| --- | --- |")
        for k, v in agg.items():
            if k == "scorer":
                continue
            lines.append(f"| {k} | {v} |")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 未命中清单(人类可读,按 id 分组)
# ---------------------------------------------------------------------------
def _fence(text) -> str:
    """把任意文本包进围栏代码块,保留换行、避免 markdown 误解析。"""
    s = "" if text is None else str(text)
    return "```\n" + s + "\n```"


def _render_failures_md(failed_ids: list, sample_by_id: dict,
                        rows_by_id: dict, metrics: dict) -> str:
    """渲染人类可读的未命中清单:仅 exact_match 错误样本,按 id 分组列全部对话轮。"""
    head = [
        f"# 未命中清单(exact_match)— {metrics.get('run_name', '')}",
        "",
        f"- 模型: `{metrics.get('model', '')}`",
        f"- 未命中样本: {metrics.get('num_failed_samples', 0)} / "
        f"{metrics.get('num_samples', 0)}(涉及 {metrics.get('num_failed_targets', 0)} 个错误目标轮)",
        f"- 评分时间: {metrics.get('scored_at', '')}",
        "",
        "> 仅纳入 **exact_match** 评分错误的样本;每个样本列出其全部对话轮便于核查。",
        "> 非 exact_match 评分(如 token_f1)不计入本清单。",
        "",
    ]
    if not failed_ids:
        head.append("✅ 无 exact_match 未命中。")
        head.append("")
        return "\n".join(head)

    blocks = ["---", ""]
    for sid in failed_ids:
        blocks.extend(_render_one_failure(sid, sample_by_id.get(sid), rows_by_id.get(sid, [])))
    return "\n".join(head + blocks)


def _render_one_failure(sid: str, sample, rows: list) -> list:
    """单个失败样本:标题(命中比例)+ 图片/元信息 + 按对话顺序展开全部轮。"""
    row_by_turn = {r["turn"]: r for r in rows}
    n_miss = sum(1 for r in rows if _is_exact_match_miss(r))
    n_em = sum(1 for r in rows if "exact_match" in r["detail"])
    out = [f"## 样本 `{sid}`  ✗ exact_match 未命中 {n_miss}/{n_em} 轮"]

    # 图片地址(可追溯回原图);优先用落盘行里的,回落到样本。
    imgs: list = []
    for r in rows:
        for im in r.get("images") or []:
            if im not in imgs:
                imgs.append(im)
    if not imgs and sample:
        imgs = list(sample.images)
    if imgs:
        out.append("- 图片: " + ", ".join(f"`{i}`" for i in imgs))
    if sample and sample.meta:
        out.append("- 元信息: " + ", ".join(f"{k}={v}" for k, v in sample.meta.items()))
    out.append("")

    turns = sample.turns if sample else []
    if turns:
        for idx, turn in enumerate(turns):
            out.extend(_render_turn(idx, turn, row_by_turn.get(idx)))
    else:                                    # 退化:无完整对话时只列目标轮
        for r in sorted(rows, key=lambda r: r["turn"]):
            out.extend(_render_turn(r["turn"], None, r))
    out.append("")
    return out


def _render_turn(idx: int, turn, row) -> list:
    """渲染一轮:非目标轮原样展示;目标轮展示模型输出 vs 标准答案 + 命中标记。"""
    if row is None:                          # 非目标轮:对话上下文
        role = getattr(turn, "role", "?")
        label = "user" if role == "user" else "assistant(标准上下文)"
        return [f"### 轮 {idx} · {label}", _fence(getattr(turn, "content", "")), ""]

    detail = row["detail"]
    is_em = "exact_match" in detail
    if is_em:
        mark = "✗ 未命中" if _is_exact_match_miss(row) else "✓ 命中"
    else:
        mark = f"(score: {row.get('score')})"
    lines = [f"### 轮 {idx} · assistant(目标 · scorer: `{row.get('scorer')}`)  {mark}"]
    if detail.get("missing_prediction"):
        lines.append("- ⚠️ 缺失预测(模型未产出该轮)")
    elif detail.get("inference_error"):
        lines.append(f"- ⚠️ 推理报错: {detail.get('inference_error')}")
    lines.append("- 模型输出:")
    lines.append(_fence(row.get("prediction")))
    lines.append("- 标准答案:")
    lines.append(_fence(row.get("reference")))
    lines.append("")
    return lines
