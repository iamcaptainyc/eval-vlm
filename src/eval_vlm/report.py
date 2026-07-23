"""report 命令:跨格式合并质量报告(HF vs 各 MNN 变体)。

把一个数据集下**所有已跑过的格式**(<数据集>/<模型>/<后端>/)的产物汇成一页:
  - 绝对质量并排(A 轴):各格式各自 vs gold 的总分 + 逐轮主指标 + Δ vs HF 基准;
  - 净质量Δ(B 轴):候选(MNN)相对 HF 的净回归/改进(读两端 scored.jsonl 交叉);
  - 行为保真(B/C 轴):引用各 MNN 目录已生成的 precision.json(若有);
  - 诊断结论:规则式,把掉分归因到「预处理不对齐 / 疑量化损失 / 未跑 precision」。

**与当前 backend 解耦、纯读取**:只读已落盘产物(metrics.json/scored.jsonl/precision.json/
run_meta.json/pred_meta.json),不跑任何模型,也不管 inference.backend 指向谁。
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from . import compare
from .config import Config
from .results import store

# 总分相对 HF 基准的相对跌幅超过此值,才算「掉分」(否则视为持平)。
_DROP_EPS = 0.01
# precision flags 里指示「预处理/输入不对齐」的关键词。
_ALIGN_KEYWORDS = ("预处理", "不对齐", "不一致")


def _read_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _primary_metric(agg: dict) -> Optional[float]:
    """逐轮 aggregate 的主指标(键名不统一:accuracy/f1/mean_score 按优先级探)。"""
    for k in ("accuracy", "f1", "mean_score"):
        if k in agg:
            return agg[k]
    return None


def _quant_of(meta: Optional[dict]) -> Optional[str]:
    return meta.get("quant") if meta else None


def _label(model: str, backend: str, quant: Optional[str]) -> str:
    return f"{model}/{backend}" + (f" [{quant}]" if quant else "")


# ---------------------------------------------------------------------------
# 汇聚
# ---------------------------------------------------------------------------
def build_report(cfg: Config) -> dict:
    """扫描数据集下全部格式产物,汇成合并报告 dict。"""
    formats: list[dict] = []
    for model, backend, run_dir in store.discover_run_dirs(cfg.dataset_dir):
        metrics = _read_json(run_dir / "metrics.json")
        meta = _read_json(run_dir / "run_meta.json") or _read_json(run_dir / "pred_meta.json")
        precision = _read_json(run_dir / "precision.json")
        quant = _quant_of(meta)
        per_turn = {}
        if metrics:
            for tk, agg in (metrics.get("per_turn") or {}).items():
                per_turn[tk] = {"scorer": agg.get("scorer"), "primary": _primary_metric(agg)}
        formats.append({
            "model": model, "backend": backend, "quant": quant,
            "label": _label(model, backend, quant),
            "dir": str(run_dir),
            "scored_path": str(run_dir / "scored.jsonl"),
            "has_metrics": metrics is not None,
            "overall_mean_score": metrics.get("overall_mean_score") if metrics else None,
            "num_samples": metrics.get("num_samples") if metrics else None,
            "per_turn": per_turn,
            "precision": precision,
        })

    baseline = next((f for f in formats if f["backend"] == "hf"), None)
    net_quality = _net_quality_vs_baseline(baseline, formats)
    diagnosis = _diagnose(baseline, formats)

    return {
        "dataset": cfg.dataset_dir.name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "baseline": ({"model": baseline["model"], "backend": baseline["backend"]}
                     if baseline else None),
        "num_formats": len(formats),
        "formats": formats,
        "net_quality": net_quality,
        "diagnosis": diagnosis,
    }


def _net_quality_vs_baseline(baseline: Optional[dict], formats: list[dict]) -> list[dict]:
    """每个非基准格式 vs HF 基准的净质量Δ(需两端 scored.jsonl 都在)。"""
    out: list[dict] = []
    if baseline is None:
        return out
    ref_scored = compare.load_scored(Path(baseline["scored_path"]))
    for f in formats:
        if f is baseline:
            continue
        if not ref_scored or not Path(f["scored_path"]).exists():
            out.append({"label": f["label"], "available": False,
                        "reason": "两端需都跑过 score(scored.jsonl)才能算净质量Δ。"})
            continue
        ct = compare.quality_crosstab(compare.load_scored(Path(f["scored_path"])), ref_scored)
        ct["available"] = True
        ct["label"] = f["label"]
        out.append(ct)
    return out


def _has_alignment_flag(precision: Optional[dict]) -> bool:
    if not precision:
        return False
    return any(any(kw in fl for kw in _ALIGN_KEYWORDS) for fl in precision.get("flags", []))


def _diagnose(baseline: Optional[dict], formats: list[dict]) -> list[str]:
    """规则式诊断:把每个格式相对 HF 的掉分归因。"""
    diag: list[str] = []
    if baseline is None:
        diag.append("⚠️ 未发现 HF 基准(backend=hf 的产物):只能列各格式绝对质量,"
                    "无法给出「转换损失」结论。请先 `eval-vlm eval --backend hf --hf-model <目录>`。")
        return diag
    base_score = baseline.get("overall_mean_score")
    if base_score is None:
        diag.append("⚠️ HF 基准无 metrics.json(未评分):请先对 HF 端跑 score/eval。")
        return diag

    for f in formats:
        if f is baseline or not f["has_metrics"]:
            continue
        score = f.get("overall_mean_score")
        if score is None:
            continue
        drop = base_score - score
        rate = drop / base_score if base_score else 0.0
        if rate <= _DROP_EPS:
            diag.append(f"✅ {f['label']} 与 HF 基准质量基本持平(总分 {score} vs {base_score})。")
        elif _has_alignment_flag(f["precision"]):
            diag.append(f"🔴 {f['label']} 质量较 HF 掉 {rate:.1%}(总分 {score} vs {base_score}),"
                        f"且其 precision 报「预处理不对齐」→ 先修预处理再谈精度(见该目录 precision.md)。")
        elif f["precision"] is not None:
            diag.append(f"🟠 {f['label']} 质量较 HF 掉 {rate:.1%}(总分 {score} vs {base_score}),"
                        f"但预处理对齐正常 → 疑量化损失,建议 Phase 2 用 MNN modelCompare 逐层定位。")
        else:
            diag.append(f"🟠 {f['label']} 质量较 HF 掉 {rate:.1%}(总分 {score} vs {base_score});"
                        f"未跑 precision,建议 `eval-vlm precision` 定位是预处理还是量化。")
    return diag


# ---------------------------------------------------------------------------
# 渲染
# ---------------------------------------------------------------------------
def render_report_md(report: dict) -> str:
    lines = [
        f"# 质量门禁合并报告 — 数据集 `{report['dataset']}`",
        "",
        f"- 生成时间: {report['generated_at']}",
        f"- 已发现格式数: {report['num_formats']}",
    ]
    b = report.get("baseline")
    lines.append(f"- HF 基准: `{b['model']}/{b['backend']}`" if b else "- HF 基准: (无)")
    lines += ["", "## 诊断结论", ""]
    lines += [f"- {d}" for d in report["diagnosis"]] or ["- (无格式可诊断)"]

    # 绝对质量并排表(各格式各自 vs gold + Δ vs HF)。
    base_score = None
    if b:
        base_fmt = next((f for f in report["formats"]
                         if f["model"] == b["model"] and f["backend"] == b["backend"]), None)
        base_score = base_fmt.get("overall_mean_score") if base_fmt else None
    turn_keys = sorted({tk for f in report["formats"] for tk in f["per_turn"]})
    lines += ["", "## 绝对质量并排(各格式 vs gold)", "",
              "| 格式 | 样本 | 总分 | Δ vs HF | "
              + " | ".join(f"{tk}" for tk in turn_keys) + " |",
              "| --- | --- | --- | --- | " + " | ".join("---" for _ in turn_keys) + " |"]
    for f in report["formats"]:
        score = f.get("overall_mean_score")
        if f["backend"] == "hf":
            delta = "(基准)"
        elif score is not None and base_score:
            delta = f"{score - base_score:+.4f}"
        else:
            delta = "—"
        turn_cells = []
        for tk in turn_keys:
            pt = f["per_turn"].get(tk)
            turn_cells.append(f"{pt['primary']} ({pt['scorer']})" if pt else "—")
        lines.append(f"| {f['label']} | {f.get('num_samples') if f.get('num_samples') is not None else '—'} "
                     f"| {score if score is not None else '未评分'} | {delta} | "
                     + " | ".join(turn_cells) + " |")

    # 净质量Δ(候选 vs HF)。
    lines += ["", "## 净质量Δ(候选 vs HF,各自 vs gold)", ""]
    if not report["net_quality"]:
        lines.append("> 无非基准格式,或无 HF 基准。")
    for nq in report["net_quality"]:
        lines.append(f"### {nq['label']}")
        if not nq.get("available"):
            lines.append(f"> ⏭️ {nq.get('reason', '未评分,跳过。')}")
            lines.append("")
            continue
        qb, qc = nq.get("binary", {}), nq.get("continuous", {})
        if qb.get("num"):
            lines += [
                f"- 二值轮 {qb['num']} 对:双对 {qb['both_correct']}、"
                f"🔴 HF对/候选错 {qb['cand_wrong_ref_correct']}(净回归 {qb['net_regression_rate']:.1%})、"
                f"HF错/候选对 {qb['cand_correct_ref_wrong']}(净改进 {qb['net_improvement_rate']:.1%})、"
                f"双错 {qb['both_wrong']}。",
            ]
        if qc.get("num"):
            lines.append(f"- 连续轮 {qc['num']} 对:候选均分 {qc['mean_score_candidate']} vs "
                         f"HF {qc['mean_score_reference']}(Δ {qc['mean_score_delta']:+g})。")
        lines.append("")

    # 行为保真(引用 precision.json)。
    lines += ["## 行为保真(引用各 MNN 目录 precision)", "",
              "| 格式 | 输出一致率 | 平均 token-F1 | 告警 |",
              "| --- | --- | --- | --- |"]
    any_prec = False
    for f in report["formats"]:
        if f["backend"] == "hf":
            continue
        p = f["precision"]
        if not p:
            lines.append(f"| {f['label']} | 未跑 precision | — | — |")
            continue
        any_prec = True
        bh = p.get("behavior", {})
        n_flags = len([fl for fl in p.get("flags", []) if not fl.startswith("✅")])
        lines.append(f"| {f['label']} | {bh.get('agreement_rate')} | {bh.get('mean_token_f1')} "
                     f"| {n_flags} 条(见该目录 precision.md) |")
    if not any_prec:
        lines.append("")
        lines.append("> 尚无任一 MNN 格式跑过 `precision`;跑了才有行为级一致率与对齐审计。")
    lines.append("")
    return "\n".join(lines)
