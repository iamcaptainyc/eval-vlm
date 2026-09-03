"""评分步骤:读 predictions.jsonl + test.json -> 逐轮打分 -> 落盘。

把每个**目标轮**的预测和标准答案按 (id, turn) 对齐,套用可插拔 scorer。
不同轮可用不同 scorer(scoring.turn_scorers,按目标顺序),缺省回落 scoring.scorer。
产出 metrics.json(含 per_turn 分组指标)/ scored.jsonl / summary.md。
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import html
from typing import Optional

from .config import Config
from .data.loader import load_samples
from .report_assets import batch_preload_images, image_ref_to_html_src
from .results import store
from .scoring import Scorer, get_scorer
from .scoring.confusion_matrix import (
    format_confusion_matrix_html,
    format_confusion_matrix_markdown,
)


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
        "failures_html_path": str(cfg.failures_html_path),
        "per_turn": per_turn,
    }

    store.write_json(cfg.metrics_path, metrics)
    store.write_jsonl(cfg.scored_path, scored_rows)
    # 人类可读、按 id 分组的未命中清单(供人工审核);机器可读逐轮数据见 scored.jsonl。
    store.write_text(cfg.failures_path,
                     _render_failures_md(failed_ids, sample_by_id, rows_by_id, metrics))
    store.write_text(cfg.failures_html_path,
                     _render_failures_html(failed_ids, sample_by_id, rows_by_id, metrics, cfg))
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
            if k in ("scorer", "confusion_matrix"):
                continue
            lines.append(f"| {k} | {v} |")
        if agg.get("confusion_matrix"):
            lines.append("")
            lines.append(format_confusion_matrix_markdown(agg["confusion_matrix"]))
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


# ---------------------------------------------------------------------------
# 未命中清单 HTML 可视化渲染(含图片 base64 内嵌、灯箱放大、搜索过滤)
# ---------------------------------------------------------------------------
def _html_escape(s: object) -> str:
    """统一转字符串再转义。"""
    return html.escape(str(s) if s is not None else "", quote=True)


def _render_failure_card(sid: str, sample, rows: list, cfg: Config) -> str:
    """渲染单个未命中样本卡片:图片、元数据、多轮对话、目标轮预测对比。"""
    row_by_turn = {r["turn"]: r for r in rows}
    n_miss = sum(1 for r in rows if _is_exact_match_miss(r))
    n_em = sum(1 for r in rows if "exact_match" in r["detail"])
    has_error = any(
        r.get("detail", {}).get("inference_error") or r.get("detail", {}).get("missing_prediction")
        for r in rows
    )

    imgs: list = []
    for r in rows:
        for im in (r.get("images") or []):
            if im not in imgs:
                imgs.append(im)
    if not imgs and sample:
        imgs = list(sample.images)

    imgs_html: list[str] = []
    for img in imgs:
        src, err = image_ref_to_html_src(img, cfg)
        if src is None:
            imgs_html.append(
                f'<figure class="img-item"><div class="img-placeholder" title="{_html_escape(err)}">'
                f'图片不可用</div><figcaption class="img-path">{_html_escape(img)}</figcaption></figure>'
            )
        else:
            imgs_html.append(
                f'<figure class="img-item"><img src="{src}" alt="{_html_escape(img)}" loading="lazy">'
                f'<figcaption class="img-path">{_html_escape(img)}</figcaption></figure>'
            )
    images_block = f'<div class="images">{"".join(imgs_html)}</div>' if imgs_html else ""

    meta_block = ""
    if sample and sample.meta:
        meta_items = [
            f"<span><code>{_html_escape(k)}</code>: {_html_escape(v)}</span>"
            for k, v in sample.meta.items()
        ]
        meta_block = f'<div class="meta-block"><strong>元信息:</strong> {" &nbsp; ".join(meta_items)}</div>'

    turns_html: list[str] = []
    turns = sample.turns if sample else []
    if turns:
        for idx, turn in enumerate(turns):
            turns_html.append(_render_turn_html(idx, turn, row_by_turn.get(idx)))
    else:
        for r in sorted(rows, key=lambda r: r["turn"]):
            turns_html.append(_render_turn_html(r["turn"], None, r))

    card_miss_turns = [str(r["turn"]) for r in rows if _is_exact_match_miss(r)]
    data_miss_turns = ",".join(card_miss_turns)
    card_cls = "card has-miss" if n_miss > 0 else "card"
    data_err = "true" if has_error else "false"
    miss_turn_label = f" (第 {', '.join(card_miss_turns)} 轮)" if card_miss_turns else ""
    return (
        f'<section class="{card_cls}" data-sample-id="{_html_escape(sid)}" data-has-error="{data_err}" data-miss-turns="{data_miss_turns}">'
        f'<div class="card-header">'
        f'<h3>样本 <code>{_html_escape(sid)}</code></h3>'
        f'<span class="tag-miss">✗ exact_match 未命中 {n_miss}/{n_em} 轮{miss_turn_label}</span>'
        f'</div>'
        f'{meta_block}'
        f'{images_block}'
        f'<div class="turns-flow">{"".join(turns_html)}</div>'
        f'</section>'
    )


def _render_turn_html(idx: int, turn, row) -> str:
    """渲染一轮对话的 HTML 结构。"""
    if row is None:
        role = getattr(turn, "role", "?")
        content = getattr(turn, "content", "")
        role_cls = "role-user" if role == "user" else "role-assistant"
        return (
            f'<div class="turn turn-context">'
            f'<div class="turn-title"><span class="role-badge {role_cls}">轮 {idx} · {role}</span></div>'
            f'<div class="turn-body">{_html_escape(content)}</div>'
            f'</div>'
        )

    detail = row["detail"]
    is_em = "exact_match" in detail
    miss = _is_exact_match_miss(row)
    if is_em:
        badge = (
            '<span class="badge badge-miss">✗ exact_match 未命中</span>'
            if miss
            else '<span class="badge badge-hit">✓ exact_match 命中</span>'
        )
    else:
        badge = f'<span class="badge badge-score">score: {row.get("score")}</span>'

    alert_html = ""
    if detail.get("missing_prediction"):
        alert_html = '<div class="alert alert-warning">⚠️ 缺失预测 (模型未产出该轮)</div>'
    elif detail.get("inference_error"):
        alert_html = f'<div class="alert alert-error">⚠️ 推理报错: {_html_escape(detail.get("inference_error"))}</div>'

    comp_cls = "comp-miss" if miss else "comp-hit"
    return (
        f'<div class="turn turn-target">'
        f'<div class="turn-title">'
        f'<span class="role-badge role-target">轮 {idx} · assistant (目标 · scorer: <code>{_html_escape(row.get("scorer"))}</code>)</span>'
        f'{badge}'
        f'</div>'
        f'{alert_html}'
        f'<div class="comparison-grid">'
        f'<div class="comp-col {comp_cls}">'
        f'<div class="comp-title">模型输出 (Prediction)</div>'
        f'<pre class="comp-content">{_html_escape(row.get("prediction"))}</pre>'
        f'</div>'
        f'<div class="comp-col comp-ref">'
        f'<div class="comp-title">标准答案 (Reference)</div>'
        f'<pre class="comp-content">{_html_escape(row.get("reference"))}</pre>'
        f'</div>'
        f'</div>'
        f'</div>'
    )


def _render_failures_html(
    failed_ids: list,
    sample_by_id: dict,
    rows_by_id: dict,
    metrics: dict,
    cfg: Config,
) -> str:
    """渲染未命中清单的 HTML 版（单文件自包含、图片 Base64、支持灯箱放大与搜索过滤）。"""
    title = f"未命中清单 (exact_match) — {cfg.run_name}"
    header = f"""<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>{_html_escape(title)}</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; margin: 24px; background: #f6f8fa; color: #24292f; }}
header.summary {{ margin-bottom: 20px; padding: 16px 20px; background: #fff; border: 1px solid #d0d7de; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.04); }}
header.summary h1 {{ margin: 0 0 8px 0; font-size: 20px; }}
header.summary p {{ margin: 4px 0; font-size: 14px; color: #57606a; }}
.card {{ background: #fff; border: 1px solid #d0d7de; border-radius: 8px; padding: 18px 20px; margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.04); }}
.card.has-miss {{ border-left: 5px solid #cf222e; }}
.card-header {{ display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px; }}
.card-header h3 {{ margin: 0; font-size: 16px; }}
.tag-miss {{ font-size: 13px; font-weight: 600; color: #cf222e; background: #ffebe9; padding: 2px 10px; border-radius: 12px; }}
.meta-block {{ margin-bottom: 12px; font-size: 13px; color: #57606a; background: #f6f8fa; padding: 8px 12px; border-radius: 6px; }}
.images {{ display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 16px; align-items: flex-start; }}
.img-item {{ margin: 0; display: flex; flex-direction: column; align-items: center; gap: 4px; }}
.img-item img {{ max-width: 100%; max-height: 560px; width: auto; object-fit: contain; border: 1px solid #d0d7de; border-radius: 6px; cursor: zoom-in; transition: transform .15s; }}
.img-item img:hover {{ transform: scale(1.01); }}
.img-path {{ font-size: 12px; color: #6e7781; word-break: break-all; max-width: 560px; text-align: center; }}
.img-placeholder {{ width: 560px; max-width: 100%; height: 120px; display: flex; align-items: center; justify-content: center; background: #f6f8fa; color: #8c959f; border: 1px dashed #d0d7de; font-size: 12px; text-align: center; padding: 8px; box-sizing: border-box; border-radius: 6px; }}
.turns-flow {{ display: flex; flex-direction: column; gap: 10px; }}
.turn {{ border-radius: 6px; padding: 12px; }}
.turn-context {{ background: #f6f8fa; border: 1px solid #eaeef2; }}
.turn-target {{ background: #ffffff; border: 1px solid #d0d7de; }}
.turn-title {{ display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px; font-size: 13px; }}
.turn-body {{ font-size: 14px; line-height: 1.6; white-space: pre-wrap; word-break: break-word; }}
.role-badge {{ display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 12px; font-weight: 600; }}
.role-user {{ background: #ddf4ff; color: #0969da; }}
.role-assistant {{ background: #fbefff; color: #8250df; }}
.role-target {{ background: #fff8c5; color: #9a6700; }}
.badge {{ font-size: 12px; font-weight: 600; padding: 2px 8px; border-radius: 12px; }}
.badge-miss {{ background: #ffebe9; color: #cf222e; }}
.badge-hit {{ background: #dafbe1; color: #1a7f37; }}
.badge-score {{ background: #f6f8fa; color: #57606a; }}
.alert {{ padding: 8px 12px; border-radius: 6px; font-size: 13px; margin-bottom: 8px; }}
.alert-warning {{ background: #fff8c5; color: #9a6700; border: 1px solid #d4a72c; }}
.alert-error {{ background: #ffebe9; color: #cf222e; border: 1px solid #ff8182; }}
.comparison-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 6px; }}
@media (max-width: 768px) {{ .comparison-grid {{ grid-template-columns: 1fr; }} }}
.comp-col {{ border-radius: 6px; padding: 10px 12px; }}
.comp-title {{ font-size: 12px; font-weight: 600; margin-bottom: 6px; text-transform: uppercase; letter-spacing: 0.5px; }}
.comp-miss {{ background: #fff5f5; border: 1px solid #ffcdd2; }}
.comp-miss .comp-title {{ color: #c62828; }}
.comp-hit {{ background: #f6ffed; border: 1px solid #b7eb8f; }}
.comp-hit .comp-title {{ color: #2e7d32; }}
.comp-ref {{ background: #f0f5ff; border: 1px solid #adc6ff; }}
.comp-ref .comp-title {{ color: #1d39c4; }}
.comp-content {{ margin: 0; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; font-size: 13px; line-height: 1.5; white-space: pre-wrap; word-break: break-word; }}
#filters {{ margin-bottom: 16px; display: flex; gap: 16px; align-items: center; flex-wrap: wrap; background: #fff; padding: 10px 16px; border: 1px solid #d0d7de; border-radius: 8px; }}
#filters label {{ font-size: 13px; cursor: pointer; display: flex; align-items: center; gap: 6px; color: #334155; }}
#flt-search {{ padding: 5px 10px; font-size: 13px; border: 1px solid #cbd5e1; border-radius: 6px; width: 200px; }}
#flt-turn {{ padding: 5px 10px; font-size: 13px; border: 1px solid #cbd5e1; border-radius: 6px; background: #fff; color: #1e293b; cursor: pointer; }}
.flt-count-badge {{ font-size: 12.5px; color: #64748b; margin-left: auto; font-weight: 600; }}
.empty-notice {{ padding: 20px; background: #dafbe1; color: #1a7f37; border-radius: 8px; font-size: 15px; font-weight: 500; }}
.cm-section {{ background: #fff; border: 1px solid #d0d7de; border-radius: 8px; padding: 16px 20px; margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.04); }}
.cm-section h3 {{ margin: 0 0 12px 0; font-size: 16px; color: #24292f; }}
.cm-table-wrapper, .cm-report-wrapper {{ overflow-x: auto; margin-bottom: 12px; }}
.cm-table, .cm-report-table {{ border-collapse: collapse; width: 100%; font-size: 13px; text-align: right; margin-bottom: 8px; }}
.cm-table th, .cm-table td, .cm-report-table th, .cm-report-table td {{ border: 1px solid #d0d7de; padding: 6px 10px; }}
.cm-table th, .cm-report-table th {{ background: #f6f8fa; color: #24292f; font-weight: 600; text-align: center; }}
.cm-table th.row-label {{ text-align: left; background: #f6f8fa; font-weight: 600; }}
.cm-report-table td.cat-name {{ text-align: left; font-weight: 600; }}
.cm-table td {{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; }}
.cm-diag {{ background: #dafbe1; color: #1a7f37; font-weight: 700; }}
.cm-zero {{ color: #8c959f; }}
.cm-total {{ background: #f6f8fa; font-weight: 600; color: #57606a; }}
.cm-total-all {{ background: #eaeef2; font-weight: 700; color: #24292f; }}
.lightbox {{ display: none; position: fixed; inset: 0; background: rgba(0,0,0,.85); z-index: 1000; align-items: center; justify-content: center; cursor: zoom-out; }}
.lightbox img {{ max-width: 96vw; max-height: 96vh; object-fit: contain; box-shadow: 0 0 24px rgba(0,0,0,.6); }}
</style></head><body>
"""
    header += f"""<header class="summary">
<h1>未命中清单 (exact_match) — {_html_escape(cfg.run_name)}</h1>
<p>模型: <code>{_html_escape(cfg.inference.result_name)}</code> &nbsp; 后端: <code>{_html_escape(cfg.inference.backend)}</code></p>
<p>未命中样本: <strong>{len(failed_ids)}</strong> / 已评 {metrics.get('num_samples', 0)} (涉及 {metrics.get('num_failed_targets', 0)} 个错误目标轮) &nbsp; 总体均分: {metrics.get('overall_mean_score', 0.0)}</p>
</header>
"""
    # 混淆矩阵展示 (若有)
    for turn_key, turn_data in (metrics.get("per_turn") or {}).items():
        if "confusion_matrix" in turn_data:
            header += format_confusion_matrix_html(
                turn_data["confusion_matrix"],
                title=f"混淆矩阵 — {turn_key} (scorer: {turn_data.get('scorer', '')})",
            )

    if not failed_ids:
        return header + '<p class="empty-notice">✅ 无 exact_match 未命中。</p></body></html>'

    # 1. 批量收集所有未命中样本的图片引用
    all_imgs: list[str] = []
    for sid in failed_ids:
        sample = sample_by_id.get(sid)
        for r in rows_by_id.get(sid, []):
            for im in (r.get("images") or []):
                all_imgs.append(im)
        if sample:
            for im in sample.images:
                all_imgs.append(im)

    # 2. 多线程并行预热解码并缓存图片(避免逐张串行处理卡顿)
    if all_imgs:
        batch_preload_images(all_imgs, cfg)

    # 3. 避免超大集合(如上千张)把单个 HTML 撑爆崩溃，设置友好展示上限
    max_cards = 250
    display_ids = failed_ids[:max_cards]
    trunc_notice = ""
    if len(failed_ids) > max_cards:
        trunc_notice = (
            f'<div class="empty-notice" style="background:#fffbeb; color:#b45309; border:1px solid #fde68a; margin-bottom:16px;">'
            f'⚠️ 未命中样本较多 (共 {len(failed_ids)} 个)，为保证网页交互流畅，当前展示前 {max_cards} 个样本卡片；'
            f'完整错误样本及文本对比请查看同目录下的 <code>failures.md</code> 或 <code>scored.jsonl</code>。'
            f'</div>'
        )

    # 统计各目标轮次 exact_match 未命中样本数
    turn_miss_stats: dict[object, dict] = {}
    for sid in failed_ids:
        sample_rows = rows_by_id.get(sid, [])
        for r in sample_rows:
            if _is_exact_match_miss(r):
                turn_val = r["turn"]
                ord_val = r.get("ordinal", turn_val)
                key = (turn_val, ord_val)
                if key not in turn_miss_stats:
                    turn_miss_stats[key] = {
                        "turn": turn_val,
                        "ordinal": ord_val,
                        "count": 0,
                    }
                turn_miss_stats[key]["count"] += 1

    sorted_keys = sorted(turn_miss_stats.keys(), key=lambda k: (k[0], k[1]))
    turn_options = []
    for turn_val, ord_val in sorted_keys:
        cnt = turn_miss_stats[(turn_val, ord_val)]["count"]
        if turn_val != ord_val:
            lbl = f"第 {turn_val} 轮 (目标 {ord_val}) 未命中 ({cnt} 个样本)"
        else:
            lbl = f"第 {turn_val} 轮未命中 ({cnt} 个样本)"
        turn_options.append(f'<option value="{turn_val}">{_html_escape(lbl)}</option>')

    turn_select_html = ""
    if turn_options:
        turn_select_html = f"""<label>按出错轮次筛选:
  <select id="flt-turn">
    <option value="">全部错误轮次 ({len(failed_ids)})</option>
    {"".join(turn_options)}
  </select>
</label>"""

    filters = f"""<div id="filters">
<input type="text" id="flt-search" placeholder="按样本 ID 搜索...">
{turn_select_html}
<label><input type="checkbox" id="flt-error"> 仅看报错/缺失样本</label>
<span id="flt-count" class="flt-count-badge"></span>
</div>""" + trunc_notice + '<div id="cards">'

    cards = "".join(
        _render_failure_card(sid, sample_by_id.get(sid), rows_by_id.get(sid, []), cfg)
        for sid in display_ids
    )

    script = """<script>
(function () {
  var cbErr = document.getElementById('flt-error');
  var inpSearch = document.getElementById('flt-search');
  var selTurn = document.getElementById('flt-turn');
  var countBadge = document.getElementById('flt-count');
  var cards = document.querySelectorAll('.card');

  function apply() {
    var onlyErr = cbErr ? cbErr.checked : false;
    var query = (inpSearch.value || '').trim().toLowerCase();
    var selTurnVal = selTurn ? selTurn.value : '';
    var visible = 0;
    cards.forEach(function (card) {
      var okErr = !onlyErr || card.getAttribute('data-has-error') === 'true';
      var sid = (card.getAttribute('data-sample-id') || '').toLowerCase();
      var okQuery = !query || sid.indexOf(query) !== -1;
      var cardTurns = (card.getAttribute('data-miss-turns') || '').split(',').filter(Boolean);
      var okTurn = !selTurnVal || cardTurns.indexOf(selTurnVal) !== -1;
      var show = okErr && okQuery && okTurn;
      card.style.display = show ? '' : 'none';
      if (show) visible++;
    });
    if (countBadge) {
      countBadge.textContent = '显示 ' + visible + ' / ' + cards.length + ' 个样本';
    }
  }
  if (cbErr) cbErr.addEventListener('change', apply);
  if (inpSearch) inpSearch.addEventListener('input', apply);
  if (selTurn) selTurn.addEventListener('change', apply);
  apply();

  // 点击图片放大 (lightbox)
  var lb = document.getElementById('lightbox');
  var lbImg = document.getElementById('lightbox-img');
  document.querySelectorAll('.images img').forEach(function (img) {
    img.addEventListener('click', function () {
      lbImg.src = img.src;
      lb.style.display = 'flex';
    });
  });
  lb.addEventListener('click', function () { lb.style.display = 'none'; });
})();
</script>"""

    return (
        header
        + filters
        + cards
        + "</div>"
        + script
        + '<div class="lightbox" id="lightbox"><img id="lightbox-img" alt=""></div>'
        + "</body></html>"
    )
