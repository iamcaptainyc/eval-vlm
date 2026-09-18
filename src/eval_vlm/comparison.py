"""Read-only, sample-first comparison of two or more completed evaluation runs.

This module deliberately does not reuse ``compare.py``: that module is the
small two-run quality crosstab used by precision/report.  Here the unit of
work is a reviewable sample turn, with the source image, conversation, gold
answer and every model output kept together.
"""
from __future__ import annotations

from collections import Counter
from html import escape
import json
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Optional

from .compare import _is_binary, _is_correct
from .config import Config
from .data.loader import load_samples
from .results.store import discover_run_dirs


def normalize_text(value: Any) -> str:
    """Comparison normalization is intentionally conservative and visible."""
    return str(value or "").strip()


def _jsonl(path: Path, warnings: list[str], label: str) -> dict[tuple[str, int], dict[str, Any]]:
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if "id" not in row:
                    raise ValueError("missing id")
                rows[(str(row["id"]), int(row.get("turn", -1)))] = row
            except Exception as exc:  # malformed rows must never hide the rest
                warnings.append(f"{label} 第 {number} 行已忽略: {exc}")
    return rows


def _run_id(model: str, backend: str) -> str:
    return f"{model}/{backend}"


def _parse_run_spec(spec: str) -> tuple[str, str]:
    clean = str(spec).replace("\\", "/").strip("/")
    if clean.count("/") != 1:
        raise ValueError(f"Run 必须是 model/backend: {spec!r}")
    model, backend = clean.split("/", 1)
    if not model or not backend or any(part in {".", ".."} for part in (model, backend)):
        raise ValueError(f"非法 Run: {spec!r}")
    return model, backend


def available_run_ids(cfg: Config) -> list[str]:
    return [_run_id(model, backend) for model, backend, _ in discover_run_dirs(cfg.dataset_dir)]


def _field_rows(
    run_dir: Path,
    warnings: list[str],
    target_turn_by_id: dict[str, int],
    dataset_dir: Optional[Path] = None,
) -> dict[tuple[str, int], dict[str, Any]]:
    path = run_dir / "field_mismatches.json"
    rows_dict: dict[tuple[str, int], dict[str, Any]] = {}
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            rows = data if isinstance(data, list) else data.get("rows", [])
            # field-eval historically stores one description result per id and
            # omits turn.  It evaluates the first selected target, so restore that
            # key from current test.json rather than losing the field comparison.
            for row in rows:
                if "id" in row:
                    sid = str(row["id"])
                    turn = int(row.get("turn", target_turn_by_id.get(sid, -1)))
                    rows_dict[(sid, turn)] = row
        except Exception as exc:
            warnings.append(f"{path.name} 无法读取: {exc}")

    # If fields_pred.jsonl exists, complement with all_correct samples that were omitted from mismatches
    pred_path = run_dir / "fields_pred.jsonl"
    ref_path = (dataset_dir / "fields_ref.jsonl") if dataset_dir else None
    if pred_path.exists():
        try:
            from .field_eval import load_fields
            pred_data = load_fields(pred_path)
            ref_data = load_fields(ref_path) if (ref_path and ref_path.exists()) else {}
            for sid, pred_fields in pred_data.items():
                turn = target_turn_by_id.get(sid, -1)
                key = (sid, turn)
                if key not in rows_dict:
                    ref_fields = ref_data.get(sid, {})
                    all_fields = sorted(set(ref_fields.keys()) | set(pred_fields.keys()))
                    field_rows = [
                        {
                            "field": f,
                            "ref": ref_fields.get(f, []),
                            "pred": pred_fields.get(f, []),
                            "correct": True,
                            "is_empty_ref": len(ref_fields.get(f, [])) == 0,
                        }
                        for f in all_fields
                    ]
                    rows_dict[key] = {
                        "id": sid,
                        "turn": turn,
                        "state": "all_correct",
                        "fields": field_rows,
                    }
        except Exception as exc:
            warnings.append(f"{pred_path.name} 补充字段失败: {exc}")

    return rows_dict


def _details(sample, target) -> dict[str, Any]:
    return {
        "id": sample.id,
        "turn": target.turn_index,
        "images": list(sample.images),
        "turns": [{"role": t.role, "content": t.content} for t in sample.turns],
        "reference": target.reference,
        "meta": sample.meta,
    }


def _record_for_run(run: dict[str, Any], key: tuple[str, int]) -> dict[str, Any]:
    pred = run["predictions"].get(key)
    scored = run["scored"].get(key)
    # score owns scorer/reference/detail; prediction owns latency/error/output.
    output = (pred or {}).get("prediction")
    error = (pred or {}).get("error")
    score = (scored or {}).get("score")
    scorer = (scored or {}).get("scorer")
    if error:
        status = "error"
    elif pred is None:
        status = "missing"
    elif scored is None:
        status = "unscored"
    elif _is_binary(scorer):
        status = "correct" if _is_correct(score) else "wrong"
    else:
        status = "scored"
    return {
        "run": run["id"], "model": run["model"], "backend": run["backend"],
        "prediction": output, "normalized_prediction": normalize_text(output),
        "latency": (pred or {}).get("latency"), "error": error,
        "score": score, "scorer": scorer, "detail": (scored or {}).get("detail"),
        "status": status, "field": run["fields"].get(key),
    }


def _categorize(outputs: list[dict[str, Any]], baseline_index: int) -> list[str]:
    categories: list[str] = []
    if any(x["status"] in {"missing", "error"} for x in outputs):
        categories.append("missing_or_error")
    if len({x["normalized_prediction"] for x in outputs}) > 1:
        categories.append("text_disagreement")
    fields = [json.dumps(x.get("field"), ensure_ascii=False, sort_keys=True, default=str) for x in outputs]
    if any(x.get("field") is not None for x in outputs) and len(set(fields)) > 1:
        categories.append("field_disagreement")
    comparable = [x for x in outputs if x["status"] in {"correct", "wrong"}]
    correctness = {x["status"] for x in comparable}
    if len(comparable) == len(outputs) and len(correctness) > 1:
        categories.append("correctness_disagreement")
    if comparable and len(comparable) == len(outputs) and all(x["status"] == "wrong" for x in comparable):
        categories.append("all_wrong")
    baseline = outputs[baseline_index]
    if baseline["status"] == "correct" and any(x["status"] == "wrong" for x in outputs):
        categories.append("regression")
    if baseline["status"] == "wrong" and any(x["status"] == "correct" for x in outputs):
        categories.append("improvement")
    return categories


def _summary(
    records: Iterable[dict[str, Any]],
    run_ids: list[str],
    run_dirs: Optional[dict[str, Path]] = None,
) -> dict[str, Any]:
    all_records = list(records)
    cats = Counter(category for row in all_records for category in row["categories"])
    turns = sorted({int(row["turn"]) for row in all_records if "turn" in row and row["turn"] is not None})
    runs: dict[str, Any] = {}

    for idx, run_id in enumerate(run_ids):
        output = [row["outputs"][idx] for row in all_records]
        scores = [float(x["score"]) for x in output if isinstance(x.get("score"), (int, float))]
        latency = [float(x["latency"]) for x in output if isinstance(x.get("latency"), (int, float))]

        # 1. 逐 Turn 指标统计
        turn_metrics: dict[str, Any] = {}
        for t in turns:
            t_rows = [(row, row["outputs"][idx]) for row in all_records if int(row.get("turn", -999)) == t]
            t_total = len(t_rows)
            t_correct = sum(1 for _, o in t_rows if o.get("status") == "correct")
            t_wrong = sum(1 for _, o in t_rows if o.get("status") == "wrong")
            t_scores = [float(o["score"]) for _, o in t_rows if isinstance(o.get("score"), (int, float))]
            t_latencies = [float(o["latency"]) for _, o in t_rows if isinstance(o.get("latency"), (int, float))]
            turn_metrics[str(t)] = {
                "turn": t,
                "total": t_total,
                "correct": t_correct,
                "wrong": t_wrong,
                "accuracy": round(t_correct / t_total, 4) if t_total > 0 else 0.0,
                "mean_score": round(mean(t_scores), 4) if t_scores else None,
                "mean_latency": round(mean(t_latencies), 4) if t_latencies else None,
            }

        # 2. field-eval 字段指标与全对率统计 (支持非空准确率与总体准确率两种口径)
        fm_path = (run_dirs.get(run_id) / "field_metrics.json") if run_dirs and run_dirs.get(run_id) else None
        field_eval_data: Optional[dict[str, Any]] = None
        if fm_path and fm_path.exists():
            try:
                raw_fm = json.loads(fm_path.read_text(encoding="utf-8"))
                ov = raw_fm.get("overall", {})
                num_samples = raw_fm.get("num_scored") or raw_fm.get("num_samples") or 0
                em_count = ov.get("exact_match_samples", ov.get("strict_exact_match_samples", 0))
                em_rate = ov.get("exact_match_rate", ov.get("strict_exact_match_rate", 0.0))
                strict_em_count = ov.get("strict_exact_match_samples", ov.get("exact_match_samples", 0))
                strict_em_rate = ov.get("strict_exact_match_rate", ov.get("exact_match_rate", 0.0))

                per_field_raw = raw_fm.get("per_field", {})
                fields_stat = {}
                for f_name, f_st in per_field_raw.items():
                    ne_tot = f_st.get("non_empty_count", f_st.get("total", 0))
                    ne_cor = f_st.get("non_empty_correct", f_st.get("correct", 0))
                    ne_acc = f_st.get("non_empty_accuracy", f_st.get("accuracy", 0.0))
                    ov_tot = f_st.get("overall_total", num_samples)
                    ov_cor = f_st.get("overall_correct", ne_cor)
                    ov_acc = f_st.get("overall_accuracy", ne_acc)
                    fields_stat[f_name] = {
                        "non_empty_total": ne_tot,
                        "non_empty_correct": ne_cor,
                        "non_empty_accuracy": ne_acc,
                        "overall_total": ov_tot,
                        "overall_correct": ov_cor,
                        "overall_accuracy": ov_acc,
                        # 兼容默认
                        "total": ne_tot,
                        "correct": ne_cor,
                        "accuracy": ne_acc,
                    }
                field_eval_data = {
                    "has_field_eval": True,
                    "total_samples": int(num_samples),
                    "exact_match_count": int(em_count),
                    "exact_match_rate": round(float(em_rate), 4),
                    "strict_exact_match_count": int(strict_em_count),
                    "strict_exact_match_rate": round(float(strict_em_rate), 4),
                    "all_correct_count": int(em_count),
                    "all_correct_rate": round(float(em_rate), 4),
                    "fields": fields_stat,
                }
            except Exception:
                field_eval_data = None

        if not field_eval_data:
            field_outputs = [o["field"] for o in output if o.get("field") and isinstance(o["field"], dict)]
            if field_outputs:
                tot_samples = len(field_outputs)
                ne_all_correct_cnt = 0
                strict_all_correct_cnt = 0
                field_accs: dict[str, dict[str, int]] = {}
                for fo in field_outputs:
                    flist = fo.get("fields") or fo.get("mismatches") or []
                    is_strict = (fo.get("state") == "all_correct") or (len(flist) > 0 and all(f.get("correct") is True for f in flist))
                    non_empty_items = [f for f in flist if not f.get("is_empty_ref")]
                    is_ne = (len(non_empty_items) > 0 and all(f.get("correct") is True for f in non_empty_items))
                    if is_strict:
                        strict_all_correct_cnt += 1
                    if is_ne:
                        ne_all_correct_cnt += 1

                    for fitem in flist:
                        fn = str(fitem.get("field", "")).strip()
                        if not fn:
                            continue
                        if fn not in field_accs:
                            field_accs[fn] = {
                                "non_empty_correct": 0, "non_empty_total": 0,
                                "overall_correct": 0, "overall_total": 0,
                            }
                        is_empty_ref = bool(fitem.get("is_empty_ref"))
                        is_cor = (fitem.get("correct") is True)
                        field_accs[fn]["overall_total"] += 1
                        if is_cor:
                            field_accs[fn]["overall_correct"] += 1
                        if not is_empty_ref:
                            field_accs[fn]["non_empty_total"] += 1
                            if is_cor:
                                field_accs[fn]["non_empty_correct"] += 1

                fields_stat = {}
                for fn, st in field_accs.items():
                    ne_tot = st["non_empty_total"]
                    ne_cor = st["non_empty_correct"]
                    ne_acc = round(ne_cor / ne_tot, 4) if ne_tot > 0 else 0.0
                    ov_tot = st["overall_total"]
                    ov_cor = st["overall_correct"]
                    ov_acc = round(ov_cor / ov_tot, 4) if ov_tot > 0 else 0.0
                    fields_stat[fn] = {
                        "non_empty_total": ne_tot,
                        "non_empty_correct": ne_cor,
                        "non_empty_accuracy": ne_acc,
                        "overall_total": ov_tot,
                        "overall_correct": ov_cor,
                        "overall_accuracy": ov_acc,
                        "total": ne_tot,
                        "correct": ne_cor,
                        "accuracy": ne_acc,
                    }

                field_eval_data = {
                    "has_field_eval": True,
                    "total_samples": tot_samples,
                    "exact_match_count": ne_all_correct_cnt,
                    "exact_match_rate": round(ne_all_correct_cnt / tot_samples, 4) if tot_samples > 0 else 0.0,
                    "strict_exact_match_count": strict_all_correct_cnt,
                    "strict_exact_match_rate": round(strict_all_correct_cnt / tot_samples, 4) if tot_samples > 0 else 0.0,
                    "all_correct_count": ne_all_correct_cnt,
                    "all_correct_rate": round(ne_all_correct_cnt / tot_samples, 4) if tot_samples > 0 else 0.0,
                    "fields": fields_stat,
                }
            else:
                field_eval_data = {
                    "has_field_eval": False,
                    "total_samples": 0,
                    "all_correct_count": 0,
                    "all_correct_rate": 0.0,
                    "exact_match_count": 0,
                    "exact_match_rate": 0.0,
                    "strict_exact_match_count": 0,
                    "strict_exact_match_rate": 0.0,
                    "fields": {},
                }

        runs[run_id] = {
            "mean_score": round(mean(scores), 4) if scores else None,
            "coverage": sum(x["status"] not in {"missing", "error"} for x in output),
            "missing_or_error": sum(x["status"] in {"missing", "error"} for x in output),
            "correct": sum(x["status"] == "correct" for x in output),
            "wrong": sum(x["status"] == "wrong" for x in output),
            "mean_latency": round(mean(latency), 4) if latency else None,
            "turn_metrics": turn_metrics,
            "field_eval": field_eval_data,
        }

    all_field_names: list[str] = []
    for r_st in runs.values():
        fe = r_st.get("field_eval") or {}
        for fn in (fe.get("fields") or {}).keys():
            if fn not in all_field_names:
                all_field_names.append(fn)

    return {
        "aligned_total": len(all_records),
        "categories": dict(cats),
        "runs": runs,
        "turns": turns,
        "field_names": all_field_names,
    }


def compare_dataset(
    cfg: Config,
    run_specs: list[str],
    baseline: Optional[str] = None,
    *, allow_mixed_dataset: bool = False,
) -> dict[str, Any]:
    """Load, align and classify comparison records.  This is read-only."""
    if len(run_specs) < 2:
        raise ValueError("至少选择两个 Run")
    if len(set(run_specs)) != len(run_specs):
        raise ValueError("Run 不能重复")
    available = set(available_run_ids(cfg))
    missing = [spec for spec in run_specs if spec not in available]
    if missing:
        raise ValueError(f"未找到 Run: {', '.join(missing)}")
    baseline = baseline or run_specs[0]
    if baseline not in run_specs:
        raise ValueError("baseline 必须位于 --runs 中")
    warnings: list[str] = []
    current_sha = _sha(cfg.test_path)
    # Comparison must follow the current test split, never the original source
    # JSON.  Field-eval's legacy mismatch artifact has no turn, so map it to
    # its first selected target (the same description target field-eval uses).
    samples = load_samples(cfg, cfg.test_path)
    target_turn_by_id = {sample.id: sample.targets[0].turn_index for sample in samples if sample.targets}
    runs: list[dict[str, Any]] = []
    sha_values: set[str] = set()
    for spec in run_specs:
        model, backend = _parse_run_spec(spec)
        run_dir = cfg.dataset_dir / model / backend
        meta: dict[str, Any] = {}
        try:
            meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
        except FileNotFoundError:
            warnings.append(f"{spec} 缺少 run_meta.json，无法验证数据版本")
        except Exception as exc:
            warnings.append(f"{spec} 的 run_meta.json 无法读取: {exc}")
        run_sha = meta.get("test_sha256")
        if run_sha:
            sha_values.add(str(run_sha))
            if current_sha and run_sha != current_sha:
                warnings.append(f"{spec} 的 test SHA 与当前数据集不一致")
        if (run_dir / "dataset_dirty.json").exists():
            warnings.append(f"{spec} 已标记为 stale")
        runs.append({"id": spec, "model": model, "backend": backend, "dir": run_dir,
                     "predictions": _jsonl(run_dir / "predictions.jsonl", warnings, f"{spec}/predictions"),
                     "scored": _jsonl(run_dir / "scored.jsonl", warnings, f"{spec}/scored"),
                     "fields": _field_rows(run_dir, warnings, target_turn_by_id, dataset_dir=cfg.dataset_dir)})
    if len(sha_values) > 1 or any("不一致" in w for w in warnings):
        if not allow_mixed_dataset:
            raise ValueError("Run 使用的数据集版本不一致；如确认需要强制比较，请传 --allow-mixed-dataset")
    records: list[dict[str, Any]] = []
    known_keys: set[tuple[str, int]] = set()
    for sample in samples:
        for target in sample.targets:
            key = (sample.id, target.turn_index)
            known_keys.add(key)
            outputs = [_record_for_run(run, key) for run in runs]
            records.append({**_details(sample, target), "outputs": outputs,
                            "categories": _categorize(outputs, run_specs.index(baseline))})
    # Surface prediction rows which no longer correspond to test.json rather than silently losing them.
    extras = sorted(set().union(*(set(r["predictions"]) | set(r["scored"]) for r in runs)) - known_keys)
    for sid, turn in extras:
        outputs = [_record_for_run(run, (sid, turn)) for run in runs]
        records.append({"id": sid, "turn": turn, "images": [], "turns": [], "reference": None, "meta": {},
                        "outputs": outputs, "categories": list(dict.fromkeys(
                            ["missing_or_error", "orphan_result"] + _categorize(outputs, run_specs.index(baseline))
                        ))})
        warnings.append(f"结果中有不在当前 test.json 的记录: {sid}/{turn}")
    return {"dataset": cfg.dataset_dir.name, "runs": run_specs, "baseline": baseline,
            "warnings": list(dict.fromkeys(warnings)), "records": records,
            "summary": _summary(records, run_specs, run_dirs={r["id"]: r["dir"] for r in runs}),
            "normalization": "prediction.strip()"}


def _sha(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_field_disagreement(record: dict[str, Any], target_field: str) -> bool:
    """判断当前样本在指定 target_field 抽取字段上是否存在不一致或判定错误。"""
    target_items = []
    target_fobjs = []
    for o in record.get("outputs", []):
        fobj = o.get("field")
        if not fobj or not isinstance(fobj, dict):
            continue
        flist = fobj.get("fields") or fobj.get("mismatches") or []
        item = next((x for x in flist if str(x.get("field", "")).strip() == target_field), None)
        if item is not None:
            target_items.append(item)
            target_fobjs.append(fobj)

    if not target_items:
        return False

    # 1. 任意模型在该字段上判定为错误、或 ref 与 pred 值不相等、或在 mismatch 清单中
    for item, fobj in zip(target_items, target_fobjs):
        if item.get("correct") is False:
            return True
        if item.get("ref") is not None and item.get("pred") is not None:
            r = sorted(item.get("ref") or [])
            p = sorted(item.get("pred") or [])
            if r != p:
                return True
        if fobj.get("state") == "mismatch" and item.get("correct") is not True:
            return True

    # 2. 各模型对该字段的预测内容存在分歧
    normalized_preds = {
        json.dumps(sorted(item.get("pred") or []), ensure_ascii=False)
        for item in target_items
    }
    if len(normalized_preds) > 1:
        return True

    # 3. 各模型对该字段的判定正确性不一致
    if len({item.get("correct") for item in target_items}) > 1:
        return True

    return False


def filter_records(records: Iterable[dict[str, Any]], *, category: Optional[str] = None,
                   query: Optional[str] = None, include_agreements: bool = False,
                   field: Optional[str] = None) -> list[dict[str, Any]]:
    needle = (query or "").strip().lower()
    target_field = (field or "").strip()
    filtered: list[dict[str, Any]] = []
    for record in records:
        categories = record["categories"]
        if category and category not in categories:
            continue
        if target_field:
            if not _is_field_disagreement(record, target_field):
                continue
        elif not include_agreements and not category:
            # ``all_wrong`` alone describes a hard sample, not a disagreement:
            # do not flood the default reviewer queue with identical failures.
            meaningful = {"text_disagreement", "field_disagreement", "correctness_disagreement",
                          "missing_or_error", "regression", "improvement", "orphan_result"}
            if not meaningful.intersection(categories):
                continue
        if needle:
            text = " ".join([str(record.get("id", "")), str(record.get("reference", ""))] +
                            [str(x.get("prediction", "")) for x in record["outputs"]]).lower()
            if needle not in text:
                continue
        filtered.append(record)
    return filtered


def sort_records(records: list[dict[str, Any]], sort: str = "priority", descending: bool = False) -> list[dict[str, Any]]:
    priority = {"regression": 0, "correctness_disagreement": 1, "improvement": 2,
                "missing_or_error": 3, "text_disagreement": 4, "all_wrong": 5}
    if sort == "score_delta":
        def key(row: dict[str, Any]):
            scorers = {x.get("scorer") for x in row["outputs"]}
            if len(scorers) != 1 or None in scorers:
                return -1  # scores from different scorers are not comparable
            values = [x.get("score") for x in row["outputs"] if isinstance(x.get("score"), (int, float))]
            return max(values) - min(values) if len(values) > 1 else -1
    elif sort == "id":
        def key(row: dict[str, Any]): return (str(row["id"]), row["turn"])
    else:
        def key(row: dict[str, Any]): return (min((priority.get(c, 99) for c in row["categories"]), default=99), str(row["id"]), row["turn"])
    return sorted(records, key=key, reverse=descending)


def _highlight(text: Any, others: list[str]) -> str:
    value = str(text or "")
    # Safe but intentionally simple: whole prediction is highlighted when unique.
    klass = " diff-text" if normalize_text(value) not in {normalize_text(x) for x in others} else ""
    return f'<pre class="prediction{klass}">{escape(value)}</pre>'


def render_html(comparison: dict[str, Any], records: list[dict[str, Any]], image_url=None) -> str:
    """Self-contained offline report. ``image_url`` may map a source ref to an API URL."""
    rows: list[str] = []
    for record in records:
        images = []
        for ref in record.get("images", []):
            source = image_url(ref) if image_url else (Path(ref).resolve().as_uri() if not str(ref).startswith(("http://", "https://", "data:")) else ref)
            images.append(f'<img src="{escape(source, quote=True)}" alt="source image">')
        cards = []
        for output in record["outputs"]:
            others = [x.get("prediction", "") for x in record["outputs"] if x is not output]
            cards.append("<article class='output-card'><h3>%s</h3><small>%s · score=%s · latency=%s</small>%s%s%s</article>" % (
                escape(output["run"]), escape(output["status"]), escape(str(output.get("score", "—"))),
                escape(str(output.get("latency", "—"))), _highlight(output.get("prediction"), others),
                f"<p class='error'>{escape(str(output['error']))}</p>" if output.get("error") else "",
                (f"<details><summary>field-eval 字段结果</summary><pre>{escape(json.dumps(output['field'], ensure_ascii=False, indent=2, default=str))}</pre></details>" if output.get("field") is not None else "") +
                (f"<details><summary>scorer detail</summary><pre>{escape(json.dumps(output['detail'], ensure_ascii=False, indent=2, default=str))}</pre></details>" if output.get("detail") is not None else "")))
        context = "".join(f"<p><b>{escape(t['role'])}:</b> {escape(t['content'])}</p>" for t in record.get("turns", []))
        rows.append("<section class='sample'><header><b>%s / turn %s</b><span>%s</span></header><div class='images'>%s</div><div class='context'>%s<h3>真值</h3><pre>%s</pre></div><div class='outputs'>%s</div></section>" % (
            escape(str(record["id"])), record["turn"], escape(", ".join(record["categories"]) or "一致"), "".join(images), context,
            escape(str(record.get("reference") or "")), "".join(cards)))
    return """<!doctype html><html lang='zh-CN'><meta charset='utf-8'><title>模型样本对比</title><style>
body{font:14px system-ui;margin:24px;background:#101827;color:#e5e7eb} .sample{border:1px solid #344155;border-radius:10px;padding:16px;margin:18px 0;background:#172033}.sample header{display:flex;justify-content:space-between;color:#7dd3fc}.images img{max-width:360px;max-height:300px;margin:10px 10px 10px 0}.outputs{display:flex;gap:12px;overflow-x:auto}.output-card{min-width:300px;flex:1;border:1px solid #40506a;border-radius:7px;padding:10px}.prediction{white-space:pre-wrap}.diff-text{background:#3b2734;border-left:3px solid #fb7185;padding-left:8px}.error{color:#fda4af}.context pre,details pre{white-space:pre-wrap}small{color:#a5b4fc}</style><body><h1>模型样本级对比</h1><p>数据集: %s；Baseline: %s；仅含当前筛选的 %s 条。文本比较规范化: %s。</p>%s</body></html>""" % (
        escape(comparison["dataset"]), escape(comparison["baseline"]), len(records), escape(comparison["normalization"]), "".join(rows))
