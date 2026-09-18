"""Contract tests for the review-first multi-run comparison core and CLI."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from eval_vlm.cli import build_parser
from eval_vlm.comparison import compare_dataset, filter_records, render_html
from eval_vlm.config import Config
from eval_vlm.data.loader import _stable_id
from eval_vlm.results.store import discover_run_dirs


def _setup(tmp_path: Path) -> tuple[Config, str]:
    record = {"messages": [
        {"role": "user", "content": "<image> first"},
        {"role": "assistant", "content": "gold one"},
        {"role": "user", "content": "second"},
        {"role": "assistant", "content": "gold two"},
    ], "images": ["photo.png"]}
    (tmp_path / "test.json").write_text(json.dumps([record]), encoding="utf-8")
    sid = _stable_id(0, record)
    cfg = Config(run_dir_path=tmp_path)
    digest = hashlib.sha256((tmp_path / "test.json").read_bytes()).hexdigest()
    for model, first, second in (("old", "gold one", "wrong"), ("new", "wrong", "gold two")):
        run = tmp_path / model / "hf"
        run.mkdir(parents=True)
        (run / "run_meta.json").write_text(json.dumps({"test_sha256": digest}), encoding="utf-8")
        (run / "predictions.jsonl").write_text("\n".join(json.dumps({"id": sid, "turn": turn, "prediction": prediction, "latency": .2}) for turn, prediction in ((1, first), (3, second))) + "\n", encoding="utf-8")
        (run / "scored.jsonl").write_text("\n".join(json.dumps({"id": sid, "turn": turn, "prediction": prediction, "reference": "gold", "scorer": "exact_match", "score": float(prediction.startswith("gold"))}) for turn, prediction in ((1, first), (3, second))) + "\n", encoding="utf-8")
    return cfg, sid


def test_comparison_aligns_multiturn_and_classifies_regression_improvement(tmp_path):
    cfg, sid = _setup(tmp_path)
    result = compare_dataset(cfg, ["old/hf", "new/hf"], baseline="old/hf")
    assert len(result["records"]) == 2
    assert {row["turn"] for row in result["records"]} == {1, 3}
    by_turn = {row["turn"]: row for row in result["records"]}
    assert "regression" in by_turn[1]["categories"]
    assert "improvement" in by_turn[3]["categories"]
    assert all(row["id"] == sid for row in result["records"])


def test_comparison_keeps_missing_errors_and_rejects_mixed_sha(tmp_path):
    cfg, _ = _setup(tmp_path)
    (tmp_path / "new" / "hf" / "predictions.jsonl").write_text("", encoding="utf-8")
    result = compare_dataset(cfg, ["old/hf", "new/hf"])
    assert all("missing_or_error" in row["categories"] for row in result["records"])
    (tmp_path / "new" / "hf" / "run_meta.json").write_text(json.dumps({"test_sha256": "other"}), encoding="utf-8")
    with pytest.raises(ValueError, match="allow-mixed-dataset"):
        compare_dataset(cfg, ["old/hf", "new/hf"])


def test_default_filter_hides_agreements_and_report_escapes_html(tmp_path):
    cfg, _ = _setup(tmp_path)
    result = compare_dataset(cfg, ["old/hf", "new/hf"])
    result["records"][0]["reference"] = "<script>alert(1)</script>"
    assert filter_records([{**result["records"][0], "categories": []}]) == []
    report = render_html(result, result["records"])
    assert "&lt;script&gt;" in report
    assert "<script>alert" not in report


def test_default_filter_hides_identical_all_wrong_but_keeps_explicit_filter(tmp_path):
    cfg, sid = _setup(tmp_path)
    for model in ("old", "new"):
        (tmp_path / model / "hf" / "predictions.jsonl").write_text(
            "".join(json.dumps({"id": sid, "turn": turn, "prediction": "same bad"}) + "\n" for turn in (1, 3)), encoding="utf-8"
        )
        (tmp_path / model / "hf" / "scored.jsonl").write_text(
            "".join(json.dumps({"id": sid, "turn": turn, "scorer": "exact_match", "score": 0.0}) + "\n" for turn in (1, 3)), encoding="utf-8"
        )
    result = compare_dataset(cfg, ["old/hf", "new/hf"])
    assert filter_records(result["records"]) == []
    assert len(filter_records(result["records"], category="all_wrong")) == 2


def test_field_mismatch_without_turn_is_mapped_to_first_target(tmp_path):
    cfg, sid = _setup(tmp_path)
    mismatch = {"rows": [{"id": sid, "state": "mismatch", "fields": [{"field": "road", "ref": ["a"], "pred": ["b"]}]}]}
    (tmp_path / "old" / "hf" / "field_mismatches.json").write_text(json.dumps(mismatch), encoding="utf-8")
    result = compare_dataset(cfg, ["old/hf", "new/hf"])
    first = next(row for row in result["records"] if row["turn"] == 1)
    assert first["outputs"][0]["field"]["state"] == "mismatch"

    # 测试按特定字段筛选不一致样本
    road_disagreements = filter_records(result["records"], field="road")
    assert len(road_disagreements) >= 1
    other_disagreements = filter_records(result["records"], field="non_existent_field")
    assert len(other_disagreements) == 0


def test_compare_cli_parser_accepts_multiple_runs():
    args = build_parser().parse_args(["compare", "--dataset", "demo", "--runs", "old/hf", "new/hf", "--output", "out"])
    assert args.command == "compare"
    assert args.runs == [["old/hf", "new/hf"]]


def test_predictions_only_directory_is_discoverable(tmp_path):
    run = tmp_path / "pred-only" / "hf"
    run.mkdir(parents=True)
    (run / "predictions.jsonl").write_text("", encoding="utf-8")
    assert [(model, backend) for model, backend, _ in discover_run_dirs(tmp_path)] == [("pred-only", "hf")]


def test_field_rows_complemented_from_fields_pred(tmp_path):
    cfg, sid = _setup(tmp_path)
    (tmp_path / "fields_ref.jsonl").write_text(json.dumps({"id": sid, "fields": {"color": ["red"]}}) + "\n", encoding="utf-8")
    (tmp_path / "new" / "hf" / "fields_pred.jsonl").write_text(json.dumps({"id": sid, "fields": {"color": ["red"]}}) + "\n", encoding="utf-8")
    result = compare_dataset(cfg, ["old/hf", "new/hf"])
    first = next(row for row in result["records"] if row["turn"] == 1)
    new_field = first["outputs"][1]["field"]
    assert new_field is not None
    assert new_field["state"] == "all_correct"
    assert new_field["fields"][0]["field"] == "color"
    assert new_field["fields"][0]["correct"] is True


def test_summary_includes_turn_metrics_and_field_eval(tmp_path):
    cfg, sid = _setup(tmp_path)
    # 给 new/hf 写入 field_metrics.json
    fm = {
        "num_scored": 1,
        "overall": {
            "strict_exact_match_samples": 1,
            "strict_exact_match_rate": 1.0,
        },
        "per_field": {
            "road": {"total": 1, "correct": 1, "accuracy": 1.0},
            "lane": {"total": 1, "correct": 1, "accuracy": 1.0},
        },
    }
    (tmp_path / "new" / "hf" / "field_metrics.json").write_text(json.dumps(fm), encoding="utf-8")

    result = compare_dataset(cfg, ["old/hf", "new/hf"])
    summary = result["summary"]
    assert "turns" in summary
    assert summary["turns"] == [1, 3]
    assert "field_names" in summary
    assert "road" in summary["field_names"]

    # 验证 turn_metrics
    old_turns = summary["runs"]["old/hf"]["turn_metrics"]
    assert "1" in old_turns and "3" in old_turns
    assert old_turns["1"]["correct"] == 1
    assert old_turns["3"]["correct"] == 0

    # 验证 field_eval
    new_fe = summary["runs"]["new/hf"]["field_eval"]
    assert new_fe["has_field_eval"] is True
    assert new_fe["total_samples"] == 1
    assert new_fe["all_correct_count"] == 1
    assert new_fe["all_correct_rate"] == 1.0
    assert "road" in new_fe["fields"]
    assert new_fe["fields"]["road"]["accuracy"] == 1.0
    assert new_fe["fields"]["road"]["non_empty_accuracy"] == 1.0
    assert new_fe["fields"]["road"]["overall_accuracy"] == 1.0
    assert new_fe["exact_match_rate"] == 1.0
    assert new_fe["strict_exact_match_rate"] == 1.0

    old_fe = summary["runs"]["old/hf"]["field_eval"]
    assert old_fe["has_field_eval"] is False


def test_field_eval_only_runs_support_regression_and_improvement(tmp_path):
    """验证纯 field-eval 运行（无 scored.jsonl）也能正确识别 regression 和 improvement。"""
    record = {"messages": [
        {"role": "user", "content": "<image> describe"},
        {"role": "assistant", "content": "description gold"},
    ], "images": ["photo.png"]}
    (tmp_path / "test.json").write_text(json.dumps([record]), encoding="utf-8")
    sid = _stable_id(0, record)
    cfg = Config(run_dir_path=tmp_path)
    digest = hashlib.sha256((tmp_path / "test.json").read_bytes()).hexdigest()

    # old/hf: 全对 (all_correct)
    run1 = tmp_path / "old" / "hf"
    run1.mkdir(parents=True)
    (run1 / "run_meta.json").write_text(json.dumps({"test_sha256": digest}), encoding="utf-8")
    (run1 / "predictions.jsonl").write_text(json.dumps({"id": sid, "turn": 1, "prediction": "old pred"}) + "\n", encoding="utf-8")
    (run1 / "field_mismatches.json").write_text(json.dumps({"rows": []}), encoding="utf-8")
    (run1 / "field_metrics.json").write_text(json.dumps({"num_scored": 1, "per_field": {"lane": {"total": 1, "correct": 1}}}), encoding="utf-8")

    # new/hf: 车道位置失配 (mismatch)
    run2 = tmp_path / "new" / "hf"
    run2.mkdir(parents=True)
    (run2 / "run_meta.json").write_text(json.dumps({"test_sha256": digest}), encoding="utf-8")
    (run2 / "predictions.jsonl").write_text(json.dumps({"id": sid, "turn": 1, "prediction": "new pred"}) + "\n", encoding="utf-8")
    (run2 / "field_mismatches.json").write_text(json.dumps({
        "rows": [{
            "id": sid, "turn": 1, "state": "mismatch",
            "fields": [{"field": "车道位置", "ref": ["第一车道"], "pred": ["第二车道"], "correct": False}]
        }]
    }), encoding="utf-8")
    (run2 / "field_metrics.json").write_text(json.dumps({"num_scored": 1, "per_field": {"lane": {"total": 1, "correct": 0}}}), encoding="utf-8")

    result = compare_dataset(cfg, ["old/hf", "new/hf"], baseline="old/hf")
    records = result["records"]
    assert len(records) == 1
    # Baseline old/hf 是全对 (correct), 候选 new/hf 是失配 (wrong) -> 必定归类为 regression!
    assert "regression" in records[0]["categories"]
    assert records[0]["outputs"][0]["status"] == "correct"
    assert records[0]["outputs"][1]["status"] == "wrong"


def test_field_filter_combined_with_category_filter(tmp_path):
    """验证按字段筛选（如车道位置）与分歧类型筛选（如 regression / improvement / all_wrong）联合生效。"""
    rec1 = {"messages": [{"role": "user", "content": "1"}, {"role": "assistant", "content": "1"}], "images": []}
    rec2 = {"messages": [{"role": "user", "content": "2"}, {"role": "assistant", "content": "2"}], "images": []}
    rec3 = {"messages": [{"role": "user", "content": "3"}, {"role": "assistant", "content": "3"}], "images": []}
    (tmp_path / "test.json").write_text(json.dumps([rec1, rec2, rec3]), encoding="utf-8")
    s1, s2, s3 = _stable_id(0, rec1), _stable_id(1, rec2), _stable_id(2, rec3)
    cfg = Config(run_dir_path=tmp_path)
    digest = hashlib.sha256((tmp_path / "test.json").read_bytes()).hexdigest()

    run1 = tmp_path / "base" / "hf"
    run1.mkdir(parents=True)
    (run1 / "run_meta.json").write_text(json.dumps({"test_sha256": digest}), encoding="utf-8")
    (run1 / "predictions.jsonl").write_text("\n".join(json.dumps({"id": sid, "turn": 1, "prediction": "p"}) for sid in (s1, s2, s3)) + "\n", encoding="utf-8")
    # base 模型在 s1 上车道对、主路错; 在 s2 上车道错; 在 s3 上车道错
    (run1 / "field_mismatches.json").write_text(json.dumps({
        "rows": [
            {"id": s1, "turn": 1, "state": "mismatch", "fields": [
                {"field": "主辅路", "ref": ["主路"], "pred": ["辅路"], "correct": False},
                {"field": "车道位置", "ref": ["第一车道"], "pred": ["第一车道"], "correct": True},
            ]},
            {"id": s2, "turn": 1, "state": "mismatch", "fields": [
                {"field": "车道位置", "ref": ["第一车道"], "pred": ["第二车道"], "correct": False},
            ]},
            {"id": s3, "turn": 1, "state": "mismatch", "fields": [
                {"field": "车道位置", "ref": ["第一车道"], "pred": ["第三车道"], "correct": False},
            ]},
        ]
    }), encoding="utf-8")

    run2 = tmp_path / "cand" / "hf"
    run2.mkdir(parents=True)
    (run2 / "run_meta.json").write_text(json.dumps({"test_sha256": digest}), encoding="utf-8")
    (run2 / "predictions.jsonl").write_text("\n".join(json.dumps({"id": sid, "turn": 1, "prediction": "p"}) for sid in (s1, s2, s3)) + "\n", encoding="utf-8")
    # cand 模型在 s1 上车道错; 在 s2 上车道对; 在 s3 上车道错
    (run2 / "field_mismatches.json").write_text(json.dumps({
        "rows": [
            {"id": s1, "turn": 1, "state": "mismatch", "fields": [
                {"field": "主辅路", "ref": ["主路"], "pred": ["辅路"], "correct": False},
                {"field": "车道位置", "ref": ["第一车道"], "pred": ["第二车道"], "correct": False},
            ]},
            {"id": s2, "turn": 1, "state": "all_correct", "fields": [
                {"field": "车道位置", "ref": ["第一车道"], "pred": ["第一车道"], "correct": True},
            ]},
            {"id": s3, "turn": 1, "state": "mismatch", "fields": [
                {"field": "车道位置", "ref": ["第一车道"], "pred": ["第四车道"], "correct": False},
            ]},
        ]
    }), encoding="utf-8")

    result = compare_dataset(cfg, ["base/hf", "cand/hf"], baseline="base/hf")
    all_recs = result["records"]

    # 1. 联合筛选：field="车道位置" 且 category="regression" (用户报告的核心 bug 场景)
    # s1 上 base 车道对、cand 车道错，应该精准命中 s1，且不能返回 0！
    reg_recs = filter_records(all_recs, category="regression", field="车道位置", baseline="base/hf")
    assert len(reg_recs) == 1
    assert reg_recs[0]["id"] == s1
    assert "regression" in reg_recs[0]["target_field_categories"]

    # 2. 联合筛选：field="车道位置" 且 category="improvement"
    # s2 上 base 车道错、cand 车道对，精准命中 s2
    imp_recs = filter_records(all_recs, category="improvement", field="车道位置", baseline="base/hf")
    assert len(imp_recs) == 1
    assert imp_recs[0]["id"] == s2
    assert "improvement" in imp_recs[0]["target_field_categories"]

    # 3. 联合筛选：field="车道位置" 且 category="all_wrong"
    # s3 上两者车道皆错，精准命中 s3
    wrong_recs = filter_records(all_recs, category="all_wrong", field="车道位置", baseline="base/hf")
    assert len(wrong_recs) == 1
    assert wrong_recs[0]["id"] == s3

    # 4. 验证 summary 中包含 categories_by_field 统计
    assert "categories_by_field" in result["summary"]
    lane_cats = result["summary"]["categories_by_field"].get("车道位置", {})
    assert lane_cats.get("regression") == 1
    assert lane_cats.get("improvement") == 1
    assert lane_cats.get("all_wrong") == 1


