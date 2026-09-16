import json
import os
from pathlib import Path
import pytest
from starlette.testclient import TestClient

from eval_vlm.webui.app import create_app
from eval_vlm.webui.settings import Settings


def write_summary(path: Path, *, model: str = "test_model", backend: str = "vllm_offline") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "datasets": ["ds1", "ds2"],
        "results": [{
            "dataset": "ds1",
            "method": "field-eval",
            "model": model,
            "backend": backend,
            "status": "ok",
            "metrics": {"num_samples": 100, "overall": {"micro_accuracy": 0.95}},
        }],
        "num_ok": 1,
        "num_error": 0,
    }, ensure_ascii=False), encoding="utf-8")


def test_api_sweep_runs_and_summary(tmp_path: Path):
    workspace = tmp_path / "ws"
    workspace.mkdir()

    sweep_dir = workspace / "_sweep" / "test_model" / "vllm_offline"
    sweep_dir.mkdir(parents=True)

    write_summary(sweep_dir / "summary.json")

    settings = Settings(workspace_dir=workspace)
    app = create_app(settings)
    client = TestClient(app)

    # 1. Test listing sweep runs
    res = client.get("/api/sweep/runs")
    assert res.status_code == 200
    runs = res.json()
    assert len(runs) == 1
    assert runs[0]["model"] == "test_model"
    assert runs[0]["backend"] == "vllm_offline"
    assert runs[0]["datasets_count"] == 2

    # 2. Test reading summary by path
    rel_path = runs[0]["path"]
    res_sum = client.get(f"/api/sweep/summary?path={rel_path}")
    assert res_sum.status_code == 200
    data = res_sum.json()
    assert data["datasets"] == ["ds1", "ds2"]
    assert len(data["results"]) == 1

    # 3. Test reading summary by model and backend
    res_mb = client.get("/api/sweep/summary?model=test_model&backend=vllm_offline")
    assert res_mb.status_code == 200
    assert res_mb.json()["results"][0]["dataset"] == "ds1"

    # 4. Test 404 for nonexistent run
    res_404 = client.get("/api/sweep/summary?model=nonexistent&backend=vllm")
    assert res_404.status_code == 404

    # 5. Test path traversal protection
    res_traversal = client.get("/api/sweep/summary?path=../../etc/passwd")
    assert res_traversal.status_code in (403, 404)


def test_sweep_listing_is_limited_to_model_backend_summary_files(tmp_path: Path):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    newest = workspace / "_sweep" / "model-new" / "backend-a" / "summary.json"
    oldest = workspace / "_sweep" / "model-old" / "backend-b" / "summary.json"
    write_summary(newest, model="different-name")
    write_summary(oldest)
    os.utime(oldest, (10, 10))
    os.utime(newest, (20, 20))

    # Neither a deep result nor a summary at the wrong level is a sweep run.
    write_summary(workspace / "_sweep" / "model-new" / "backend-a" / "nested" / "summary.json")
    write_summary(workspace / "_sweep" / "model-only" / "summary.json")

    client = TestClient(create_app(Settings(workspace_dir=workspace)))
    runs = client.get("/api/sweep/runs").json()

    assert [run["path"] for run in runs] == [
        "_sweep/model-new/backend-a/summary.json",
        "_sweep/model-old/backend-b/summary.json",
    ]
    # Directory names are the browse categories, independent of summary content.
    assert runs[0]["model"] == "model-new"
    assert runs[0]["backend"] == "backend-a"


def test_sweep_summary_rejects_noncanonical_or_outside_paths(tmp_path: Path):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    valid = workspace / "_sweep" / "model" / "backend" / "summary.json"
    write_summary(valid)
    outside = tmp_path / "ws_evil" / "_sweep" / "model" / "backend" / "summary.json"
    write_summary(outside)

    client = TestClient(create_app(Settings(workspace_dir=workspace)))
    assert client.get("/api/sweep/summary?path=_sweep/model/backend/summary.json").status_code == 200
    assert client.get("/api/sweep/summary?path=_sweep/model/backend/not-summary.json").status_code == 404
    assert client.get("/api/sweep/summary?path=_sweep/model/backend/nested/summary.json").status_code == 404
    assert client.get("/api/sweep/summary?path=../ws_evil/_sweep/model/backend/summary.json").status_code == 403


def test_sweep_frontend_scan_always_leaves_loading_state():
    source = (Path(__file__).parents[1] / "src" / "eval_vlm" / "webui" / "static" / "app.js").read_text(encoding="utf-8")
    scan = source.split("async function loadSweepRunsList", 1)[1].split("async function onSweepRunSelect", 1)[0]
    assert "runsLoadPromise" in scan
    assert "AbortController" in scan
    assert "controller.abort(), 5000" in scan
    assert "正在扫描工作区 _sweep/" in scan
    assert "扫描失败" in scan
    assert "clearSweepResultsState()" in scan
    assert "forceRefresh || !activeExists || !state.sweepResults.data" in scan
    assert "finally" in scan
    clear_state = source.split("function clearSweepResultsState", 1)[1].split("function showSweepEmptyState", 1)[0]
    assert "selectionRequestId++" in clear_state


def test_sweep_frontend_overall_accuracy_and_heatmap_colormap():
    source = (Path(__file__).parents[1] / "src" / "eval_vlm" / "webui" / "static" / "app.js").read_text(encoding="utf-8")
    assert "overall.micro_overall_accuracy" in source
    assert "getSweepResultNonEmptyAccuracy" in source
    assert "总体准确率 (overall_acc):" in source
    assert "非空准确率 (non_empty_acc):" in source
    assert "getHeatmapCellProps" in source
    assert "Math.pow(norm, 0.65)" in source
    assert "sr-cm-gradient-bar" in source
    assert "val / rowTotal" in source
    assert "cm-cell-pct" in source

    # Validate unified confusion matrix helpers and eval detail rendering
    assert "buildConfusionMatrixTableHtml" in source
    assert "buildPerClassTableHtml" in source
    assert "buildCmLendAndPickerHtml" in source
    eval_detail = source.split("function renderSweepEvalDetail", 1)[1].split("function openSweepRawJsonModal", 1)[0]
    assert "buildConfusionMatrixTableHtml(cm)" in eval_detail
    assert "buildPerClassTableHtml(cm.per_class" in eval_detail
    assert "buildCmLendAndPickerHtml()" in eval_detail
    assert "总体准确率 overall_accuracy:" in eval_detail
    assert "renderSweepEvalDetail(current)" in source
    assert "renderSweepSelectedDatasetDetail()" in source

