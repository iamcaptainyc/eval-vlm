import json
from pathlib import Path
import pytest
from starlette.testclient import TestClient

from eval_vlm.webui.app import create_app
from eval_vlm.webui.settings import Settings


def test_api_sweep_runs_and_summary(tmp_path: Path):
    workspace = tmp_path / "ws"
    workspace.mkdir()

    sweep_dir = workspace / "_sweep" / "test_model" / "vllm_offline"
    sweep_dir.mkdir(parents=True)

    summary_data = {
        "datasets": ["ds1", "ds2"],
        "results": [
            {
                "dataset": "ds1",
                "method": "field-eval",
                "model": "test_model",
                "backend": "vllm_offline",
                "status": "ok",
                "metrics": {
                    "num_samples": 100,
                    "overall": {"micro_accuracy": 0.95, "exact_match_rate": 0.85},
                },
            }
        ],
        "num_ok": 1,
        "num_error": 0,
    }
    (sweep_dir / "summary.json").write_text(json.dumps(summary_data, ensure_ascii=False), encoding="utf-8")

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
