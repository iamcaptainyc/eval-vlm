"""WebUI field-eval 评测指标与透明化任务队列测试。"""
from __future__ import annotations

import json
from pathlib import Path
import pytest
from starlette.testclient import TestClient

from eval_vlm import workspace
from eval_vlm.config import load_dataset_config
from eval_vlm.data.splitter import split_dataset
from eval_vlm.webui.app import create_app
from eval_vlm.webui.jobs import JobManager
from eval_vlm.webui.settings import Settings

FIXTURES = Path(__file__).parent / "fixtures"
SOURCE = FIXTURES / "llamafactory_demo.json"


@pytest.fixture
def field_eval_env(tmp_path, monkeypatch):
    cfg_file = tmp_path / "global.yaml"
    monkeypatch.setenv("EVAL_VLM_CONFIG", str(cfg_file))
    ws = tmp_path / "workspace"
    ws.mkdir()
    settings = Settings(workspace_dir=ws)

    ds_dir = workspace.init_dataset(str(SOURCE), ws, media_root=str(ws))
    cfg = load_dataset_config(ds_dir)
    split_dataset(cfg)

    # 构造一个含 field-eval 产物的运行目录: <dataset>/qwen2-vl/openai
    run_dir = ds_dir / "qwen2-vl" / "openai"
    run_dir.mkdir(parents=True, exist_ok=True)

    field_metrics = {
        "fields": ["主辅路", "道路结构", "车道位置", "警示标志"],
        "num_samples": 10,
        "num_scored": 10,
        "num_pred_missing": 0,
        "overall": {
            "micro_accuracy": 0.925,
            "macro_accuracy": 0.900,
            "exact_match_samples": 8,
            "exact_match_rate": 0.800,
        },
        "per_field": {
            "主辅路": {"correct": 10, "total": 10, "accuracy": 1.0},
            "道路结构": {"correct": 9, "total": 10, "accuracy": 0.9},
            "车道位置": {"correct": 9, "total": 10, "accuracy": 0.9},
            "警示标志": {"correct": 9, "total": 10, "accuracy": 0.9},
        },
    }
    (run_dir / "field_metrics.json").write_text(json.dumps(field_metrics, ensure_ascii=False), encoding="utf-8")

    mismatches = [
        {
            "id": "sample-001",
            "images": ["sample.jpg"],
            "state": "mismatch",
            "pred_desc": "主路上有一辆车，无警示标志",
            "fields": [
                {"field": "主辅路", "ref": ["主路"], "pred": ["主路"], "correct": True},
                {"field": "道路结构", "ref": ["高架路"], "pred": ["地面道路"], "correct": False},
            ],
        }
    ]
    (run_dir / "field_mismatches.json").write_text(json.dumps(mismatches, ensure_ascii=False), encoding="utf-8")
    (run_dir / "field_mismatches.html").write_text("<html><body>Field Mismatches Report</body></html>", encoding="utf-8")

    app = create_app(settings)
    client = TestClient(app)
    return {"client": client, "settings": settings, "ds_name": ds_dir.name, "run_dir": run_dir}


def test_runs_list_detects_field_eval(field_eval_env):
    client = field_eval_env["client"]
    ds = field_eval_env["ds_name"]

    res = client.get(f"/api/datasets/{ds}/runs")
    assert res.status_code == 200
    runs = res.json()
    assert len(runs) == 1
    r = runs[0]
    assert r["model"] == "qwen2-vl"
    assert r["backend"] == "openai"
    assert r["has_field_eval"] is True
    assert r["has_field_metrics"] is True
    assert r["has_field_mismatches_html"] is True
    assert r["has_field_mismatches_json"] is True
    assert r["field_metrics_summary"] is not None
    assert r["field_metrics_summary"]["micro_accuracy"] == 0.925
    assert r["field_metrics_summary"]["exact_match_rate"] == 0.8


def test_field_metrics_and_mismatches_api(field_eval_env):
    client = field_eval_env["client"]
    ds = field_eval_env["ds_name"]

    # 1. 详细指标 API
    res_m = client.get(f"/api/datasets/{ds}/runs/qwen2-vl/openai/field-metrics")
    assert res_m.status_code == 200
    m_data = res_m.json()
    assert m_data["overall"]["macro_accuracy"] == 0.9
    assert len(m_data["fields"]) == 4

    # 2. 失配清单 API
    res_mis = client.get(f"/api/datasets/{ds}/runs/qwen2-vl/openai/field-mismatches?offset=0&limit=10")
    assert res_mis.status_code == 200
    mis_data = res_mis.json()
    assert mis_data["total"] == 1
    assert mis_data["records"][0]["id"] == "sample-001"

    # 3. HTML 报告
    res_html = client.get(f"/api/datasets/{ds}/runs/qwen2-vl/openai/field-mismatches.html")
    assert res_html.status_code == 200
    assert "Field Mismatches Report" in res_html.text


def test_job_command_construction_with_parameters(field_eval_env):
    settings = field_eval_env["settings"]
    ds = field_eval_env["ds_name"]
    mgr = JobManager(settings)

    # 提交带各种参数的 field-eval 任务
    summary = mgr.submit_job(
        job_type="field-eval",
        dataset=ds,
        params={
            "match_mode": "contain",
            "targets": "first",
            "limit": 5,
            "fail_fast": True,
            "overwrite": True,
            "backend": "openai",
        },
        user="test_engineer",
    )

    job_obj = mgr.jobs[summary.id]
    cmd = mgr._build_cmd(job_obj)
    cmd_str = " ".join(cmd)

    # 验证 CLI 命令拼装正确
    assert "-m eval_vlm field-eval" in cmd_str
    assert f"-d {ds}" in cmd_str
    assert "--match-mode contain" in cmd_str
    assert "--targets first" in cmd_str
    assert "--limit 5" in cmd_str
    assert "--fail-fast" in cmd_str
    assert "--overwrite" in cmd_str
    assert "--backend openai" in cmd_str

    # 验证 summary 与 meta 保存了命令及日志物理路径
    assert summary.log_file is not None
    assert str(job_obj.log_file.resolve()) in summary.log_file
    assert summary.params["limit"] == 5
