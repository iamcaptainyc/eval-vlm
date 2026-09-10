"""WebUI 全局设置与模型扫描 API 测试。"""
from __future__ import annotations

from pathlib import Path
import pytest
from starlette.testclient import TestClient

from eval_vlm.webui.app import create_app
from eval_vlm.webui.settings import Settings


@pytest.fixture
def settings_client(tmp_path, monkeypatch):
    cfg_file = tmp_path / "global.yaml"
    monkeypatch.setenv("EVAL_VLM_CONFIG", str(cfg_file))
    ws = tmp_path / "workspace"
    ws.mkdir()
    settings = Settings(workspace_dir=ws)

    # 创建测试模型目录
    hf_dir = tmp_path / "hf_models"
    hf_dir.mkdir()
    m1 = hf_dir / "Qwen2-VL-7B"
    m1.mkdir()
    (m1 / "config.json").write_text("{}", encoding="utf-8")

    mnn_dir = tmp_path / "mnn_models"
    mnn_dir.mkdir()
    m2 = mnn_dir / "qwen2-vl.mnn"
    m2.write_bytes(b"dummy mnn binary")

    app = create_app(settings)
    client = TestClient(app)
    return client, settings, hf_dir, mnn_dir


def test_settings_and_models_flow(settings_client):
    client, settings, hf_dir, mnn_dir = settings_client

    # 1. 获取默认设置
    resp = client.get("/api/settings")
    assert resp.status_code == 200
    data = resp.json()
    assert "workspace" in data
    assert data["hf_models_dir"] is None
    assert "split" in data
    assert data["split"]["train"] == 0.95
    assert data["split"]["test"] == 0.05

    # 2. 更新设置 (包含 split 与路径)
    resp = client.put(
        "/api/settings",
        json={
            "hf_models_dir": str(hf_dir),
            "mnn_models_dir": str(mnn_dir),
            "split": {
                "train": 0.8,
                "test": 0.2,
                "seed": 999,
            },
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["hf_models_dir"] == str(hf_dir)
    assert data["mnn_models_dir"] == str(mnn_dir)
    assert data["split"]["train"] == 0.8
    assert data["split"]["test"] == 0.2
    assert data["split"]["seed"] == 999

    # 3. 获取模型列表并验证扫描成功
    resp = client.get("/api/models")
    assert resp.status_code == 200
    models_data = resp.json()
    assert len(models_data["hf_models"]) == 1
    assert models_data["hf_models"][0]["name"] == "Qwen2-VL-7B"
    assert len(models_data["mnn_models"]) == 1
    assert models_data["mnn_models"][0]["name"] == "qwen2-vl.mnn"

    # 4. 创建 sweep job 支持指定多数据集
    resp = client.post(
        "/api/sweep/jobs",
        json={
            "type": "sweep",
            "dataset": "datasetA,datasetB",
            "params": {"backend": "vllm", "model": str(hf_dir / "Qwen2-VL-7B")},
        },
    )
    assert resp.status_code == 200
    job = resp.json()
    assert job["type"] == "sweep"
    assert job["dataset"] == "datasetA,datasetB"
    assert "-d" in job["command"]
    d_idx = job["command"].index("-d")
    assert job["command"][d_idx + 1] == "datasetA,datasetB"
