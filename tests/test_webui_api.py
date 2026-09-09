"""WebUI REST API 接口测试。"""
from __future__ import annotations

import json
from pathlib import Path
import pytest
from starlette.testclient import TestClient

from eval_vlm import workspace
from eval_vlm.config import load_dataset_config
from eval_vlm.data.splitter import split_dataset
from eval_vlm.results.store import write_json
from eval_vlm.webui.app import create_app
from eval_vlm.webui.settings import Settings

FIXTURES = Path(__file__).parent / "fixtures"
SOURCE = FIXTURES / "llamafactory_demo.json"


@pytest.fixture
def api_client(tmp_path, monkeypatch):
    cfg_file = tmp_path / "global.yaml"
    monkeypatch.setenv("EVAL_VLM_CONFIG", str(cfg_file))
    ws = tmp_path / "workspace"
    ws.mkdir()
    settings = Settings(workspace_dir=ws)

    # 创建一个真实测试图片文件
    img_file = ws / "sample.jpg"
    from PIL import Image
    im = Image.new("RGB", (100, 100), color="blue")
    im.save(img_file, format="JPEG")

    # 初始化数据集
    ds_dir = workspace.init_dataset(str(SOURCE), ws, media_root=str(ws))
    cfg = load_dataset_config(ds_dir)
    split_dataset(cfg)

    # 写入一条引用本地有效图片的记录到 test.json
    test_record = [
        {
            "messages": [
                {"role": "user", "content": "<image>\n请描述图片"},
                {"role": "assistant", "content": "这是一张蓝色的图"},
            ],
            "images": ["sample.jpg"],
        }
    ]
    cfg.test_path.write_text(json.dumps(test_record, ensure_ascii=False, indent=2), encoding="utf-8")

    app = create_app(settings)
    client = TestClient(app)
    return client, ds_dir.name, cfg, settings


def test_api_whoami(api_client):
    client, _, _, _ = api_client
    resp = client.get("/api/whoami")
    assert resp.status_code == 200
    assert "role" in resp.json()


def test_api_datasets_and_detail(api_client):
    client, ds_name, cfg, _ = api_client
    # 列表
    resp = client.get("/api/datasets")
    assert resp.status_code == 200
    ds_list = resp.json()
    assert len(ds_list) >= 1
    assert ds_list[0]["name"] == ds_name

    # 详情
    detail_resp = client.get(f"/api/datasets/{ds_name}")
    assert detail_resp.status_code == 200
    assert detail_resp.json()["name"] == ds_name


def test_api_samples_and_image_stream(api_client):
    client, ds_name, cfg, _ = api_client
    # 样本分页
    resp = client.get(f"/api/datasets/{ds_name}/samples?offset=0&limit=10")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 1
    sample = data["samples"][0]
    assert len(sample["images"]) == 1
    assert sample["images"][0]["exists"] is True

    # 原图
    img_url = sample["images"][0]["url"]
    img_resp = client.get(img_url)
    assert img_resp.status_code == 200

    # 缩略图模式
    thumb_resp = client.get(f"{img_url}&thumb=1")
    assert thumb_resp.status_code == 200
    assert thumb_resp.headers["content-type"] == "image/jpeg"


def test_api_image_path_traversal_protection(api_client):
    client, ds_name, _, _ = api_client
    # 尝试访问越界路径
    resp = client.get(f"/api/datasets/{ds_name}/image?ref=../../../../Windows/win.ini")
    assert resp.status_code in (400, 403, 404)


def test_api_config_read_and_update(api_client):
    client, ds_name, cfg, _ = api_client
    # 读
    r_resp = client.get(f"/api/datasets/{ds_name}/config")
    assert r_resp.status_code == 200

    # 写
    w_resp = client.put(
        f"/api/datasets/{ds_name}/config",
        json={"updates": [{"key": "scoring.scorer", "value": "contain"}]},
    )
    assert w_resp.status_code == 200
    assert w_resp.json()["success"] is True

    # 验证写回生效
    updated_cfg = load_dataset_config(cfg.dataset_dir)
    assert updated_cfg.scoring.scorer == "contain"


def test_api_auth_token_enforcement(tmp_path, monkeypatch):
    """验证配置了 Token 时未认证请求被拦截为 401。"""
    cfg_file = tmp_path / "global.yaml"
    monkeypatch.setenv("EVAL_VLM_CONFIG", str(cfg_file))
    monkeypatch.setenv("EVAL_VLM_WEBUI_TOKEN", "secret123")
    ws = tmp_path / "ws"
    ws.mkdir()
    settings = Settings(workspace_dir=ws)
    app = create_app(settings)
    client = TestClient(app)

    # 未附带 Token
    resp = client.get("/api/datasets")
    assert resp.status_code == 401

    # 附带正确 Bearer Token
    ok_resp = client.get("/api/datasets", headers={"Authorization": "Bearer secret123"})
    assert ok_resp.status_code == 200
