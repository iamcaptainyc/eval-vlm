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


def test_api_config_all_backends_full_parameters(api_client):
    """验证所有推理引擎后端（MNN 完整参数、OpenAI 扩展参数、vLLM Offline 批处理参数）的读写配置持久化。"""
    client, ds_name, cfg, _ = api_client

    updates = [
        {"key": "inference.backend", "value": "mnn"},
        # MNN 完整参数
        {"key": "inference.mnn.config_path", "value": "/models/mnn/config.json"},
        {"key": "inference.mnn.quant", "value": "int4"},
        {"key": "inference.mnn.system_prompt", "value": "You are an expert evaluator."},
        {"key": "inference.mnn.max_tokens", "value": 2048},
        {"key": "inference.mnn.temperature", "value": 0.7},
        {"key": "inference.mnn.top_p", "value": 0.85},
        {"key": "inference.mnn.top_k", "value": 40},
        {"key": "inference.mnn.repetition_penalty", "value": 1.15},
        {"key": "inference.mnn.frequency_penalty", "value": 0.3},
        {"key": "inference.mnn.presence_penalty", "value": 0.2},
        {"key": "inference.mnn.penalty_window", "value": 128},
        {"key": "inference.mnn.image_max_side", "value": 1024},
        {"key": "inference.mnn.image_max_pixels", "value": 1048576},
        {"key": "inference.mnn.image_min_pixels", "value": 4096},
        # OpenAI 补充参数
        {"key": "inference.openai.api_key_env", "value": "CUSTOM_OPENAI_KEY"},
        {"key": "inference.openai.top_p", "value": 0.95},
        {"key": "inference.openai.request_timeout", "value": 90.0},
        {"key": "inference.openai.max_retries", "value": 5},
        # vLLM Offline 补充参数
        {"key": "inference.vllm_offline.max_num_batched_tokens", "value": 8192},
    ]

    w_resp = client.put(f"/api/datasets/{ds_name}/config", json={"updates": updates})
    assert w_resp.status_code == 200
    assert w_resp.json()["success"] is True

    # 通过 GET /api/datasets/{ds_name}/config 读回
    r_resp = client.get(f"/api/datasets/{ds_name}/config")
    assert r_resp.status_code == 200
    cfg_data = r_resp.json().get("config", {})

    inf = cfg_data.get("inference", {})
    assert inf.get("backend") == "mnn"

    mnn = inf.get("mnn", {})
    assert mnn.get("config_path") == "/models/mnn/config.json"
    assert mnn.get("quant") == "int4"
    assert mnn.get("system_prompt") == "You are an expert evaluator."
    assert mnn.get("max_tokens") == 2048
    assert mnn.get("temperature") == 0.7
    assert mnn.get("top_p") == 0.85
    assert mnn.get("top_k") == 40
    assert mnn.get("repetition_penalty") == 1.15
    assert mnn.get("frequency_penalty") == 0.3
    assert mnn.get("presence_penalty") == 0.2
    assert mnn.get("penalty_window") == 128
    assert mnn.get("image_max_side") == 1024
    assert mnn.get("image_max_pixels") == 1048576
    assert mnn.get("image_min_pixels") == 4096

    openai = inf.get("openai", {})
    assert openai.get("api_key_env") == "CUSTOM_OPENAI_KEY"
    assert openai.get("top_p") == 0.95
    assert openai.get("request_timeout") == 90.0
    assert openai.get("max_retries") == 5

    vllm = inf.get("vllm_offline", {})
    assert vllm.get("max_num_batched_tokens") == 8192

