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


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.168.1.10", "webui.example.test"])
def test_api_non_loopback_host_requires_auth(tmp_path, monkeypatch, host):
    """LAN/public binds cannot fall back to the anonymous editor account."""
    cfg_file = tmp_path / "global.yaml"
    monkeypatch.setenv("EVAL_VLM_CONFIG", str(cfg_file))
    monkeypatch.delenv("EVAL_VLM_WEBUI_TOKEN", raising=False)
    ws = tmp_path / "ws"
    ws.mkdir()
    app = create_app(Settings(workspace_dir=ws, host=host))
    response = TestClient(app).get("/api/whoami")
    assert response.status_code == 401
    assert "EVAL_VLM_WEBUI_TOKEN" in response.json()["detail"]


@pytest.mark.parametrize("host", ["127.0.0.1", "::1", "localhost"])
def test_api_loopback_host_allows_local_anonymous_development(tmp_path, monkeypatch, host):
    cfg_file = tmp_path / "global.yaml"
    monkeypatch.setenv("EVAL_VLM_CONFIG", str(cfg_file))
    monkeypatch.delenv("EVAL_VLM_WEBUI_TOKEN", raising=False)
    ws = tmp_path / "ws"
    ws.mkdir()
    app = create_app(Settings(workspace_dir=ws, host=host))
    response = TestClient(app).get("/api/whoami")
    assert response.status_code == 200
    assert response.json() == {"username": "anonymous", "role": "editor"}


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


def test_api_dataset_html_reports(api_client):
    """验证数据集目录下 HTML 文件检索、在线打开/预览以及别名支持与防越界。"""
    client, ds_name, cfg, _ = api_client

    # 1. 初始状态下无 HTML 文件
    res_empty = client.get(f"/api/datasets/{ds_name}/html-files")
    assert res_empty.status_code == 200
    assert res_empty.json() == []

    # 2. 在运行目录下创建 failures.html 与 field_mismatches.html
    run_dir = cfg.dataset_dir / "qwen2-vl" / "hf"
    run_dir.mkdir(parents=True, exist_ok=True)
    failures_file = run_dir / "failures.html"
    failures_file.write_text("<html><body><h1>Failures Report</h1></body></html>", encoding="utf-8")
    field_file = run_dir / "field_mismatches.html"
    field_file.write_text("<html><body><h1>Field Mismatches</h1></body></html>", encoding="utf-8")

    # 3. 检索全部 HTML 文件
    res_list = client.get(f"/api/datasets/{ds_name}/html-files")
    assert res_list.status_code == 200
    files = res_list.json()
    assert len(files) == 2
    names = [f["name"] for f in files]
    assert "failures.html" in names
    assert "field_mismatches.html" in names

    # 4. 通过 html-view 访问
    rel_path = f"qwen2-vl/hf/failures.html"
    res_view = client.get(f"/api/datasets/{ds_name}/html-view?path={rel_path}")
    assert res_view.status_code == 200
    assert "Failures Report" in res_view.text
    assert "text/html" in res_view.headers.get("content-type", "")

    # 5. 测试 failures.html 及 failure.html 别名端点
    res_run_fail1 = client.get(f"/api/datasets/{ds_name}/runs/qwen2-vl/hf/failures.html")
    assert res_run_fail1.status_code == 200
    assert "Failures Report" in res_run_fail1.text

    res_run_fail2 = client.get(f"/api/datasets/{ds_name}/runs/qwen2-vl/hf/failure.html")
    assert res_run_fail2.status_code == 200
    assert "Failures Report" in res_run_fail2.text

    # 6. 测试数据集顶层 failure.html / failures.html 回退访问
    res_root_fail = client.get(f"/api/datasets/{ds_name}/failure.html")
    assert res_root_fail.status_code == 200
    assert "Failures Report" in res_root_fail.text

    # 7. 测试 field-mismatches.html 及 field-mismatch.html 别名端点
    res_fm1 = client.get(f"/api/datasets/{ds_name}/runs/qwen2-vl/hf/field-mismatches.html")
    assert res_fm1.status_code == 200
    assert "Field Mismatches" in res_fm1.text

    res_fm2 = client.get(f"/api/datasets/{ds_name}/runs/qwen2-vl/hf/field-mismatch.html")
    assert res_fm2.status_code == 200
    assert "Field Mismatches" in res_fm2.text

    # 8. 路径穿越防护验证
    res_traverse = client.get(f"/api/datasets/{ds_name}/html-view?path=../../forbidden.html")
    assert res_traverse.status_code in (400, 403)


def test_api_dataset_html_rejects_same_prefix_sibling(api_client):
    """A sibling named like the dataset must not pass the containment check."""
    client, ds_name, cfg, _ = api_client
    escaped_dir = cfg.dataset_dir.parent / f"{cfg.dataset_dir.name}_backup"
    escaped_dir.mkdir()
    (escaped_dir / "failure.html").write_text("outside dataset", encoding="utf-8")

    resp = client.get(
        f"/api/datasets/{ds_name}/html-view?path=../{escaped_dir.name}/failure.html"
    )
    assert resp.status_code == 403


def test_api_rejects_unknown_job_type(api_client):
    client, _, _, _ = api_client
    resp = client.post("/api/jobs", json={"type": "shell", "params": {}})
    assert resp.status_code == 422
