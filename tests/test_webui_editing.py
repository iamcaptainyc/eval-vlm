"""WebUI 核心测试: 样本删除、单图删除、split_meta 同步、回收站与恢复。"""
from __future__ import annotations

import json
from pathlib import Path
import pytest
from fastapi import HTTPException

from eval_vlm import workspace
from eval_vlm.config import load_dataset_config
from eval_vlm.data.loader import load_raw_records, load_samples, _stable_id
from eval_vlm.data.splitter import split_dataset, load_split_meta
from eval_vlm.results.store import write_json
from eval_vlm.webui.editing import delete_sample, list_trash, restore_sample
from eval_vlm.webui.locks import calc_file_sha256
from eval_vlm.webui.settings import Settings

FIXTURES = Path(__file__).parent / "fixtures"
SOURCE = FIXTURES / "llamafactory_demo.json"


@pytest.fixture
def test_env(tmp_path, monkeypatch):
    """构建独立临时的 workspace 和全局配置。"""
    cfg_file = tmp_path / "global.yaml"
    monkeypatch.setenv("EVAL_VLM_CONFIG", str(cfg_file))
    ws = tmp_path / "workspace"
    ws.mkdir()
    settings = Settings(workspace_dir=ws)

    # 初始化数据集并执行 split
    ds_dir = workspace.init_dataset(str(SOURCE), ws, media_root=str(FIXTURES))
    workspace.set_dataset_value(ds_dir, "split.train", 0.5)
    workspace.set_dataset_value(ds_dir, "split.test", 0.5)
    workspace.set_dataset_value(ds_dir, "split.val", 0.0)
    cfg = load_dataset_config(ds_dir)
    split_dataset(cfg)

    # 模拟一个下游运行结果目录
    run_dir = ds_dir / "test_model" / "openai"
    run_dir.mkdir(parents=True)
    write_json(run_dir / "metrics.json", {"overall_mean_score": 0.85, "num_samples": 5})
    (run_dir / "predictions.jsonl").write_text("{}", encoding="utf-8")

    return cfg, settings


def test_delete_sample_whole_record(test_env):
    cfg, settings = test_env
    test_path = cfg.test_path
    records_before = load_raw_records(test_path)
    count_before = len(records_before)
    target_pos = 1
    target_id = _stable_id(target_pos, records_before[target_pos])
    sha_before = calc_file_sha256(test_path)

    # 执行整条删除
    resp = delete_sample(
        cfg=cfg,
        settings=settings,
        sample_id=target_id,
        expected_sha256=sha_before,
        user="alice",
        reason="bad sample",
    )

    assert resp.success is True
    assert resp.mode == "record"
    assert resp.deleted_id == target_id
    assert resp.new_sha256 != sha_before
    assert "test_model/openai" in resp.invalidated_runs

    # 验证 test.json 记录数减少 1 且字节排版一致 (indent=2)
    records_after = load_raw_records(test_path)
    assert len(records_after) == count_before - 1
    raw_text = test_path.read_text(encoding="utf-8")
    assert raw_text.startswith("[\n  {")

    # 验证 split_meta 同步
    meta = load_split_meta(cfg)
    assert meta["counts"]["test"] == len(records_after)
    assert len(meta["indices"]["test"]) == len(records_after)
    assert meta["test_modified_after_split"] is True
    assert len(meta["webui_edits"]) == 1
    assert meta["webui_edits"][0]["user"] == "alice"

    # 验证下游 run 被标记为 dirty
    dirty_file = cfg.dataset_dir / "test_model" / "openai" / "dataset_dirty.json"
    assert dirty_file.exists()
    dirty_data = json.loads(dirty_file.read_text(encoding="utf-8"))
    assert dirty_data["by"] == "alice"
    assert target_id in dirty_data["deleted_ids"]

    # 验证回收站记录存在
    trash_list = list_trash(cfg, settings)
    assert len(trash_list) == 1
    assert trash_list[0]["sample_id"] == target_id

    # 验证恢复操作
    restore_resp = restore_sample(
        cfg=cfg,
        settings=settings,
        trash_id=trash_list[0]["trash_id"],
        expected_sha256=resp.new_sha256,
        user="alice",
    )
    assert restore_resp.success is True
    records_restored = load_raw_records(test_path)
    assert len(records_restored) == count_before
    assert _stable_id(target_pos, records_restored[target_pos]) == target_id


def test_delete_sample_optimistic_sha_conflict(test_env):
    cfg, settings = test_env
    test_path = cfg.test_path
    records = load_raw_records(test_path)
    target_id = _stable_id(0, records[0])

    # 传入错误的 expected_sha256
    with pytest.raises(HTTPException) as exc_info:
        delete_sample(
            cfg=cfg,
            settings=settings,
            sample_id=target_id,
            expected_sha256="wrong_sha256_hash",
        )
    assert exc_info.value.status_code == 409


def test_delete_single_image_mode(test_env):
    cfg, settings = test_env
    test_path = cfg.test_path

    # 造一个包含多图的样本记录写入 test.json
    multi_img_record = {
        "messages": [
            {"role": "user", "content": "<image>\n第一张图说明\n<image>\n第二张图说明"},
            {"role": "assistant", "content": "好的两张图解答"},
        ],
        "images": ["img1.jpg", "img2.jpg"],
    }
    records = [multi_img_record]
    test_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    sha_before = calc_file_sha256(test_path)
    target_id = _stable_id(0, multi_img_record)

    # 删除第 1 张图 (image_index=0)
    resp = delete_sample(
        cfg=cfg,
        settings=settings,
        sample_id=target_id,
        expected_sha256=sha_before,
        mode="image",
        image_index=0,
    )
    assert resp.success is True
    assert resp.mode == "image"

    # 验证结果
    records_after = load_raw_records(test_path)
    assert len(records_after) == 1
    assert len(records_after[0]["images"]) == 1
    assert records_after[0]["images"][0] == "img2.jpg"

    # 验证 <image> 占位符被精确剔除一个
    user_content = records_after[0]["messages"][0]["content"]
    assert user_content.count("<image>") == 1
    # 验证 load_samples 不会报占位符不匹配
    samples = load_samples(cfg, source=test_path)
    assert len(samples) == 1
