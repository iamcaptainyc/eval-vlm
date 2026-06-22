"""工作目录模型:全局配置 + 数据集文件夹自包含 + 一键 run+score。"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval_vlm import workspace
from eval_vlm.config import load_dataset_config
from eval_vlm.data.splitter import split_dataset
from eval_vlm.runner import run_inference
from eval_vlm.evaluate import score_predictions

FIXTURES = Path(__file__).parent / "fixtures"
SOURCE = FIXTURES / "llamafactory_demo.json"


@pytest.fixture
def temp_global(tmp_path, monkeypatch):
    """把全局配置指向临时文件,避免污染真实 ~/.eval_vlm。返回 workspace 路径。"""
    monkeypatch.setenv("EVAL_VLM_CONFIG", str(tmp_path / "global.yaml"))
    return tmp_path / "ws"


def test_global_config_autocreate_and_set(tmp_path, monkeypatch):
    monkeypatch.setenv("EVAL_VLM_CONFIG", str(tmp_path / "g.yaml"))
    cfg = workspace.load_global_config()                 # 缺失 -> 自动生成
    assert workspace.global_config_path().exists()
    assert set(("workspace", "media_root", "image_strip_prefix")) <= set(cfg)

    # set 保留注释、改值;null 写成 None
    workspace.set_global_value("workspace", "/tmp/myws")
    workspace.set_global_value("image_strip_prefix", None)   # None -> 写成 yaml null
    cfg2 = workspace.load_global_config()
    assert cfg2["workspace"] == "/tmp/myws"
    assert cfg2["image_strip_prefix"] is None
    assert "#" in workspace.global_config_path().read_text(encoding="utf-8")  # 注释仍在


def test_global_config_has_split_defaults(tmp_path, monkeypatch):
    monkeypatch.setenv("EVAL_VLM_CONFIG", str(tmp_path / "g.yaml"))
    cfg = workspace.load_global_config()
    assert cfg["split"]["train"] == 0.95 and cfg["split"]["test"] == 0.05
    assert cfg["split"]["seed"] == 42 and cfg["split"]["stratify_by"] is None


def test_set_split_default_coerces_and_preserves_comments(tmp_path, monkeypatch):
    monkeypatch.setenv("EVAL_VLM_CONFIG", str(tmp_path / "g.yaml"))
    workspace.set_global_value("split.train", "0.7")     # CLI 传字符串 -> 转 float
    workspace.set_global_value("split.test", "0.3")
    workspace.set_global_value("split.seed", "7")        # -> int
    cfg = workspace.load_global_config()
    assert cfg["split"]["train"] == 0.7 and isinstance(cfg["split"]["train"], float)
    assert cfg["split"]["test"] == 0.3
    assert cfg["split"]["seed"] == 7 and isinstance(cfg["split"]["seed"], int)
    # 注释仍在(嵌套行替换保留行尾注释)
    text = workspace.global_config_path().read_text(encoding="utf-8")
    assert "训练集比例" in text


def test_unknown_split_key_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("EVAL_VLM_CONFIG", str(tmp_path / "g.yaml"))
    with pytest.raises(KeyError):
        workspace.set_global_value("split.nope", "1")


def test_split_defaults_flow_into_dataset_and_cli_overrides(temp_global):
    ws = temp_global                          # fixture 已把 EVAL_VLM_CONFIG 指向临时文件
    workspace.set_global_value("split.train", "0.7")
    workspace.set_global_value("split.test", "0.3")
    gcfg = workspace.load_global_config()

    # 不传 --train -> 用全局默认
    folder = workspace.init_dataset(
        str(SOURCE), ws, media_root=str(FIXTURES), split_defaults=gcfg["split"],
    )
    cfg = load_dataset_config(folder)
    assert cfg.split.train == 0.7 and cfg.split.test == 0.3

    # 命令行覆盖优先于全局默认
    folder2 = workspace.init_dataset(
        str(SOURCE), ws, name="ds_override", media_root=str(FIXTURES),
        split_defaults=gcfg["split"], split_overrides={"train": 0.5, "test": 0.5},
    )
    cfg2 = load_dataset_config(folder2)
    assert cfg2.split.train == 0.5 and cfg2.split.test == 0.5


def test_init_dataset_creates_self_contained_folder(temp_global):
    ws = temp_global
    folder = workspace.init_dataset(
        str(SOURCE), ws, media_root=str(FIXTURES), split_overrides={"train": 0.6, "test": 0.4},
    )
    assert folder == (ws / "llamafactory_demo").resolve()
    assert (folder / "config.yaml").exists()

    cfg = load_dataset_config(folder)
    assert cfg.run_dir == folder                          # 产物目录钉到文件夹本身
    assert cfg.data.source.endswith("llamafactory_demo.json")
    assert cfg.data.media_root == str(FIXTURES)
    assert cfg.split.train == 0.6 and cfg.split.test == 0.4


def test_init_dataset_force_required_when_exists(temp_global):
    ws = temp_global
    workspace.init_dataset(str(SOURCE), ws, media_root=str(FIXTURES))
    with pytest.raises(FileExistsError):
        workspace.init_dataset(str(SOURCE), ws, media_root=str(FIXTURES))
    # --force 可重建
    folder = workspace.init_dataset(str(SOURCE), ws, media_root=str(FIXTURES), force=True)
    assert (folder / "config.yaml").exists()


def test_resolve_dataset_dir_by_name_and_path(temp_global):
    ws = temp_global
    folder = workspace.init_dataset(str(SOURCE), ws, media_root=str(FIXTURES))
    assert workspace.resolve_dataset_dir("llamafactory_demo", ws) == folder
    assert workspace.resolve_dataset_dir(str(folder), ws) == folder
    with pytest.raises(FileNotFoundError):
        workspace.resolve_dataset_dir("does_not_exist", ws)


def test_end_to_end_via_workspace(temp_global):
    """初始化 -> 分割 -> run(fake) -> score,产物全部落在数据集文件夹内。"""
    ws = temp_global
    folder = workspace.init_dataset(
        str(SOURCE), ws, media_root=str(FIXTURES), split_overrides={"train": 0.6, "test": 0.4},
    )
    cfg = load_dataset_config(folder)
    cfg.inference.backend = "fake"                        # 离线回显,等价部署后跑

    split_dataset(cfg)
    run_inference(cfg)
    metrics = score_predictions(cfg)

    for name in ("config.yaml", "split_meta.json", "test.json",
                 "predictions.jsonl", "metrics.json", "scored.jsonl",
                 "failures.md", "summary.md"):
        assert (folder / name).exists(), f"缺少产物 {name}"

    # fake 回显标准答案 -> 全命中,无未命中
    assert metrics["overall_mean_score"] == 1.0
    assert metrics["num_failed_samples"] == 0

    # 预测带原图地址(可追溯)
    first = json.loads((folder / "predictions.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert first["images"]
