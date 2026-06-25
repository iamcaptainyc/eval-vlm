"""工作目录模型:全局配置 + 数据集文件夹自包含 + 一键 run+score。"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval_vlm import workspace
from eval_vlm.config import load_dataset_config, safe_model_dirname
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


def test_describe_settable_keys_lists_all_and_marks_non_settable():
    """config keys 文案应覆盖全部可设置键,并提示数据集级(不可全局设)的项。"""
    text = workspace.describe_settable_keys()
    for key in workspace._all_keys():            # 每个可设置键都出现
        assert key in text
    # 每个可设置键都能被 set_global_value 接受(_all_keys 与校验保持一致)
    assert set(workspace._all_keys()) == set(
        workspace._TOP_KEYS + tuple(f"split.{k}" for k in workspace._SPLIT_DEFAULTS)
    )
    # 数据集级、不可全局设置的项也要标注出来
    assert "scoring.scorer" in text and "inference.backend" in text
    assert "需手改" in text


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
    assert cfg.dataset_dir == folder                      # 数据集文件夹钉到文件夹本身
    # 产物按模型分目录:工作目录/数据集/<inference.model>(默认 trained-vlm)
    assert cfg.run_dir == folder / "trained-vlm"
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

    # split 产物(各模型共享)落数据集文件夹本身
    for name in ("config.yaml", "split_meta.json", "test.json"):
        assert (folder / name).exists(), f"缺少数据集级产物 {name}"
    # run/score 产物按模型分目录:数据集/<inference.model>/
    mdir = folder / "trained-vlm"
    for name in ("predictions.jsonl", "metrics.json", "scored.jsonl",
                 "failures.md", "summary.md"):
        assert (mdir / name).exists(), f"缺少模型级产物 {name}"

    # fake 回显标准答案 -> 全命中,无未命中
    assert metrics["overall_mean_score"] == 1.0
    assert metrics["num_failed_samples"] == 0

    # 预测带原图地址(可追溯)
    first = json.loads((mdir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert first["images"]


# ---------------------------------------------------------------------------
# 按模型分目录:不同模型对同一数据集结果互不覆盖
# ---------------------------------------------------------------------------
def test_safe_model_dirname():
    """模型名里的路径分隔符/非法字符折成 '_';空回落 default。"""
    assert safe_model_dirname("Qwen/Qwen2-VL-7B") == "Qwen_Qwen2-VL-7B"
    assert safe_model_dirname("a:b*c?") == "a_b_c"
    assert safe_model_dirname("trained-vlm") == "trained-vlm"
    assert safe_model_dirname("") == "default"
    assert safe_model_dirname("  ") == "default"


def test_per_model_dirs_isolate_results(temp_global):
    """两个模型跑同一数据集:各自 数据集/<模型>/ 目录,互不覆盖;split 产物共享。"""
    ws = temp_global
    folder = workspace.init_dataset(
        str(SOURCE), ws, media_root=str(FIXTURES), split_overrides={"train": 0.6, "test": 0.4},
    )
    cfg = load_dataset_config(folder)
    cfg.inference.backend = "fake"
    split_dataset(cfg)
    # split 产物落数据集文件夹本身(共享)
    assert (folder / "test.json").exists()

    # 模型 A
    cfg.inference.openai.model = "model_a"
    run_inference(cfg)
    score_predictions(cfg)
    # 模型 B(同一 cfg,仅换模型名)
    cfg.inference.openai.model = "model_b"
    run_inference(cfg)
    score_predictions(cfg)

    a = folder / "model_a"
    b = folder / "model_b"
    assert (a / "predictions.jsonl").exists() and (a / "metrics.json").exists()
    assert (b / "predictions.jsonl").exists() and (b / "metrics.json").exists()
    # 互不覆盖:各自 metrics 记录自己的模型名
    assert json.loads((a / "metrics.json").read_text(encoding="utf-8"))["model"] == "model_a"
    assert json.loads((b / "metrics.json").read_text(encoding="utf-8"))["model"] == "model_b"


# ---------------------------------------------------------------------------
# CLI 覆盖永久写回 config.yaml(用户参数优先且持久化)
# ---------------------------------------------------------------------------
def test_set_dataset_value_persists_and_keeps_comments(temp_global):
    ws = temp_global
    folder = workspace.init_dataset(str(SOURCE), ws, media_root=str(FIXTURES))

    workspace.set_dataset_value(folder, "inference.openai.model", "Qwen/Qwen2-VL")
    workspace.set_dataset_value(folder, "inference.openai.base_url", "http://h:9/v1")
    workspace.set_dataset_value(folder, "scoring.scorer", "token_f1")

    cfg = load_dataset_config(folder)
    assert cfg.inference.openai.model == "Qwen/Qwen2-VL"  # 用户值优先且持久化
    assert cfg.inference.openai.base_url == "http://h:9/v1"
    assert cfg.scoring.scorer == "token_f1"
    # 注释保留(整行替换只改值)
    assert "#" in (folder / "config.yaml").read_text(encoding="utf-8")
    # 产物目录随写回的模型名走(非法字符折成 _)
    assert cfg.run_dir == folder / "Qwen_Qwen2-VL"


def test_set_dataset_value_missing_config_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        workspace.set_dataset_value(tmp_path / "nope", "inference.openai.model", "x")


def test_set_dataset_value_three_level_nested(temp_global):
    """3 层点号键(inference.openai.* / inference.mnn.*)能精确写回到对应子块,互不干扰。"""
    ws = temp_global
    folder = workspace.init_dataset(str(SOURCE), ws, media_root=str(FIXTURES))

    workspace.set_dataset_value(folder, "inference.backend", "mnn")
    workspace.set_dataset_value(folder, "inference.mnn.config_path", "/m/qwen-mnn/config.json")
    workspace.set_dataset_value(folder, "inference.mnn.image_max_side", 1024)
    workspace.set_dataset_value(folder, "inference.openai.model", "openai-model")

    cfg = load_dataset_config(folder)
    assert cfg.inference.backend == "mnn"
    assert cfg.inference.mnn.config_path == "/m/qwen-mnn/config.json"
    assert cfg.inference.mnn.image_max_side == 1024
    # 写 mnn 块没有污染 openai 块
    assert cfg.inference.openai.model == "openai-model"
    # mnn 后端产物子目录名取 config.json 所在目录名
    assert cfg.run_dir == folder / "qwen-mnn"
    # 注释保留
    assert "#" in (folder / "config.yaml").read_text(encoding="utf-8")


def test_mnn_result_name_falls_back_when_no_config_path():
    """mnn 后端未设 config_path 时产物子目录名回落 'mnn-model'。"""
    from eval_vlm.config import Config
    cfg = Config()
    cfg.inference.backend = "mnn"
    assert cfg.inference.result_name == "mnn-model"


def test_result_name_unknown_backend_raises():
    """未知后端:result_name 与 active 一致地报错,不伪装成 openai。"""
    from eval_vlm.config import Config
    cfg = Config()
    cfg.inference.backend = "nope"
    with pytest.raises(ValueError):
        _ = cfg.inference.result_name


def test_set_dataset_value_reinserts_deleted_subblock(temp_global):
    """用户手删 openai 子块后,写回 inference.openai.* 应补回**到 inference 块内**
    (而非追加到文末造成非法 YAML),且能被重新加载。"""
    import yaml

    ws = temp_global
    folder = workspace.init_dataset(str(SOURCE), ws, media_root=str(FIXTURES))
    config_path = folder / "config.yaml"

    # 删掉整个 openai: 子块(连同其缩进子行),保留 inference: 与 mnn:
    lines = config_path.read_text(encoding="utf-8").splitlines(keepends=True)
    kept, skipping = [], False
    for ln in lines:
        stripped = ln.strip()
        if stripped.startswith("openai:"):
            skipping = True
            continue
        if skipping:
            # 子块行(缩进 > 2)或块内注释跳过;遇到下一个缩进<=2 的非空行结束跳过
            indent = len(ln) - len(ln.lstrip(" "))
            if stripped == "" or indent > 2:
                continue
            skipping = False
        kept.append(ln)
    config_path.write_text("".join(kept), encoding="utf-8")
    assert "openai:" not in config_path.read_text(encoding="utf-8")

    workspace.set_dataset_value(folder, "inference.openai.model", "reinserted")

    # 仍是合法 YAML,且值进了 inference.openai.model
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert data["inference"]["openai"]["model"] == "reinserted"
    assert load_dataset_config(folder).inference.openai.model == "reinserted"
