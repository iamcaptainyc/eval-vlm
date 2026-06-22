"""splitter:三路划分、verbatim LlamaFactory 输出、确定性。"""
from __future__ import annotations

import json

from eval_vlm.data.loader import load_raw_records
from eval_vlm.data.splitter import split_dataset


def _read(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def test_split_produces_train_test_verbatim(messages_config):
    cfg = messages_config
    meta = split_dataset(cfg)

    # train/test 必出,val 比例为 0 -> 不产出
    assert cfg.train_path.exists()
    assert cfg.test_path.exists()
    assert not cfg.val_path.exists()
    assert meta["counts"]["train"] == 3
    assert meta["counts"]["test"] == 2
    assert meta["format"] == "llamafactory"

    src = load_raw_records(cfg.source_path)
    train = _read(cfg.train_path)
    test = _read(cfg.test_path)

    # 1) 是原样 LlamaFactory 记录(含完整 messages,assistant 答案仍在)
    rec = test[0]
    assert "messages" in rec and "images" in rec
    assert rec["messages"][-1]["role"] == "assistant"

    # 2) 逐条 verbatim:切出来的每条都等于源中的某条(图片路径也不变)
    for rec in train + test:
        assert rec in src

    # 3) train/test 不相交且覆盖全部
    assert len(train) + len(test) == len(src)


def test_split_optional_val(messages_config):
    cfg = messages_config
    cfg.split.train = 0.6
    cfg.split.val = 0.2
    cfg.split.test = 0.2
    meta = split_dataset(cfg)
    assert cfg.val_path.exists()
    assert meta["counts"]["val"] >= 1
    # 三份互不相交,合计 = 总数
    total = sum(meta["counts"].values())
    assert total == meta["total_samples"]


def test_split_deterministic(messages_config):
    a = split_dataset(messages_config)["indices"]
    b = split_dataset(messages_config)["indices"]
    assert a == b  # 同 seed 同结果


def test_split_seed_changes_partition(messages_config):
    messages_config.split.seed = 1
    a = split_dataset(messages_config)["indices"]["test"]
    messages_config.split.seed = 12345
    b = split_dataset(messages_config)["indices"]["test"]
    assert a != b


def test_image_paths_unchanged(messages_config):
    cfg = messages_config
    split_dataset(cfg)
    src = {r["images"][0] for r in load_raw_records(cfg.source_path)}
    test = _read(cfg.test_path)
    for rec in test:
        assert rec["images"][0] in src  # 路径原样保留


def test_train_is_loadable_llamafactory(messages_config):
    """train.json 能被同一个 LlamaFactory loader 解析(即合法 LF 格式)。"""
    from eval_vlm.data.loader import load_samples
    cfg = messages_config
    split_dataset(cfg)
    samples = load_samples(cfg, source=cfg.train_path)
    assert len(samples) == 3
    assert samples[0].reference is not None
