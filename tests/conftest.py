"""测试共享夹具。"""
from __future__ import annotations

from pathlib import Path

import pytest

from eval_vlm.config import load_config, Config

FIXTURES = Path(__file__).parent / "fixtures"


def _make_config(tmp_path: Path, source: str, mapping: dict | None = None,
                 **overrides) -> Config:
    cfg = load_config(FIXTURES.parent.parent / "configs" / "example.yaml")
    # 指向 fixture 数据(绝对路径,与 CWD 无关),输出到临时目录,默认离线 fake 后端。
    cfg.data.source = str(FIXTURES / source)
    cfg.data.media_root = str(FIXTURES)
    cfg.output_dir = str(tmp_path)
    cfg.run_name = "test_run"
    cfg.inference.backend = "fake"
    if mapping:
        from eval_vlm.config import Mapping, Tags
        cfg.data.mapping = Mapping(
            messages=mapping["messages"],
            images=mapping["images"],
            tags=Tags(**mapping["tags"]),
        )
    for k, v in overrides.items():
        setattr(cfg.split, k, v)
    return cfg


@pytest.fixture
def messages_config(tmp_path):
    # 5 条 -> train 3 / test 2
    return _make_config(tmp_path, "llamafactory_demo.json",
                        train=0.6, test=0.4, val=0.0)


@pytest.fixture
def tworound_config(tmp_path):
    # 两轮对话(图像描述 -> 情绪标签),最后一轮 assistant 标签为评测目标
    return _make_config(tmp_path, "llamafactory_tworound.json",
                        train=0.5, test=0.5, val=0.0)


@pytest.fixture
def conversations_config(tmp_path):
    return _make_config(
        tmp_path, "llamafactory_conversations.json",
        mapping={"messages": "conversations", "images": "images",
                 "tags": {"role": "from", "content": "value",
                          "user": "human", "assistant": "gpt"}},
        train=0.5, test=0.5, val=0.0,
    )
