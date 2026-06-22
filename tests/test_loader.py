"""loader 解析两种 LlamaFactory 格式,并构建完整对话 + 评测目标轮。"""
from __future__ import annotations

import json

import pytest

from eval_vlm.data.loader import load_samples, DataFormatError


def test_load_messages_format(messages_config):
    samples = load_samples(messages_config)
    assert len(samples) == 5
    s0 = samples[0]
    assert s0.images == ["images/sample1.png"]
    # 单轮(user+assistant):完整对话两轮,assistant 即唯一目标轮
    assert [t.role for t in s0.turns] == ["user", "assistant"]
    assert "<image>" in s0.turns[0].content
    assert len(s0.targets) == 1
    assert s0.targets[0].turn_index == 1
    assert s0.targets[0].reference == "A cat."
    assert s0.reference == "A cat."          # 便捷属性 = 最后一个目标轮
    assert s0.meta.get("task_type") == "vqa"
    assert len({s.id for s in samples}) == 5


def test_tworound_targets_are_both_assistants(tworound_config):
    """两轮对话:targets 含 2 项——轮1描述 + 轮2标签。
    期望值从原始记录动态取出,不硬编码。"""
    src = json.loads(tworound_config.source_path.read_text(encoding="utf-8"))
    samples = load_samples(tworound_config)
    s = samples[0]
    # 完整对话保留:user, assistant(描述), user, assistant(标签)
    assert [t.role for t in s.turns] == ["user", "assistant", "user", "assistant"]
    assert "<image>" in s.turns[0].content

    raw_turns = src[0]["messages"]
    assert len(s.targets) == 2
    # 轮1描述 与 轮2标签 均从原始记录动态取出
    assert s.targets[0].reference == raw_turns[1]["content"]   # 描述
    assert s.targets[1].reference == raw_turns[-1]["content"]  # 标签
    assert s.reference == raw_turns[-1]["content"]


def test_targets_last_mode(tworound_config):
    """eval.targets=last 时仅评最后一轮(标签),退回旧行为。"""
    tworound_config.eval.targets = "last"
    src = json.loads(tworound_config.source_path.read_text(encoding="utf-8"))
    samples = load_samples(tworound_config)
    s = samples[0]
    assert len(s.targets) == 1
    assert s.targets[0].reference == src[0]["messages"][-1]["content"]
    # 完整对话仍保留(供构造上下文)
    assert len(s.turns) == 4


def test_load_conversations_format(conversations_config):
    samples = load_samples(conversations_config)
    assert len(samples) == 2
    assert samples[0].reference == "A cat."
    assert samples[0].turns[0].role == "user"  # human -> 归一化为 user


def test_image_placeholder_mismatch(messages_config, tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps([
        {"messages": [{"role": "user", "content": "no placeholder"},
                      {"role": "assistant", "content": "x"}],
         "images": ["a.png", "b.png"]}
    ]), encoding="utf-8")
    messages_config.data.source = str(bad)
    with pytest.raises(DataFormatError):
        load_samples(messages_config)
