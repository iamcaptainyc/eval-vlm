"""验证多轮 rollout:第二轮的上下文用的是模型自己生成的第一轮输出。"""
from __future__ import annotations

import json

from eval_vlm.data.splitter import split_dataset
from eval_vlm.data.schema import Prediction
from eval_vlm.inference.fake_backend import FakeBackend
from eval_vlm.runner import run_inference


def _install_capturing_backend(monkeypatch):
    """让 fake 后端记录每次收到的 context,并返回可识别的模型输出。"""
    captured = []

    def complete(self, context, images, sample_id, expected=None):
        captured.append([(t.role, t.content) for t in context])
        # 返回与 gold 不同的内容,便于区分上下文来源
        n_user = sum(1 for t in context if t.role == "user")
        return Prediction(id=sample_id, prediction=f"MODEL_OUT_{n_user}")

    monkeypatch.setattr(FakeBackend, "complete", complete)
    return captured


def test_rollout_uses_model_own_first_turn(tworound_config, monkeypatch):
    cfg = tworound_config
    cfg.eval.context = "rollout"
    cfg.inference.max_concurrency = 1   # 串行,断言更稳

    src = json.loads(cfg.source_path.read_text(encoding="utf-8"))
    gold_descriptions = {rec["messages"][1]["content"] for rec in src}

    split_dataset(cfg)
    captured = _install_capturing_backend(monkeypatch)
    run_inference(cfg)

    # 找出"第二轮"的上下文(含一个历史 assistant 轮)
    round2_ctxs = [ctx for ctx in captured
                   if any(role == "assistant" for role, _ in ctx)]
    assert round2_ctxs, "应当存在第二轮调用"
    for ctx in round2_ctxs:
        assistant_contents = [c for role, c in ctx if role == "assistant"]
        # rollout:历史 assistant 内容应是模型自己的输出,而非数据集 gold 描述
        assert any(c.startswith("MODEL_OUT_") for c in assistant_contents)
        assert all(c not in gold_descriptions for c in assistant_contents)


def test_gold_context_uses_dataset_first_turn(tworound_config, monkeypatch):
    cfg = tworound_config
    cfg.eval.context = "gold"
    cfg.inference.max_concurrency = 1

    src = json.loads(cfg.source_path.read_text(encoding="utf-8"))
    gold_descriptions = {rec["messages"][1]["content"] for rec in src}

    split_dataset(cfg)
    captured = _install_capturing_backend(monkeypatch)
    run_inference(cfg)

    round2_ctxs = [ctx for ctx in captured
                   if any(role == "assistant" for role, _ in ctx)]
    assert round2_ctxs
    # gold 模式:历史 assistant 内容应是数据集标准描述
    for ctx in round2_ctxs:
        assistant_contents = [c for role, c in ctx if role == "assistant"]
        assert any(c in gold_descriptions for c in assistant_contents)
