"""端到端:split -> run(fake) -> score,以及断点续跑、逐轮评分。"""
from __future__ import annotations

import json

from eval_vlm.data.splitter import split_dataset
from eval_vlm.runner import run_inference
from eval_vlm.evaluate import score_predictions
from eval_vlm.results import store


def test_full_pipeline(messages_config):
    cfg = messages_config
    split_dataset(cfg)
    stats = run_inference(cfg)
    assert stats["errors"] == 0
    assert cfg.predictions_path.exists()

    metrics = score_predictions(cfg)
    # 单轮:1 个目标轮(turn_0),fake 回显 reference -> exact_match 满分
    assert metrics["per_turn"]["turn_0"]["accuracy"] == 1.0
    assert metrics["overall_mean_score"] == 1.0
    assert cfg.metrics_path.exists()
    assert cfg.scored_path.exists()
    assert cfg.summary_path.exists()


def test_tworound_both_turns_pipeline(tworound_config):
    """两轮对话:轮1描述 + 轮2标签都评测,各自出分。

    标签集**从数据动态取出**,不在测试里硬编码。
    """
    cfg = tworound_config

    src = json.loads(cfg.source_path.read_text(encoding="utf-8"))
    labels = {rec["messages"][-1]["content"] for rec in src}

    split_dataset(cfg)
    # test.json 仍是 LlamaFactory 格式,最后一轮 assistant = 标签,取值来自数据
    test = json.loads(cfg.test_path.read_text(encoding="utf-8"))
    for rec in test:
        assert rec["messages"][-1]["role"] == "assistant"
        assert rec["messages"][-1]["content"] in labels

    stats = run_inference(cfg)
    # 每条样本两个目标轮(描述+标签)
    assert stats["num_targets"] == stats["test_size"] * 2

    # predictions 每样本两条,按 (id, turn) 唯一
    preds = store.load_predictions(cfg.predictions_path)
    keys = {(p.id, p.turn) for p in preds}
    assert len(keys) == stats["num_targets"]
    turns_per_sample = {}
    for p in preds:
        turns_per_sample.setdefault(p.id, set()).add(p.turn)
    assert all(t == {1, 3} for t in turns_per_sample.values())

    metrics = score_predictions(cfg)
    # 两轮均评:turn_0(描述)+ turn_1(标签);fake 回显标准答案 -> 均满分
    assert set(metrics["per_turn"]) == {"turn_0", "turn_1"}
    assert metrics["per_turn"]["turn_0"]["accuracy"] == 1.0
    assert metrics["per_turn"]["turn_1"]["accuracy"] == 1.0


def test_per_turn_scorers(tworound_config):
    """不同轮用不同 scorer:轮1描述 token_f1,轮2标签 exact_match。"""
    cfg = tworound_config
    cfg.scoring.turn_scorers = ["token_f1", "exact_match"]

    split_dataset(cfg)
    run_inference(cfg)
    metrics = score_predictions(cfg)

    assert metrics["per_turn"]["turn_0"]["scorer"] == "token_f1"
    assert metrics["per_turn"]["turn_1"]["scorer"] == "exact_match"
    # fake 回显标准答案:token_f1 的 f1 与 exact_match 的 accuracy 都满分
    assert metrics["per_turn"]["turn_0"]["f1"] == 1.0
    assert metrics["per_turn"]["turn_1"]["accuracy"] == 1.0


def test_resume_only_fills_missing(messages_config):
    cfg = messages_config
    split_dataset(cfg)
    run_inference(cfg)

    # 删掉预测文件最后一行,模拟中断
    lines = cfg.predictions_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) >= 2
    cfg.predictions_path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")

    before = len(store.load_prediction_keys(cfg.predictions_path))
    stats = run_inference(cfg)
    after = len(store.load_prediction_keys(cfg.predictions_path))

    # 单轮样本每条 1 个目标轮,删 1 行 -> 只补 1 条
    assert stats["newly_completed"] == 1
    assert after == before + 1


def test_inference_error_recorded_not_raised(messages_config, monkeypatch):
    cfg = messages_config
    split_dataset(cfg)

    from eval_vlm.inference.fake_backend import FakeBackend
    from eval_vlm.data.schema import Prediction

    def boom(self, context, images, sample_id, expected=None):
        return Prediction(id=sample_id, error="boom")

    monkeypatch.setattr(FakeBackend, "complete", boom)
    stats = run_inference(cfg)
    assert stats["errors"] == stats["num_targets"]
    # 评分不应崩溃,失败样本计 0 分
    metrics = score_predictions(cfg)
    assert metrics["overall_mean_score"] == 0.0
    assert metrics["per_turn"]["turn_0"]["accuracy"] == 0.0
