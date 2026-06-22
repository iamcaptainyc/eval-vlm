"""验证两项可追溯/审核能力:
  1) predictions.jsonl 每条预测都带原图地址(images),可追溯回原图。
  2) score 阶段自动产出 failures.md:仅 exact_match 未命中的样本,按 id 分组列出
     全部对话轮,人类可读,供人工审核。非 exact_match 评分(token_f1)不计入。
"""
from __future__ import annotations

import json

from eval_vlm.data.schema import Prediction
from eval_vlm.data.splitter import split_dataset
from eval_vlm.inference.fake_backend import FakeBackend
from eval_vlm.runner import run_inference
from eval_vlm.evaluate import score_predictions
from eval_vlm.results import store


def test_predictions_carry_original_images(messages_config):
    cfg = messages_config
    split_dataset(cfg)
    run_inference(cfg)

    preds = store.load_predictions(cfg.predictions_path)
    assert preds
    # 每条预测都应带非空 images,且能在 test.json 的样本里找到对应原图引用
    test = json.loads(cfg.test_path.read_text(encoding="utf-8"))
    valid_imgs = {img for rec in test for img in rec["images"]}
    for p in preds:
        assert p.images, f"预测 {p.id} 缺少原图地址"
        assert all(img in valid_imgs for img in p.images)

    # 落盘的 jsonl 原始行里也确实含 images 字段(可追溯)
    first = json.loads(cfg.predictions_path.read_text(encoding="utf-8").splitlines()[0])
    assert "images" in first and first["images"]


def test_failures_md_groups_wrong_exact_match(tworound_config, monkeypatch):
    """强制错误答案 -> 每个样本(含两轮)exact_match 未命中,按 id 分组进 failures.md。"""
    cfg = tworound_config
    split_dataset(cfg)

    def wrong(self, context, images, sample_id, expected=None):
        return Prediction(id=sample_id, prediction="__definitely_wrong__")

    monkeypatch.setattr(FakeBackend, "complete", wrong)
    run_inference(cfg)
    metrics = score_predictions(cfg)

    # 以 id 为单位:所有样本都因 exact_match 错误被纳入;错误目标轮 = 全部目标轮
    assert metrics["num_failed_samples"] == metrics["num_samples"]
    assert metrics["num_failed_targets"] == metrics["num_targets"]
    assert cfg.failures_path.exists() and cfg.failures_path.name == "failures.md"

    md = cfg.failures_path.read_text(encoding="utf-8")
    assert "## 样本" in md                       # 按 id 分组的标题
    assert "✗ 未命中" in md                       # 命中标记
    assert "__definitely_wrong__" in md          # 模型输出
    assert "请描述这张图片" in md                 # 含完整对话上下文(user 轮)
    # 同一样本的两个目标轮(描述 + 标签)都在 -> 分组到一起
    assert md.count("scorer: `exact_match`") >= metrics["num_targets"]


def test_no_failures_when_all_correct(messages_config):
    """fake 回显标准答案 -> 全部命中 -> failures.md 标注无未命中。"""
    cfg = messages_config
    split_dataset(cfg)
    run_inference(cfg)
    metrics = score_predictions(cfg)

    assert metrics["num_failed_samples"] == 0
    assert metrics["num_failed_targets"] == 0
    assert "无 exact_match 未命中" in cfg.failures_path.read_text(encoding="utf-8")


def test_non_exact_match_scorer_not_in_failures(messages_config, monkeypatch):
    """token_f1 评分即使分数<1,也不计入 failures(只看 exact_match)。"""
    cfg = messages_config
    cfg.scoring.scorer = "token_f1"
    split_dataset(cfg)

    def wrong(self, context, images, sample_id, expected=None):
        return Prediction(id=sample_id, prediction="完全不同的答案")

    monkeypatch.setattr(FakeBackend, "complete", wrong)
    run_inference(cfg)
    metrics = score_predictions(cfg)

    assert metrics["overall_mean_score"] < 1.0      # 确有低分
    assert metrics["num_failed_samples"] == 0       # 但非 exact_match -> 不计入
    assert "无 exact_match 未命中" in cfg.failures_path.read_text(encoding="utf-8")
