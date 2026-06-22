"""验证两项可追溯/审核能力:
  1) predictions.jsonl 每条预测都带原图地址(images),可追溯回原图。
  2) score 阶段自动产出 failures.jsonl:exact_match 未命中(含缺失/报错)的清单,
     且每条带原图地址,供人工审核。
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


def test_failures_list_lists_wrong_exact_match(messages_config, monkeypatch):
    """强制模型输出错误答案 -> 所有目标轮 exact_match 未命中,全进 failures 清单。"""
    cfg = messages_config
    split_dataset(cfg)

    def wrong(self, context, images, sample_id, expected=None):
        return Prediction(id=sample_id, prediction="__definitely_wrong__")

    monkeypatch.setattr(FakeBackend, "complete", wrong)
    run_inference(cfg)
    metrics = score_predictions(cfg)

    assert metrics["num_failures"] == metrics["num_targets"]
    assert cfg.failures_path.exists()

    rows = [json.loads(l) for l in
            cfg.failures_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(rows) == metrics["num_targets"]
    for r in rows:
        assert r["detail"]["exact_match"] == 0.0
        assert r["images"]            # 带原图地址,供人工核查
        assert r["prediction"] == "__definitely_wrong__"


def test_no_failures_when_all_correct(messages_config):
    """fake 回显标准答案 -> 全部命中 -> failures 清单为空。"""
    cfg = messages_config
    split_dataset(cfg)
    run_inference(cfg)
    metrics = score_predictions(cfg)

    assert metrics["num_failures"] == 0
    rows = [l for l in cfg.failures_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert rows == []
