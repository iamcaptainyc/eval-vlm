"""验证当模型预测已存在时，跳过后端加载（避免昂贵后端如 vllm_offline 无谓初始化）。"""
from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from eval_vlm import cli, workspace
from eval_vlm.config import Config, InferenceConfig, load_dataset_config
from eval_vlm.data.splitter import split_dataset
from eval_vlm.predict import predict_folder
from eval_vlm.runner import run_inference

FIXTURES = Path(__file__).parent / "fixtures"


def test_run_inference_skips_build_backend_when_all_done(messages_config, monkeypatch):
    """当所有测试样本的目标轮均已存在于 predictions.jsonl 中时，run_inference 不得调用 build_backend。"""
    cfg = messages_config
    split_dataset(cfg)

    # 首次正常跑生成 predictions.jsonl
    stats1 = run_inference(cfg)
    assert stats1["newly_completed"] > 0
    assert cfg.predictions_path.exists()

    # 监控/拦截 build_backend：如果再次被调用则报错
    def fail_on_build(config):
        raise RuntimeError("不应该启动后端！")

    monkeypatch.setattr("eval_vlm.runner.build_backend", fail_on_build)

    # 二次运行：所有样本都已完成，应跳过 build_backend
    stats2 = run_inference(cfg)
    assert stats2["newly_completed"] == 0
    assert stats2["skipped_samples_already_done"] == stats1["test_size"]


def test_run_inference_calls_build_backend_when_incomplete(messages_config, monkeypatch):
    """当有样本未完成时，run_inference 必须正常调用 build_backend 补齐缺失。"""
    cfg = messages_config
    split_dataset(cfg)
    run_inference(cfg)

    # 模拟中断：删去最后一行预测
    lines = cfg.predictions_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) >= 2
    cfg.predictions_path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")

    build_called = []
    from eval_vlm.inference.fake_backend import FakeBackend

    def spy_build(config):
        build_called.append(True)
        return FakeBackend(config)

    monkeypatch.setattr("eval_vlm.runner.build_backend", spy_build)

    stats = run_inference(cfg)
    assert len(build_called) == 1
    assert stats["newly_completed"] == 1


def test_predict_folder_skips_build_backend_when_all_done(tmp_path, monkeypatch):
    """predict_folder：全部图片已完成时，不得调用 build_backend。"""
    imgs = tmp_path / "images"
    imgs.mkdir()
    (imgs / "1.jpg").write_bytes(b"")
    (imgs / "2.png").write_bytes(b"")

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    cfg = Config(
        run_name="test_pred",
        output_dir=str(tmp_path),
        inference=InferenceConfig(backend="fake"),
        run_dir_path=out_dir,
    )
    cfg.data.media_root = str(imgs)

    # 首次运行正常完成
    stats1 = predict_folder(cfg, imgs)
    assert stats1["newly_completed"] == 2

    # 二次运行：拦截 build_backend
    def fail_on_build(config):
        raise RuntimeError("不应该启动后端！")

    monkeypatch.setattr("eval_vlm.predict.build_backend", fail_on_build)

    stats2 = predict_folder(cfg, imgs)
    assert stats2["newly_completed"] == 0
    assert stats2["skipped_already_done"] == 2


def test_run_eval_once_reuses_existing_predictions(tmp_path, monkeypatch):
    """eval 命令：当预测完整存在时，run_eval_once 直接复用预测并评分，不调 _do_run 也不起后端。"""
    ws = tmp_path / "ws"
    ws.mkdir()
    folder = workspace.init_dataset(
        str(FIXTURES / "llamafactory_demo.json"),
        ws,
        name="ds",
        media_root=str(FIXTURES),
        split_overrides={"train": 0.0, "test": 1.0},
    )
    workspace.set_dataset_value(folder, "inference.backend", "fake")
    cfg = load_dataset_config(folder)
    split_dataset(cfg)
    run_inference(cfg)
    assert cfg.predictions_path.exists()

    monkeypatch.setattr(workspace, "load_global_config", lambda: {})

    def fail_on_run(config, *args, **kwargs):
        raise RuntimeError("_do_run 不应被调用！")

    monkeypatch.setattr(cli, "_do_run", fail_on_run)

    args = argparse.Namespace(
        dataset="ds",
        workspace=str(ws),
        scorer=None,
        backend="fake",
        report_html=False,
    )
    result = cli.run_eval_once(folder, args)
    assert result["method"] == "eval"
    assert "metrics" in result
    assert result["metrics"]["overall_mean_score"] == 1.0


def test_run_field_eval_once_reuses_existing_predictions(tmp_path, monkeypatch):
    """field-eval 命令：预测完整存在时，直接复用预测，不调 _do_run 补跑。"""
    ws = tmp_path / "ws"
    ws.mkdir()
    folder = workspace.init_dataset(
        str(FIXTURES / "llamafactory_demo.json"),
        ws,
        name="ds",
        media_root=str(FIXTURES),
        split_overrides={"train": 0.0, "test": 1.0},
    )
    workspace.set_dataset_value(folder, "inference.backend", "fake")
    cfg = load_dataset_config(folder)
    split_dataset(cfg)
    run_inference(cfg)
    assert cfg.predictions_path.exists()

    monkeypatch.setattr(workspace, "load_global_config", lambda: {})

    def fail_on_run(config, *args, **kwargs):
        raise RuntimeError("_do_run 不应被调用！")

    monkeypatch.setattr(cli, "_do_run", fail_on_run)

    # mock run_field_eval 以免真正调远程 value-extract 服务
    monkeypatch.setattr(
        cli,
        "run_field_eval",
        lambda cfg, **kw: {
            "overall": {"micro_accuracy": 1.0, "macro_accuracy": 1.0, "exact_match_rate": 1.0},
            "num_scored": 1,
            "num_pred_missing": 0,
            "skipped_ref": 0,
            "skipped_pred_error": 0,
            "fields": [],
            "per_field": {},
            "confusion_matrices": {},
        },
    )

    args = argparse.Namespace(
        dataset="ds",
        workspace=str(ws),
        backend="fake",
        report_html=False,
    )
    res = cli.run_field_eval_once(folder, args)
    assert res["method"] == "field-eval"
