"""命令行参数层(工作目录模型)+ split 自定义输出位置。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from eval_vlm import workspace
from eval_vlm.cli import build_parser, _cmd_split, _cmd_run, _cmd_score, _cmd_eval
from eval_vlm.config import load_dataset_config
from eval_vlm.data.splitter import split_dataset

FIXTURES = Path(__file__).parent / "fixtures"


def test_split_custom_output_paths(messages_config, tmp_path):
    """train_out/test_out 把产物重定向到任意(可不存在的)目录,父目录自动创建。"""
    cfg = messages_config
    dest = tmp_path / "lf_data" / "emotion_train.json"      # 嵌套目录尚不存在
    cfg.split.train_out = str(dest)
    cfg.split.test_out = str(tmp_path / "held_out" / "test.json")

    meta = split_dataset(cfg)

    assert dest.exists()                                     # 父目录被自动创建
    assert cfg.train_path == dest
    assert Path(meta["files"]["train"]) == dest
    # 已重定向 -> 默认 run_dir 下不再产出 train.json
    assert not (cfg.run_dir / "train.json").exists()
    train = json.loads(dest.read_text(encoding="utf-8"))
    assert train and "messages" in train[0]


def test_parser_split_ratios():
    """split: --train/--test 设置比例,路由到 _cmd_split。"""
    parser = build_parser()
    args = parser.parse_args([
        "split", "--dataset", "/data/emo_v4.json",
        "--train", "0.8", "--test", "0.2", "--seed", "7", "--name", "emo",
    ])
    assert args.func is _cmd_split
    assert args.dataset == "/data/emo_v4.json"
    assert args.train == 0.8 and args.test == 0.2
    assert args.seed == 7 and args.name == "emo"


def test_parser_eval_routes_and_overrides():
    """eval = run+score;接受 --base-url/--model/--scorer 临时覆盖。"""
    parser = build_parser()
    args = parser.parse_args([
        "eval", "--dataset", "emo_v4",
        "--base-url", "http://h:9/v1", "--model", "m", "--scorer", "token_f1",
    ])
    assert args.func is _cmd_eval
    assert args.dataset == "emo_v4"
    assert args.base_url == "http://h:9/v1" and args.model == "m"
    assert args.scorer == "token_f1"


def test_parser_run_and_score_require_dataset():
    parser = build_parser()
    assert parser.parse_args(["run", "--dataset", "x"]).func is _cmd_run
    assert parser.parse_args(["score", "--dataset", "x"]).func is _cmd_score


def test_config_flag_removed():
    """旧的 --config 已移除:传入应报错退出。"""
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["run", "--config", "x.yaml"])


def test_eval_cli_persists_model_and_uses_model_dir(tmp_path, monkeypatch):
    """eval --model 永久写回 config.yaml,且产物落在 数据集/<模型>/ 目录(用户参数优先)。"""
    monkeypatch.setenv("EVAL_VLM_CONFIG", str(tmp_path / "g.yaml"))
    ws = tmp_path / "ws"
    folder = workspace.init_dataset(
        str(FIXTURES / "llamafactory_demo.json"), ws,
        media_root=str(FIXTURES), split_overrides={"train": 0.6, "test": 0.4},
    )
    workspace.set_dataset_value(folder, "inference.backend", "fake")   # 离线回显
    split_dataset(load_dataset_config(folder))                         # 先产出 test.json

    ns = argparse.Namespace(dataset="llamafactory_demo", workspace=str(ws),
                            base_url=None, model="cli_model", scorer=None)
    assert _cmd_eval(ns) == 0

    # --model 写回 config.yaml(永久),产物落到该模型子目录
    assert "cli_model" in (folder / "config.yaml").read_text(encoding="utf-8")
    assert (folder / "cli_model" / "predictions.jsonl").exists()
    assert (folder / "cli_model" / "metrics.json").exists()
    # 重新加载确认持久化生效
    assert load_dataset_config(folder).inference.model == "cli_model"
