"""命令行参数层(工作目录模型)+ split 自定义输出位置。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from eval_vlm import workspace
from eval_vlm.cli import build_parser, _cmd_split, _cmd_pred, _cmd_score, _cmd_eval
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


def test_parser_train_out_flag_vs_path():
    """--train-out 支持两种形态:光杆旗标(const="")与显式路径。"""
    parser = build_parser()
    bare = parser.parse_args(["split", "-d", "x.json", "--train-out"])
    assert bare.train_out == ""                              # 光杆旗标 -> const
    withpath = parser.parse_args(["split", "-d", "x.json", "--train-out", "/p/t.json"])
    assert withpath.train_out == "/p/t.json"
    none = parser.parse_args(["split", "-d", "x.json"])
    assert none.train_out is None                            # 未提供


def test_resolve_split_out_semantics(tmp_path):
    """resolve_split_out:None->None;显式路径原样;光杆旗标->全局目录/<名>_<份>.json。"""
    assert workspace.resolve_split_out(None, "train", "emo", {}) is None
    assert workspace.resolve_split_out("/a/b.json", "train", "emo", {}) == "/a/b.json"
    cfg = {"train_out_dir": str(tmp_path / "lf")}
    got = workspace.resolve_split_out("", "train", "emo", cfg)
    assert Path(got) == tmp_path / "lf" / "emo_train.json"
    # 光杆旗标但未设全局目录 -> 报错(main() 会转成 [error])
    with pytest.raises(ValueError):
        workspace.resolve_split_out("", "train", "emo", {})


def test_split_train_out_flag_autonames(tmp_path, monkeypatch):
    """光杆 --train-out:train 产物落到全局 train_out_dir/<数据集名>_train.json。"""
    monkeypatch.setenv("EVAL_VLM_CONFIG", str(tmp_path / "g.yaml"))
    ws = tmp_path / "ws"
    lf = tmp_path / "lf_data"
    workspace.set_global_value("workspace", str(ws))
    workspace.set_global_value("train_out_dir", str(lf))

    ns = argparse.Namespace(
        dataset=str(FIXTURES / "llamafactory_demo.json"), name=None,
        train=0.6, test=0.4, val=None, seed=None, stratify_by=None,
        train_out="", val_out=None, test_out=None,          # 光杆 --train-out
        force=False, workspace=str(ws),
    )
    assert _cmd_split(ns) == 0
    out = lf / "llamafactory_demo_train.json"
    assert out.exists()                                      # 自动命名 + 落到全局目录
    assert json.loads(out.read_text(encoding="utf-8"))      # 非空
    # 默认数据集文件夹内不再产出 train.json(已重定向)
    assert not (ws / "llamafactory_demo" / "train.json").exists()


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


def test_parser_eval_backend_flags():
    """eval 也接受后端/权重 flag(与 pred 一致),让每个格式一条 eval 跑完 pred+score。"""
    from eval_vlm.cli import _PERSIST_MAP
    parser = build_parser()
    args = parser.parse_args([
        "eval", "--dataset", "emo_v4", "--backend", "mnn",
        "--mnn-config", "/mnn/emo-4bit/config.json", "--mnn-quant", "hqq-4bit",
        "--mnn-image-max-side", "1536",
    ])
    assert args.func is _cmd_eval
    assert args.backend == "mnn" and args.mnn_config == "/mnn/emo-4bit/config.json"
    assert args.mnn_quant == "hqq-4bit" and args.mnn_image_max_side == 1536
    hf = parser.parse_args(["eval", "-d", "x", "--backend", "hf", "--hf-model", "/ckpt/h"])
    assert hf.backend == "hf" and hf.hf_model == "/ckpt/h"
    # 这些 flag 都在持久化映射里,_persist_overrides 会自动写回 config.yaml
    persist_attrs = {a for a, _ in _PERSIST_MAP}
    assert {"backend", "hf_model", "mnn_config", "mnn_quant", "mnn_image_max_side"} <= persist_attrs


def test_parser_pred_and_score_require_target():
    parser = build_parser()
    # pred 取代旧 run:--dataset 走数据集预测
    assert parser.parse_args(["pred", "--dataset", "x"]).func is _cmd_pred
    assert parser.parse_args(["score", "--dataset", "x"]).func is _cmd_score


def test_pred_requires_exactly_one_target():
    """pred 的 --dataset / --datadir 互斥且必填其一。"""
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["pred"])                                   # 都不给
    with pytest.raises(SystemExit):
        parser.parse_args(["pred", "--dataset", "x", "--datadir", "y"])  # 同时给


def test_config_flag_removed():
    """旧的 --config 已移除:传入应报错退出。"""
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["pred", "--dataset", "x", "--config", "x.yaml"])


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
    assert (folder / "cli_model" / "fake" / "predictions.jsonl").exists()
    assert (folder / "cli_model" / "fake" / "metrics.json").exists()
    # 重新加载确认持久化生效
    assert load_dataset_config(folder).inference.openai.model == "cli_model"
