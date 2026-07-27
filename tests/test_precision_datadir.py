"""precision --datadir:按对话格式解析 pred --datadir 产物,不再误报 100% 一致。

背景:pred --datadir 落盘 LlamaFactory 对话格式(输出在 messages 最后一个 assistant
轮,无顶层 prediction/turn)。默认 load_predictions 会读成空串,使 precision 把两端都
当 "" 而报 100% 一致;precision 用 --datadir(而非 --dataset)入口时走对话格式解析修正之
(CLI 层把它翻译成 compare_precision(datadir_format=True))。
"""
from __future__ import annotations

import json

from eval_vlm.config import Config
from eval_vlm.precision import compare_precision
from eval_vlm.results import store


def _write_datadir_jsonl(path, id_to_text):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps({
            "id": sid,
            "images": [sid],
            "messages": [
                {"role": "user", "content": "<image>描述"},
                {"role": "assistant", "content": text},
            ],
        }, ensure_ascii=False)
        for sid, text in id_to_text.items()
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def test_default_format_reads_datadir_as_empty(tmp_path):
    """默认(run/dataset)格式解析 datadir 产物:无 prediction 字段 -> 空串(这正是 bug 源)。"""
    p = tmp_path / "predictions.jsonl"
    _write_datadir_jsonl(p, {"a.png": "第一车道", "b.png": "第三车道"})
    assert all(pr.prediction == "" for pr in store.load_predictions(p))


def test_datadir_format_reads_assistant_turn(tmp_path):
    """datadir_format=True:从 messages 最后一个 assistant 轮取输出与其轮下标。"""
    p = tmp_path / "predictions.jsonl"
    _write_datadir_jsonl(p, {"a.png": "第一车道", "b.png": "第三车道"})
    preds = {pr.id: (pr.prediction, pr.turn) for pr in store.load_predictions(p, datadir_format=True)}
    assert preds == {"a.png": ("第一车道", 1), "b.png": ("第三车道", 1)}


def test_precision_datadir_no_false_100(tmp_path):
    """precision --datadir:两端内容有别 -> 一致率应真实反映(此处 0.5),不再假 100%。"""
    cfg = Config(run_dir_path=tmp_path)
    _write_datadir_jsonl(tmp_path / "m" / "mnn" / "predictions.jsonl",
                         {"a.png": "第一车道", "b.png": "第二车道"})
    _write_datadir_jsonl(tmp_path / "m" / "hf" / "predictions.jsonl",
                         {"a.png": "第一车道", "b.png": "第三车道"})   # b 不同
    summary = compare_precision(cfg, candidate="m", reference="m", datadir_format=True)
    assert summary["num_compared"] == 2
    assert summary["behavior"]["agreement_rate"] == 0.5     # 仅 a 一致


def test_precision_datadir_off_gives_false_100(tmp_path):
    """不加 datadir_format:两端都读成空串 -> 误报 100%(回归守卫,证明该开关的必要)。"""
    cfg = Config(run_dir_path=tmp_path)
    _write_datadir_jsonl(tmp_path / "m" / "mnn" / "predictions.jsonl",
                         {"a.png": "第一车道", "b.png": "第二车道"})
    _write_datadir_jsonl(tmp_path / "m" / "hf" / "predictions.jsonl",
                         {"a.png": "第一车道", "b.png": "第三车道"})
    summary = compare_precision(cfg, candidate="m", reference="m")   # datadir_format=False
    assert summary["behavior"]["agreement_rate"] == 1.0     # 全空串 -> 假 100%
