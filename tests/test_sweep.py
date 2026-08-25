"""sweep:同一模型对一批数据集依次跑各自的 eval / field-eval。

方法分发(eval.method 字段)与汇总报告是纯逻辑,run_eval_once / run_field_eval_once 用
monkeypatch 替换为 canned 返回,不跑真实推理 / 不联网。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from eval_vlm import sweep, workspace
from eval_vlm.config import load_dataset_config

FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# _read_dataset_names:合并 --dataset 与 --dataset-list,去重保序
# ---------------------------------------------------------------------------
def test_read_dataset_names_comma_and_dedup():
    args = argparse.Namespace(dataset="a,b,a, c", dataset_list=None)
    assert sweep._read_dataset_names(args) == ["a", "b", "c"]


def test_read_dataset_names_list_file(tmp_path):
    lst = tmp_path / "list.txt"
    lst.write_text(
        "# 要跑的数据集\na\n\n  b  # 行内注释\n# 注释行\nc\n",
        encoding="utf-8",
    )
    args = argparse.Namespace(dataset="a", dataset_list=str(lst))
    assert sweep._read_dataset_names(args) == ["a", "b", "c"]   # a 去重,空行/注释跳过


def test_read_dataset_names_empty_raises():
    args = argparse.Namespace(dataset=None, dataset_list=None)
    with pytest.raises(ValueError):
        sweep._read_dataset_names(args)


def test_read_dataset_names_list_missing_raises(tmp_path):
    args = argparse.Namespace(dataset=None, dataset_list=str(tmp_path / "nope.txt"))
    with pytest.raises(FileNotFoundError):
        sweep._read_dataset_names(args)


# ---------------------------------------------------------------------------
# run_sweep:方法分发 + 汇总落盘
# ---------------------------------------------------------------------------
def _init_two_datasets(ws: Path):
    """在 ws 下建两个数据集:dsA 默认 eval,dsB 用 set_dataset_value 改成 field-eval。"""
    a = workspace.init_dataset(
        str(FIXTURES / "llamafactory_demo.json"), ws, name="dsA",
        media_root=str(FIXTURES))
    b = workspace.init_dataset(
        str(FIXTURES / "llamafactory_demo.json"), ws, name="dsB",
        media_root=str(FIXTURES))
    workspace.set_dataset_value(b, "eval.method", "field-eval")
    return a, b


def _canned_eval(folder: Path) -> dict:
    return {"dataset": folder.name, "method": "eval", "model": "fake_model",
            "backend": "fake", "metrics": {"overall_mean_score": 0.8},
            "report": str(folder / "fake_model" / "fake" / "summary.md")}


def _canned_field(folder: Path) -> dict:
    return {"dataset": folder.name, "method": "field-eval", "model": "fake_model",
            "backend": "fake",
            "metrics": {"overall": {"micro_accuracy": 0.5, "macro_accuracy": 0.5,
                                    "exact_match_rate": 0.3}},
            "report": str(folder / "fake_model" / "fake" / "field_summary.md")}


def _ns(**kw) -> argparse.Namespace:
    base = dict(dataset=None, dataset_list=None, method=None, dry_run=False,
                stop_on_error=False, workspace=None)
    base.update(kw)
    return argparse.Namespace(**base)


def test_run_sweep_dispatches_and_writes_summary(tmp_path, monkeypatch):
    ws = tmp_path / "ws"
    a, b = _init_two_datasets(ws)

    calls = {"eval": [], "field": []}
    monkeypatch.setattr(workspace, "load_global_config", lambda: {})

    from eval_vlm import cli
    monkeypatch.setattr(cli, "run_eval_once",
                        lambda folder, args: calls["eval"].append(folder.name) or _canned_eval(folder))
    monkeypatch.setattr(cli, "run_field_eval_once",
                        lambda folder, args: calls["field"].append(folder.name) or _canned_field(folder))

    summary = sweep.run_sweep(_ns(dataset="dsA,dsB", workspace=str(ws)))

    # 按各自 eval.method 分发到对应函数
    assert calls["eval"] == ["dsA"]
    assert calls["field"] == ["dsB"]
    assert summary["num_ok"] == 2 and summary["num_error"] == 0

    # 汇总落盘:目录 = ws/_sweep/fake_model/fake/
    sdir = ws / "_sweep" / "fake_model" / "fake"
    assert (sdir / "summary.json").exists()
    assert (sdir / "summary.md").exists()
    data = json.loads((sdir / "summary.json").read_text(encoding="utf-8"))
    assert [r["dataset"] for r in data["results"]] == ["dsA", "dsB"]
    assert all(r["status"] == "ok" for r in data["results"])
    md = (sdir / "summary.md").read_text(encoding="utf-8")
    assert "dsA" in md and "dsB" in md and "field-eval" in md


def test_run_sweep_error_continues_and_records(tmp_path, monkeypatch):
    ws = tmp_path / "ws"
    a, _ = _init_two_datasets(ws)
    monkeypatch.setattr(workspace, "load_global_config", lambda: {})

    from eval_vlm import cli
    def boom(folder, args):
        raise ValueError("boom")
    monkeypatch.setattr(cli, "run_eval_once", boom)
    monkeypatch.setattr(cli, "run_field_eval_once", lambda folder, args: _canned_field(folder))

    # dsB 是 field-eval、missing 不存在 -> eval 那条失败,其余继续
    summary = sweep.run_sweep(_ns(dataset="dsA,dsB,missing", workspace=str(ws)))

    assert summary["num_ok"] == 1 and summary["num_error"] == 2
    statuses = {r["dataset"]: r["status"] for r in summary["results"]}
    assert statuses == {"dsA": "error", "dsB": "ok", "missing": "error"}
    assert "boom" in summary["results"][0]["error"]


def test_run_sweep_stop_on_error_raises(tmp_path, monkeypatch):
    ws = tmp_path / "ws"
    _init_two_datasets(ws)
    monkeypatch.setattr(workspace, "load_global_config", lambda: {})

    from eval_vlm import cli
    monkeypatch.setattr(cli, "run_eval_once", lambda folder, args: (_ for _ in ()).throw(ValueError("x")))

    with pytest.raises(ValueError):
        sweep.run_sweep(_ns(dataset="dsA,dsB", workspace=str(ws), stop_on_error=True))


def test_run_sweep_dry_run_no_artifacts(tmp_path, monkeypatch):
    ws = tmp_path / "ws"
    _init_two_datasets(ws)
    monkeypatch.setattr(workspace, "load_global_config", lambda: {})

    from eval_vlm import cli
    monkeypatch.setattr(cli, "run_eval_once", lambda folder, args: _canned_eval(folder))
    monkeypatch.setattr(cli, "run_field_eval_once", lambda folder, args: _canned_field(folder))

    summary = sweep.run_sweep(_ns(dataset="dsA,dsB", workspace=str(ws), dry_run=True))
    assert summary["dry_run"] is True and summary["results"] == []
    assert not (ws / "_sweep").exists()          # dry-run 不落任何产物


def test_run_sweep_method_override_persists(tmp_path, monkeypatch):
    """--method 批量覆盖并永久写回每个数据集的 eval.method。"""
    ws = tmp_path / "ws"
    a, b = _init_two_datasets(ws)
    monkeypatch.setattr(workspace, "load_global_config", lambda: {})

    from eval_vlm import cli
    monkeypatch.setattr(cli, "run_eval_once", lambda folder, args: _canned_eval(folder))
    monkeypatch.setattr(cli, "run_field_eval_once", lambda folder, args: _canned_field(folder))

    sweep.run_sweep(_ns(dataset="dsA,dsB", workspace=str(ws), method="field-eval"))

    # 两个数据集都被写回 field-eval
    assert load_dataset_config(a).eval.method == "field-eval"
    assert load_dataset_config(b).eval.method == "field-eval"


def test_parser_sweep_flags():
    from eval_vlm.cli import build_parser, _cmd_sweep
    parser = build_parser()
    args = parser.parse_args(["sweep", "-d", "a,b", "--dataset-list", "lst.txt",
                              "--method", "field-eval", "--dry-run", "--backend", "fake"])
    assert args.func is _cmd_sweep
    assert args.dataset == "a,b" and args.dataset_list == "lst.txt"
    assert args.method == "field-eval" and args.dry_run is True
    assert args.backend == "fake"
