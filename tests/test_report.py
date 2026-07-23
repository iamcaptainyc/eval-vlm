"""report(跨格式合并报告)+ discover_run_dirs + quality_crosstab 单元测试。

全部不跑模型:手写 metrics.json / scored.jsonl / precision.json / run_meta.json,
验证发现、交叉、渲染与诊断。
"""
from __future__ import annotations

from pathlib import Path

from eval_vlm.compare import load_scored, quality_crosstab
from eval_vlm.config import Config
from eval_vlm.report import build_report, render_report_md
from eval_vlm.results import store


# ---------------------------------------------------------------------------
# discover_run_dirs
# ---------------------------------------------------------------------------
def test_discover_run_dirs_finds_and_skips(tmp_path):
    # 顶层数据集级文件(应被跳过,因不是二级目录)
    (tmp_path / "config.yaml").write_text("x", encoding="utf-8")
    (tmp_path / "test.json").write_text("[]", encoding="utf-8")
    # 有效产物目录:<model>/<backend>/metrics.json
    store.write_json(tmp_path / "hf-ckpt" / "hf" / "metrics.json", {"overall_mean_score": 1.0})
    store.write_json(tmp_path / "mnn-4bit" / "mnn" / "precision.json", {"flags": []})
    # 无标志文件的后端目录(应被跳过)
    (tmp_path / "mnn-4bit" / "empty").mkdir(parents=True)

    found = store.discover_run_dirs(tmp_path)
    triples = {(m, b) for m, b, _ in found}
    assert triples == {("hf-ckpt", "hf"), ("mnn-4bit", "mnn")}


def test_discover_run_dirs_empty_when_missing(tmp_path):
    assert store.discover_run_dirs(tmp_path / "nope") == []


# ---------------------------------------------------------------------------
# quality_crosstab
# ---------------------------------------------------------------------------
def _row(scorer, score):
    return {"scorer": scorer, "score": score}


def test_quality_crosstab_binary_counts():
    cand = {("A", 0): _row("exact_match", 1.0), ("B", 0): _row("exact_match", 0.0)}
    ref = {("A", 0): _row("exact_match", 1.0), ("B", 0): _row("exact_match", 1.0)}
    ct = quality_crosstab(cand, ref)
    b = ct["binary"]
    assert ct["num_common"] == 2
    assert b["num"] == 2
    assert b["both_correct"] == 1
    assert b["cand_wrong_ref_correct"] == 1        # B:HF对 候选错 -> 净回归
    assert b["net_regression_rate"] == 0.5
    assert ct["continuous"]["num"] == 0


def test_quality_crosstab_continuous_mean_delta():
    cand = {("A", 0): _row("token_f1", 0.6), ("B", 0): _row("token_f1", 0.4)}
    ref = {("A", 0): _row("token_f1", 0.9), ("B", 0): _row("token_f1", 0.8)}
    ct = quality_crosstab(cand, ref)
    assert ct["binary"]["num"] == 0
    c = ct["continuous"]
    assert c["num"] == 2
    assert c["mean_score_candidate"] == 0.5
    assert c["mean_score_reference"] == 0.85
    assert c["mean_score_delta"] == -0.35          # 候选更差


def test_quality_crosstab_mixed_turns_split():
    """轮0二值、轮1连续:各归各类。"""
    cand = {("A", 0): _row("prefix_match", 1.0), ("A", 1): _row("token_f1", 0.5)}
    ref = {("A", 0): _row("prefix_match", 1.0), ("A", 1): _row("token_f1", 0.9)}
    ct = quality_crosstab(cand, ref)
    assert ct["binary"]["num"] == 1 and ct["binary"]["both_correct"] == 1
    assert ct["continuous"]["num"] == 1
    assert set(ct["per_turn"]) == {0, 1}


def test_load_scored_roundtrip(tmp_path):
    path = tmp_path / "scored.jsonl"
    store.write_jsonl(path, [{"id": "A", "turn": 0, "scorer": "exact_match", "score": 1.0}])
    d = load_scored(path)
    assert d[("A", 0)]["score"] == 1.0
    assert load_scored(tmp_path / "missing.jsonl") == {}


# ---------------------------------------------------------------------------
# build_report / render — 造一个 hf + 两个 mnn 变体的数据集目录
# ---------------------------------------------------------------------------
def _cfg(tmp_path) -> Config:
    cfg = Config()
    cfg.run_dir_path = tmp_path           # dataset_dir 钉到 tmp
    return cfg


def _seed_format(root: Path, model: str, backend: str, *, overall, per_turn,
                 scored, quant=None, precision=None):
    d = root / model / backend
    store.write_json(d / "metrics.json",
                     {"overall_mean_score": overall, "num_samples": len(scored),
                      "per_turn": per_turn})
    store.write_jsonl(d / "scored.jsonl", scored)
    if quant is not None:
        store.write_json(d / "run_meta.json", {"backend": backend, "quant": quant})
    if precision is not None:
        store.write_json(d / "precision.json", precision)


def _seed_dataset(tmp_path):
    # HF 基准:全对
    _seed_format(tmp_path, "hf-ckpt", "hf", overall=1.0,
                 per_turn={"turn_0": {"scorer": "exact_match", "accuracy": 1.0}},
                 scored=[{"id": "A", "turn": 0, "scorer": "exact_match", "score": 1.0},
                         {"id": "B", "turn": 0, "scorer": "exact_match", "score": 1.0}])
    # MNN 4bit:掉一半 + precision 报预处理不对齐
    _seed_format(tmp_path, "mnn-4bit", "mnn", overall=0.5,
                 per_turn={"turn_0": {"scorer": "exact_match", "accuracy": 0.5}},
                 scored=[{"id": "A", "turn": 0, "scorer": "exact_match", "score": 1.0},
                         {"id": "B", "turn": 0, "scorer": "exact_match", "score": 0.0}],
                 quant="hqq-4bit",
                 precision={"behavior": {"agreement_rate": 0.5, "mean_token_f1": 0.6},
                            "flags": ["🔴 prompt token 数不对齐:预处理疑不一致。"]})
    # MNN 8bit:略掉分,无 precision
    _seed_format(tmp_path, "mnn-8bit", "mnn", overall=0.9,
                 per_turn={"turn_0": {"scorer": "exact_match", "accuracy": 0.9}},
                 scored=[{"id": "A", "turn": 0, "scorer": "exact_match", "score": 1.0},
                         {"id": "B", "turn": 0, "scorer": "exact_match", "score": 1.0}],
                 quant="hqq-8bit")


def test_build_report_structure_and_diagnosis(tmp_path):
    cfg = _cfg(tmp_path)
    _seed_dataset(tmp_path)
    report = build_report(cfg)

    assert report["num_formats"] == 3
    assert report["baseline"] == {"model": "hf-ckpt", "backend": "hf"}

    labels = {f["label"] for f in report["formats"]}
    assert "mnn-4bit/mnn [hqq-4bit]" in labels
    assert "mnn-8bit/mnn [hqq-8bit]" in labels

    # 净质量Δ:4bit 对 hf 有 1/2 净回归;8bit 无回归
    nq = {n["label"]: n for n in report["net_quality"]}
    assert nq["mnn-4bit/mnn [hqq-4bit]"]["binary"]["net_regression_rate"] == 0.5
    assert nq["mnn-8bit/mnn [hqq-8bit]"]["binary"]["net_regression_rate"] == 0.0

    diag = " ".join(report["diagnosis"])
    assert "🔴" in diag and "预处理" in diag          # 4bit:掉分 + 预处理不对齐
    assert "🟠" in diag and "未跑 precision" in diag   # 8bit:掉分 + 无 precision


def test_render_report_md_writes_table(tmp_path):
    cfg = _cfg(tmp_path)
    _seed_dataset(tmp_path)
    report = build_report(cfg)
    md = render_report_md(report)

    assert "质量门禁合并报告" in md
    assert "绝对质量并排" in md and "净质量Δ" in md and "诊断结论" in md
    assert "hqq-4bit" in md and "hqq-8bit" in md
    assert "Δ vs HF" in md


def test_report_cli_end_to_end(tmp_path, monkeypatch):
    """真实工作目录:split -> pred(fake) -> score 产出一个格式,再 `report` 落盘合并报告。"""
    import argparse

    from eval_vlm import workspace
    from eval_vlm.cli import _cmd_report
    from eval_vlm.config import load_dataset_config
    from eval_vlm.data.splitter import split_dataset
    from eval_vlm.evaluate import score_predictions
    from eval_vlm.runner import run_inference

    fixtures = Path(__file__).parent / "fixtures"
    monkeypatch.setenv("EVAL_VLM_CONFIG", str(tmp_path / "g.yaml"))
    ws = tmp_path / "ws"
    folder = workspace.init_dataset(
        str(fixtures / "llamafactory_demo.json"), ws,
        media_root=str(fixtures), split_overrides={"train": 0.6, "test": 0.4})
    workspace.set_dataset_value(folder, "inference.backend", "fake")

    cfg = load_dataset_config(folder)
    split_dataset(cfg)
    run_inference(cfg)
    score_predictions(cfg)                      # 产出 metrics.json + scored.jsonl

    ns = argparse.Namespace(dataset="llamafactory_demo", workspace=str(ws))
    assert _cmd_report(ns) == 0
    assert (folder / "report.md").exists() and (folder / "report.json").exists()
    md = (folder / "report.md").read_text(encoding="utf-8")
    assert "质量门禁合并报告" in md
    assert "trained-vlm/fake" in md             # fake 后端模型名取 openai.model


def test_build_report_no_hf_baseline(tmp_path):
    """无 hf 产物 -> 诊断给出无基准提示,不崩。"""
    cfg = _cfg(tmp_path)
    _seed_format(tmp_path, "mnn-4bit", "mnn", overall=0.7,
                 per_turn={"turn_0": {"scorer": "exact_match", "accuracy": 0.7}},
                 scored=[{"id": "A", "turn": 0, "scorer": "exact_match", "score": 1.0}],
                 quant="hqq-4bit")
    report = build_report(cfg)
    assert report["baseline"] is None
    assert any("无 HF 基准" in d or "未发现 HF 基准" in d for d in report["diagnosis"])
    # 无基准时净质量Δ为空,不报错
    assert report["net_quality"] == []
    render_report_md(report)   # 不崩即可
