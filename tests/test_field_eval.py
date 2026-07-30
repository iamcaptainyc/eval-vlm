"""field-eval:第一轮描述 -> value-extract 服务抽固定字段 -> 逐字段严格相等准确率。

抽取 HTTP 全程 monkeypatch(不联网):替换 field_eval.extract_fields_one / load_samples。
比对与聚合是纯逻辑,直接喂合成字段字典断言。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval_vlm import field_eval
from eval_vlm.cli import build_parser, _cmd_field_eval
from eval_vlm.config import Config, LabelExtractConfig
from eval_vlm.data.schema import Sample, Turn


# ---------------------------------------------------------------------------
# parse_cn_fields:保留字段结构、保留「无警示牌」、丢弃 none_label「无」
# ---------------------------------------------------------------------------
def _data(cn: dict) -> dict:
    return {"labels": {"cn": cn}}


def test_parse_cn_fields_keeps_structure_and_drops_none():
    out = field_eval.parse_cn_fields(_data({
        "主辅路": ["主路"],
        "道路结构": ["无"],          # none_label -> 空
        "车道位置": [],              # 本就空
        "警示标志": ["学校路口", "礼让行人"],
    }))
    assert out == {
        "主辅路": ["主路"],
        "道路结构": [],
        "车道位置": [],
        "警示标志": ["学校路口", "礼让行人"],   # 排序后
    }


def test_parse_cn_fields_keeps_no_warning_sign():
    """「无警示牌」是合法枚举值(≠ none_label「无」),必须保留。"""
    out = field_eval.parse_cn_fields(_data({"警示标志": ["无警示牌"]}))
    assert out == {"警示标志": ["无警示牌"]}


def test_parse_cn_fields_dedupes():
    out = field_eval.parse_cn_fields(_data({"警示标志": ["礼让行人", "礼让行人", "学校路口"]}))
    assert out == {"警示标志": ["学校路口", "礼让行人"]}


def test_parse_cn_fields_missing_labels():
    assert field_eval.parse_cn_fields({}) == {}
    assert field_eval.parse_cn_fields({"labels": {}}) == {}


def test_parse_cn_fields_custom_none_label():
    out = field_eval.parse_cn_fields(_data({"x": ["none"], "y": ["主路"]}), none_label="none")
    assert out == {"x": [], "y": ["主路"]}


# ---------------------------------------------------------------------------
# _aggregate:逐字段严格相等 + 跳过/缺失规则(纯逻辑)
# ---------------------------------------------------------------------------
def test_aggregate_correctness_matrix():
    """覆盖:全对、单字段错、pred 无输出(全错)、ref 抽取失败(跳过)、pred 抽取失败(跳过)。"""
    samples = [Sample(id=x) for x in ("a", "b", "c", "d", "e")]
    # d 不在 desc_turn -> ref 无描述 -> 跳过;其余都有 ref 描述轮
    desc_turn = {"a": 1, "b": 1, "c": 1, "e": 1}
    ref_fields = {
        "a": {"主辅路": ["主路"], "警示标志": ["礼让行人"]},
        "b": {"主辅路": ["主路"], "警示标志": []},
        "c": {"主辅路": ["主路"], "警示标志": ["学校路口"]},
        "e": {"主辅路": ["主路"], "警示标志": []},
    }
    pred_fields = {
        "a": {"主辅路": ["主路"], "警示标志": ["礼让行人"]},   # 全对
        "b": {"主辅路": ["主路"], "警示标志": ["学校路口"]},   # 警示标志 错
        # e 有描述文本但抽取失败 -> 不在 pred_fields
    }
    # c 无描述文本(模型没产出);e 有文本但抽取失败
    pred_desc_ids = {"a", "b", "e"}

    metrics, rows = field_eval._aggregate(samples, desc_turn, ref_fields, pred_fields, pred_desc_ids)

    assert metrics["num_samples"] == 5
    assert metrics["skipped_ref"] == 1          # d
    assert metrics["skipped_pred_error"] == 1   # e
    assert metrics["num_scored"] == 3           # a, b, c
    assert metrics["num_pred_missing"] == 1     # c
    assert metrics["fields"] == ["主辅路", "警示标志"]

    pf = metrics["per_field"]
    assert pf["主辅路"] == {"correct": 2, "total": 3, "accuracy": 0.6667}   # a✓ b✓ c✗
    assert pf["警示标志"] == {"correct": 1, "total": 3, "accuracy": 0.3333}  # a✓ b✗ c✗

    ov = metrics["overall"]
    assert ov["micro_accuracy"] == 0.5          # (2+1)/(3+3)
    assert ov["macro_accuracy"] == 0.5          # (0.6667+0.3333)/2
    assert ov["exact_match_samples"] == 1       # 仅 a 全对
    assert ov["exact_match_rate"] == 0.3333

    # 失配清单:仅含 b、c(a 全对不列),c 标 pred_missing
    ids = {r["id"]: r["state"] for r in rows}
    assert ids == {"b": "compared", "c": "pred_missing"}

    # 逐取值(per-class)准确率:该取值在 ref 出现的样本中 pred 命中的比例
    pv = metrics["per_value"]
    # 主辅路 ref 全是「主路」(a,b,c),pred 命中 a、b,c 无输出 -> 2/3
    assert pv["主辅路"]["主路"] == {"correct": 2, "support": 3, "accuracy": 0.6667}
    assert "辅路" not in pv["主辅路"]                      # ref 中未出现 -> 不统计
    # 警示标志:礼让行人(a,命中)1/1;学校路口(c,pred_missing 未命中)0/1
    assert pv["警示标志"]["礼让行人"] == {"correct": 1, "support": 1, "accuracy": 1.0}
    assert pv["警示标志"]["学校路口"] == {"correct": 0, "support": 1, "accuracy": 0.0}


def test_aggregate_both_empty_is_correct():
    """ref 与 pred 该字段都为空(都「无」)-> 判对。"""
    samples = [Sample(id="a")]
    metrics, _ = field_eval._aggregate(
        samples, {"a": 1},
        ref_fields={"a": {"主辅路": []}},
        pred_fields={"a": {"主辅路": []}},
        pred_desc_ids={"a"},
    )
    assert metrics["per_field"]["主辅路"]["accuracy"] == 1.0
    assert metrics["overall"]["exact_match_rate"] == 1.0


def test_aggregate_multivalue_set_equality():
    """多值字段按集合相等(顺序无关)判对。"""
    samples = [Sample(id="a")]
    metrics, _ = field_eval._aggregate(
        samples, {"a": 1},
        ref_fields={"a": {"警示标志": ["礼让行人", "学校路口"]}},
        pred_fields={"a": {"警示标志": ["学校路口", "礼让行人"]}},
        pred_desc_ids={"a"},
    )
    assert metrics["per_field"]["警示标志"]["accuracy"] == 1.0


def test_aggregate_pred_missing_field_is_wrong():
    """pred 缺某字段(空集)而 ref 非空 -> 该字段错。"""
    samples = [Sample(id="a")]
    metrics, _ = field_eval._aggregate(
        samples, {"a": 1},
        ref_fields={"a": {"主辅路": ["主路"]}},
        pred_fields={"a": {}},               # 该字段缺 -> 空集
        pred_desc_ids={"a"},
    )
    assert metrics["per_field"]["主辅路"]["accuracy"] == 0.0


def test_canonical_fields_union_and_fallback():
    assert field_eval._canonical_fields({"a": {"x": [], "y": []}, "b": {"z": []}}) == ["x", "y", "z"]
    assert field_eval._canonical_fields({}) == ["主辅路", "道路结构", "车道位置", "警示标志"]


# ---------------------------------------------------------------------------
# run_field_eval:端到端(monkeypatch load_samples + extract_fields_one)
# ---------------------------------------------------------------------------
def _cfg(tmp_path: Path) -> Config:
    cfg = Config()
    cfg.run_dir_path = tmp_path                       # dataset_dir & run_dir 都锚到 tmp_path 下
    cfg.label_extract = LabelExtractConfig(max_concurrency=1)   # 串行,断言确定
    return cfg


def _write_predictions(cfg: Config, rows: list[tuple[str, str]]) -> None:
    """写 predictions.jsonl(对话格式:id + 末 assistant=描述)。"""
    cfg.predictions_path.parent.mkdir(parents=True, exist_ok=True)
    with cfg.predictions_path.open("w", encoding="utf-8") as f:
        for sid, desc in rows:
            rec = {"id": sid, "images": [f"{sid}.jpg"],
                   "messages": [{"role": "user", "content": "<image>描述"},
                                {"role": "assistant", "content": desc}]}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# 抽取器:按描述文本查表返回字段字典(不联网)
_EXTRACT = {
    "ref_a": {"主辅路": ["主路"]}, "pred_a": {"主辅路": ["主路"]},   # 全对
    "ref_b": {"主辅路": ["主路"]}, "pred_b": {"主辅路": ["辅路"]},   # 错
}


def _install_fakes(cfg: Config, monkeypatch, calls: dict) -> None:
    samples = [
        Sample(id="a", turns=[Turn("user", "q"), Turn("assistant", "ref_a")]),
        Sample(id="b", turns=[Turn("user", "q"), Turn("assistant", "ref_b")]),
    ]
    monkeypatch.setattr(field_eval, "load_samples", lambda cfg, source=None: samples)

    def fake_extract(text, le):
        calls[text] = calls.get(text, 0) + 1
        return dict(_EXTRACT[text])

    monkeypatch.setattr(field_eval, "extract_fields_one", fake_extract)
    cfg.test_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.test_path.write_text("[]", encoding="utf-8")           # 仅为通过存在性检查
    _write_predictions(cfg, [("a", "pred_a"), ("b", "pred_b")])


def test_run_field_eval_end_to_end(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    calls: dict = {}
    _install_fakes(cfg, monkeypatch, calls)

    metrics = field_eval.run_field_eval(cfg)

    assert metrics["num_scored"] == 2
    assert metrics["per_field"]["主辅路"] == {"correct": 1, "total": 2, "accuracy": 0.5}
    assert metrics["overall"]["exact_match_rate"] == 0.5

    # 落盘产物齐全
    assert cfg.field_ref_path.exists() and cfg.field_pred_path.exists()
    assert cfg.field_metrics_path.exists() and cfg.field_summary_path.exists()
    assert cfg.field_mismatches_path.exists()
    # ref 缓存在数据集级、pred 在运行级
    assert cfg.field_ref_path == tmp_path / "fields_ref.jsonl"
    assert cfg.field_pred_path.parent == cfg.run_dir
    # 失配清单含 b,不含全对的 a
    mm = cfg.field_mismatches_path.read_text(encoding="utf-8")
    assert "样本 `b`" in mm and "样本 `a`" not in mm


def test_run_field_eval_resume_skips_done(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    calls: dict = {}
    _install_fakes(cfg, monkeypatch, calls)

    field_eval.run_field_eval(cfg)
    n1 = sum(calls.values())
    assert n1 == 4                        # 2 ref + 2 pred
    field_eval.run_field_eval(cfg)        # 再跑:全部已完成 -> 不再请求
    assert sum(calls.values()) == n1


def test_run_field_eval_overwrite_reextracts(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    calls: dict = {}
    _install_fakes(cfg, monkeypatch, calls)

    field_eval.run_field_eval(cfg)
    n1 = sum(calls.values())
    field_eval.run_field_eval(cfg, overwrite=True)
    assert sum(calls.values()) == n1 * 2  # 重抽一遍


def test_run_field_eval_missing_predictions(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    cfg.test_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.test_path.write_text("[]", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        field_eval.run_field_eval(cfg)


# ---------------------------------------------------------------------------
# CLI 解析
# ---------------------------------------------------------------------------
def test_parser_field_eval_flags():
    parser = build_parser()
    args = parser.parse_args(["field-eval", "-d", "mydata"])
    assert args.func is _cmd_field_eval
    assert args.dataset == "mydata" and args.overwrite is False

    args2 = parser.parse_args([
        "field-eval", "-d", "mydata", "--overwrite",
        "--label-extract-url", "http://h/", "--label-extract-token", "bearer T",
    ])
    assert args2.overwrite is True
    assert args2.label_extract_url == "http://h/" and args2.label_extract_token == "bearer T"
