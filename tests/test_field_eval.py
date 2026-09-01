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
from eval_vlm.data.schema import Sample, Turn, EvalTurn


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
        Sample(id="a", turns=[Turn("user", "q"), Turn("assistant", "ref_a")],
               targets=[EvalTurn(turn_index=1, reference="ref_a")]),
        Sample(id="b", turns=[Turn("user", "q"), Turn("assistant", "ref_b")],
               targets=[EvalTurn(turn_index=1, reference="ref_b")]),
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

    # 结构化 JSON(供离线重渲染)+ 图文 HTML(图片内嵌)
    assert cfg.field_mismatches_json_path.exists()
    assert cfg.field_mismatches_html_path.exists()
    payload = json.loads(cfg.field_mismatches_json_path.read_text(encoding="utf-8"))
    assert payload["num_scored"] == 2
    assert [r["id"] for r in payload["rows"]] == ["b"]
    html = cfg.field_mismatches_html_path.read_text(encoding="utf-8")
    assert "样本 <code>b</code>" in html and "样本 <code>a</code>" not in html


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
# 描述轮跟随 eval.targets:取第 2 个 assistant 轮
# ---------------------------------------------------------------------------
def test_description_target_follows_eval_targets():
    """描述轮 = eval.targets 选中的目标轮:数字 2 -> 第 2 个 assistant 轮(而非第一轮)。"""
    s = Sample(id="a", turns=[
        Turn("user", "q1"), Turn("assistant", "r1"),
        Turn("user", "q2"), Turn("assistant", "r2"),
    ], targets=[EvalTurn(turn_index=3, reference="r2")])
    assert field_eval._description_target(s) == (3, "r2")
    # all 模式(多目标)回落第一个目标轮 = 第一轮描述
    s_all = Sample(id="a", turns=[Turn("user", "q"), Turn("assistant", "r1")],
                   targets=[EvalTurn(turn_index=1, reference="r1"),
                            EvalTurn(turn_index=1, reference="r1")])
    assert field_eval._description_target(s_all) == (1, "r1")
    # 无 targets(异常) -> (-1, "")
    assert field_eval._description_target(Sample(id="x")) == (-1, "")


def test_run_field_eval_second_round(tmp_path, monkeypatch):
    """eval.targets=2 时,field-eval 抽取并比对第 2 个 assistant 轮(不是第一轮)。"""
    cfg = _cfg(tmp_path)
    cfg.eval.targets = 2
    samples = [
        Sample(id="a", turns=[
            Turn("user", "<image>q1"), Turn("assistant", "ref_a1"),
            Turn("user", "q2"), Turn("assistant", "ref_a2"),
        ], targets=[EvalTurn(turn_index=3, reference="ref_a2")]),
        Sample(id="b", turns=[
            Turn("user", "<image>q1"), Turn("assistant", "ref_b1"),
            Turn("user", "q2"), Turn("assistant", "ref_b2"),
        ], targets=[EvalTurn(turn_index=3, reference="ref_b2")]),
    ]
    monkeypatch.setattr(field_eval, "load_samples", lambda cfg, source=None: samples)
    calls: dict = {}

    def fake_extract(text, le):
        calls[text] = calls.get(text, 0) + 1
        return dict({"主辅路": ["主路"]} if text.startswith(("ref_", "pred_")) else {})

    monkeypatch.setattr(field_eval, "extract_fields_one", fake_extract)
    cfg.test_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.test_path.write_text("[]", encoding="utf-8")
    # 运行级预测只含第 2 个 assistant 轮(turn=3)
    cfg.predictions_path.parent.mkdir(parents=True, exist_ok=True)
    with cfg.predictions_path.open("w", encoding="utf-8") as f:
        for sid in ("a", "b"):
            f.write(json.dumps({"id": sid, "turn": 3, "prediction": f"pred_{sid}2"}) + "\n")

    metrics = field_eval.run_field_eval(cfg)

    # 抽取的全是第二轮文本(ref_a2/pred_a2…),第一轮文本未被动过
    assert set(calls) == {"ref_a2", "pred_a2", "ref_b2", "pred_b2"}
    assert metrics["num_scored"] == 2
    assert metrics["per_field"]["主辅路"] == {"correct": 2, "total": 2, "accuracy": 1.0}


# ---------------------------------------------------------------------------
# CLI 解析
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# _cmd_field_eval:自给自足(无预测则自动 pred,有则直接评)
# ---------------------------------------------------------------------------
def _canned_metrics() -> dict:
    return {"overall": {"micro_accuracy": 0.5, "macro_accuracy": 0.5, "exact_match_rate": 0.3},
            "num_scored": 1, "num_pred_missing": 0, "skipped_ref": 0, "skipped_pred_error": 0,
            "fields": [], "per_field": {}, "per_value": {}}


def test_cmd_field_eval_auto_pred_when_missing(tmp_path, monkeypatch):
    """预测缺失/不完整 -> 先 _do_run(pred);完整覆盖 -> 直接评、不再 pred。"""
    import argparse
    from eval_vlm import cli
    from eval_vlm.config import Config
    from eval_vlm.data.schema import Sample, EvalTurn

    cfg = Config()
    cfg.run_dir_path = tmp_path
    samples = [Sample(id="a", targets=[EvalTurn(turn_index=1, reference="x")])]
    monkeypatch.setattr(cli, "_resolve_folder", lambda args: tmp_path)
    monkeypatch.setattr(cli, "_persist_overrides", lambda folder, args: [])
    monkeypatch.setattr(cli, "_report_persist", lambda *a, **k: None)
    monkeypatch.setattr(cli, "load_dataset_config", lambda folder: cfg)
    monkeypatch.setattr(cli, "load_samples", lambda cfg, source=None: samples)
    monkeypatch.setattr(cli, "run_field_eval", lambda cfg, overwrite=False: _canned_metrics())
    calls = {"pred": 0}
    monkeypatch.setattr(cli, "_do_run", lambda cfg, tag="pred": calls.__setitem__("pred", calls["pred"] + 1))

    ns = argparse.Namespace(dataset="ds", overwrite=False, fail_fast=False, workspace=None)

    # 预测不存在 -> 自动 pred 一次
    assert cli._cmd_field_eval(ns) == 0
    assert calls["pred"] == 1

    # 预测**完整覆盖**目标轮(id=a,turn=1)-> 直接评,不再 pred
    cfg.predictions_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.predictions_path.write_text(
        json.dumps({"id": "a", "turn": 1, "prediction": "p"}) + "\n", encoding="utf-8")
    assert cli._cmd_field_eval(ns) == 0
    assert calls["pred"] == 1

    # 预测存在但**不完整**(缺 turn=1)-> 补跑 pred
    cfg.predictions_path.write_text(
        json.dumps({"id": "a", "turn": 0, "prediction": "p"}) + "\n", encoding="utf-8")
    assert cli._cmd_field_eval(ns) == 0
    assert calls["pred"] == 2


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


def test_parser_field_eval_targets_flag():
    """--targets 纯数字转 int(第 N 轮)、字符串原样保留。"""
    from eval_vlm import cli
    parser = build_parser()
    assert parser.parse_args(["field-eval", "-d", "mydata", "--targets", "2"]).targets == 2
    assert parser.parse_args(["field-eval", "-d", "mydata", "--targets", "first"]).targets == "first"
    assert parser.parse_args(["field-eval", "-d", "mydata", "--targets", "all"]).targets == "all"
    assert cli._parse_targets("2") == 2
    assert cli._parse_targets(" 2 ") == 2
    assert cli._parse_targets("last") == "last"
    assert cli._parse_targets("0") == 0


def test_persist_targets_int_roundtrip(tmp_path):
    """field-eval --targets 2 永久写回 eval.targets 为 int,重读仍为 int。"""
    import argparse

    from eval_vlm import cli, workspace
    from eval_vlm.config import load_dataset_config
    from eval_vlm.data.loader import load_samples

    ws = tmp_path / "ws"
    folder = workspace.init_dataset(
        str(Path(__file__).parent / "fixtures" / "llamafactory_tworound.json"),
        ws, name="ds", media_root=str(Path(__file__).parent / "fixtures"))
    cli._persist_overrides(folder, argparse.Namespace(targets=2))

    cfg = load_dataset_config(folder)
    assert cfg.eval.targets == 2                        # int,不是字符串
    s = load_samples(cfg, source=cfg.source_path)[0]
    assert s.targets[0].turn_index == 3                 # 第 2 个 assistant 轮


# ---------------------------------------------------------------------------
# report_assets.image_ref_to_html_src:本地缩放内嵌 / URL 透传 / 缺失占位
# ---------------------------------------------------------------------------
def _fixtures_images_dir() -> Path:
    return Path(__file__).parent / "fixtures" / "images"


def test_image_ref_to_html_src_embeds_local(tmp_path):
    from eval_vlm.report_assets import image_ref_to_html_src
    cfg = Config()
    cfg.data.media_root = str(_fixtures_images_dir())
    src, err = image_ref_to_html_src("sample1.png", cfg)
    assert err is None
    assert src.startswith("data:image/jpeg;base64,")


def test_image_ref_to_html_src_resizes_large(tmp_path):
    from PIL import Image
    import base64
    import io

    from eval_vlm.report_assets import image_ref_to_html_src

    big = tmp_path / "big.png"
    Image.new("RGB", (2000, 1000), (255, 0, 0)).save(big)

    cfg = Config()
    cfg.data.media_root = str(tmp_path)
    src, err = image_ref_to_html_src("big.png", cfg)
    assert err is None
    raw = base64.b64decode(src.split(",", 1)[1])
    with Image.open(io.BytesIO(raw)) as im:
        assert max(im.size) <= 960                        # 长边被缩到上限内


def test_image_ref_to_html_src_missing_file(tmp_path):
    from eval_vlm.report_assets import image_ref_to_html_src
    cfg = Config()
    cfg.data.media_root = str(tmp_path)
    src, err = image_ref_to_html_src("nope.png", cfg)
    assert src is None
    assert err and "nope.png" in err


def test_image_ref_to_html_src_passthrough_urls():
    from eval_vlm.report_assets import image_ref_to_html_src
    cfg = Config()
    assert image_ref_to_html_src("https://example.com/a.jpg", cfg) == \
        ("https://example.com/a.jpg", None)
    assert image_ref_to_html_src("data:image/png;base64,AAAA", cfg) == \
        ("data:image/png;base64,AAAA", None)


# ---------------------------------------------------------------------------
# HTML 渲染:转义 / ✓✗ 标记 / pred_missing 标签 / 图片内嵌 / 占位符
# ---------------------------------------------------------------------------
def _render_metrics(**over) -> dict:
    m = {"overall": {"micro_accuracy": 0.5, "macro_accuracy": 0.5, "exact_match_rate": 0.5},
         "num_scored": 2, "num_pred_missing": 0,
         "fields": ["主辅路", "警示标志"]}
    m.update(over)
    return m


def test_render_mismatches_html_escapes_and_marks():
    rows = [{
        "id": "b", "images": [], "state": "compared",
        "fields": [
            {"field": "主辅路", "ref": ["主路"], "pred": ["主路"], "correct": True},
            {"field": "警示标志", "ref": ["<b>学校路口</b>"], "pred": ["礼让行人"], "correct": False},
        ],
    }]
    cfg = Config()
    html = field_eval._render_mismatches_html(rows, _render_metrics(), cfg)
    assert "<html" in html and "</html>" in html
    assert "✓" in html and "✗" in html
    assert "&lt;b&gt;学校路口&lt;/b&gt;" in html        # 转义后
    assert "<b>学校路口</b>" not in html                # 原文不残留
    assert "样本 <code>b</code>" in html


def test_render_mismatches_html_pred_missing_tagged():
    rows = [{
        "id": "c", "images": [], "state": "pred_missing",
        "fields": [{"field": "主辅路", "ref": ["主路"], "pred": [], "correct": False}],
    }]
    cfg = Config()
    html = field_eval._render_mismatches_html(rows, _render_metrics(num_pred_missing=1), cfg)
    assert 'class="card pred-missing"' in html
    assert "模型未产出描述" in html
    assert 'data-bad-fields="主辅路"' in html


def test_render_mismatches_html_no_rows():
    cfg = Config()
    html = field_eval._render_mismatches_html([], _render_metrics(), cfg)
    assert "无字段失配" in html


def test_render_mismatch_card_embeds_local_image():
    cfg = Config()
    cfg.data.media_root = str(_fixtures_images_dir())
    row = {"id": "a", "images": ["sample1.png"], "state": "compared",
           "fields": [{"field": "主辅路", "ref": ["主路"], "pred": ["主路"], "correct": True}]}
    card = field_eval._render_mismatch_card(row, cfg)
    assert '<img src="data:image/jpeg;base64,' in card


def test_render_mismatch_card_missing_image_placeholder(tmp_path):
    cfg = Config()
    cfg.data.media_root = str(tmp_path)                 # 空目录,图片必缺失
    row = {"id": "a", "images": ["nope.png"], "state": "compared",
           "fields": [{"field": "主辅路", "ref": ["主路"], "pred": [], "correct": False}]}
    card = field_eval._render_mismatch_card(row, cfg)
    assert 'class="img-placeholder"' in card
    assert "nope.png" in card
