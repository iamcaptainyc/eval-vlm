"""label-extract:描述后调远程服务抽取中文标签,落 label.jsonl。

HTTP 全程 monkeypatch(不真的联网):替换 label_extract._post_data / extract_one。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval_vlm import label_extract
from eval_vlm.cli import build_parser, _cmd_pred
from eval_vlm.config import Config, LabelExtractConfig


# 用户给的服务返回样例(截取 data 段;parse 只看 data.labels.cn)。
SAMPLE_DATA = {
    "main_task": "label_extract",
    "labels": {
        "cn": {
            "天气识别": ["无"],
            "自然风光": ["日落", "山脉", "森林"],
            "车辆识别": ["汽车"],
            "物品识别": ["无"],
        },
        "en": {
            "weather": ["none"],
            "nature": ["sunset", "mountains", "forest"],
            "vehicle": ["car_desc"],
            "item": ["none"],
        },
    },
}


# ---------------------------------------------------------------------------
# parse_cn_labels(纯逻辑)
# ---------------------------------------------------------------------------
def test_parse_cn_labels_flattens_non_none():
    """取 cn、丢「无」、按类目/列表顺序扁平合并。"""
    assert label_extract.parse_cn_labels(SAMPLE_DATA) == ["日落", "山脉", "森林", "汽车"]


def test_parse_cn_labels_all_none_empty():
    data = {"labels": {"cn": {"天气识别": ["无"], "物品识别": ["无"]}}}
    assert label_extract.parse_cn_labels(data) == []


def test_parse_cn_labels_dedupes_preserving_order():
    data = {"labels": {"cn": {"a": ["猫", "狗"], "b": ["狗", "鸟"]}}}
    assert label_extract.parse_cn_labels(data) == ["猫", "狗", "鸟"]


def test_parse_cn_labels_missing_labels():
    assert label_extract.parse_cn_labels({}) == []
    assert label_extract.parse_cn_labels({"labels": {}}) == []


def test_parse_cn_labels_custom_none_label():
    data = {"labels": {"cn": {"x": ["none"], "y": ["猫"]}}}
    assert label_extract.parse_cn_labels(data, none_label="none") == ["猫"]


# ---------------------------------------------------------------------------
# _post_data / extract_one:响应校验与重试(monkeypatch urlopen)
# ---------------------------------------------------------------------------
class _FakeResp:
    def __init__(self, payload: str):
        self._payload = payload.encode("utf-8")

    def read(self):
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_post_data_ok(monkeypatch):
    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["body"] = req.data
        captured["auth"] = req.get_header("Authorization")
        return _FakeResp(json.dumps({"code": 200, "data": SAMPLE_DATA}))

    monkeypatch.setattr(label_extract.urllib.request, "urlopen", fake_urlopen)
    le = LabelExtractConfig()
    data = label_extract._post_data("一段描述", le)
    assert data["labels"]["cn"]["车辆识别"] == ["汽车"]
    # 请求体是 {"text": ...},带 Authorization 头,地址拼接正确(无双斜杠)。
    assert json.loads(captured["body"].decode("utf-8")) == {"text": "一段描述"}
    assert captured["auth"] == le.auth_token
    assert captured["url"] == "https://canghai-agent-api-test.aijidou.com/api/v1/vlm/label-extract"


def test_post_data_non_200_raises(monkeypatch):
    monkeypatch.setattr(label_extract.urllib.request, "urlopen",
                        lambda req, timeout=None: _FakeResp(json.dumps({"code": 500, "msg": "boom"})))
    with pytest.raises(label_extract.LabelExtractError):
        label_extract._post_data("x", LabelExtractConfig())


def test_extract_one_retries_then_succeeds(monkeypatch):
    calls = {"n": 0}

    def flaky(text, le):
        calls["n"] += 1
        if calls["n"] < 2:
            raise label_extract.LabelExtractError("transient")
        return SAMPLE_DATA

    monkeypatch.setattr(label_extract, "_post_data", flaky)
    monkeypatch.setattr(label_extract.time, "sleep", lambda *_: None)   # 不真的退避等待
    le = LabelExtractConfig(max_retries=3)
    assert label_extract.extract_one("desc", le) == ["日落", "山脉", "森林", "汽车"]
    assert calls["n"] == 2


def test_extract_one_exhausts_retries(monkeypatch):
    def always_fail(text, le):
        raise label_extract.LabelExtractError("nope")

    monkeypatch.setattr(label_extract, "_post_data", always_fail)
    monkeypatch.setattr(label_extract.time, "sleep", lambda *_: None)
    with pytest.raises(label_extract.LabelExtractError):
        label_extract.extract_one("desc", LabelExtractConfig(max_retries=2))


# ---------------------------------------------------------------------------
# run_label_extract:读 predictions.jsonl -> 写 label.jsonl(monkeypatch extract_one)
# ---------------------------------------------------------------------------
def _make_cfg(tmp_path: Path) -> Config:
    """run_dir 钉到 tmp_path;串行抽取(max_concurrency=1)便于确定性断言。"""
    cfg = Config()
    cfg.run_dir_path = tmp_path
    cfg.label_extract = LabelExtractConfig(max_concurrency=1)
    return cfg


def _write_predictions(cfg: Config, rows: list[tuple[str, str]]) -> None:
    """写 predictions.jsonl:每行 LlamaFactory 记录(id + 末 assistant=描述)。"""
    cfg.predictions_path.parent.mkdir(parents=True, exist_ok=True)
    with cfg.predictions_path.open("w", encoding="utf-8") as f:
        for img, desc in rows:
            rec = {"id": img, "images": [img],
                   "messages": [{"role": "user", "content": "<image>请描述图片"},
                                {"role": "assistant", "content": desc}]}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def test_run_label_extract_end_to_end(tmp_path, monkeypatch):
    cfg = _make_cfg(tmp_path)
    _write_predictions(cfg, [("a.jpg", "日落下的山脉森林"), ("b.jpg", "一辆汽车")])

    seen = {}

    def fake_extract(text, le):
        seen[text] = True
        return ["日落", "山脉"] if "山脉" in text else ["汽车"]

    monkeypatch.setattr(label_extract, "extract_one", fake_extract)
    stats = label_extract.run_label_extract(cfg)

    assert stats["newly_completed"] == 2 and stats["errors"] == 0
    rows = [json.loads(ln) for ln in
            cfg.labels_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    by_img = {r["image"]: r["labels"] for r in rows}
    assert by_img == {"a.jpg": ["日落", "山脉"], "b.jpg": ["汽车"]}
    assert (cfg.run_dir / "label_extract_meta.json").exists()


def test_run_label_extract_records_failures(tmp_path, monkeypatch):
    """某条抽取失败:记 label_failures.jsonl,不中断,其余成功。"""
    cfg = _make_cfg(tmp_path)
    _write_predictions(cfg, [("ok.jpg", "好图"), ("bad.jpg", "坏图")])

    def fake_extract(text, le):
        if text == "坏图":
            raise label_extract.LabelExtractError("服务 500")
        return ["猫"]

    monkeypatch.setattr(label_extract, "extract_one", fake_extract)
    stats = label_extract.run_label_extract(cfg)

    assert stats["newly_completed"] == 1 and stats["errors"] == 1
    ok_rows = [json.loads(ln) for ln in
               cfg.labels_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert ok_rows == [{"image": "ok.jpg", "labels": ["猫"]}]
    fail_rows = [json.loads(ln) for ln in
                 cfg.label_failures_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(fail_rows) == 1 and fail_rows[0]["image"] == "bad.jpg"
    assert "服务 500" in fail_rows[0]["error"]


def test_run_label_extract_resume_skips_done(tmp_path, monkeypatch):
    """断点续跑:label.jsonl 里已成功的图片再次运行被跳过(不重复请求)。"""
    cfg = _make_cfg(tmp_path)
    _write_predictions(cfg, [("a.jpg", "d1"), ("b.jpg", "d2")])
    calls = {"n": 0}

    def counting_extract(text, le):
        calls["n"] += 1
        return ["x"]

    monkeypatch.setattr(label_extract, "extract_one", counting_extract)
    label_extract.run_label_extract(cfg)
    assert calls["n"] == 2
    # 再跑:全部已完成 -> 0 次新请求,label.jsonl 不重复累加。
    stats2 = label_extract.run_label_extract(cfg)
    assert calls["n"] == 2 and stats2["newly_completed"] == 0
    assert stats2["skipped_already_done"] == 2
    n_lines = len(cfg.labels_path.read_text(encoding="utf-8").splitlines())
    assert n_lines == 2


def test_run_label_extract_overwrite_reextracts(tmp_path, monkeypatch):
    cfg = _make_cfg(tmp_path)
    _write_predictions(cfg, [("a.jpg", "d1")])
    monkeypatch.setattr(label_extract, "extract_one", lambda t, le: ["y"])

    label_extract.run_label_extract(cfg)
    stats = label_extract.run_label_extract(cfg, overwrite=True)
    assert stats["newly_completed"] == 1 and stats["skipped_already_done"] == 0
    assert len(cfg.labels_path.read_text(encoding="utf-8").splitlines()) == 1  # 截断重写


def test_run_label_extract_no_predictions(tmp_path):
    cfg = _make_cfg(tmp_path)
    with pytest.raises(FileNotFoundError):
        label_extract.run_label_extract(cfg)


# ---------------------------------------------------------------------------
# CLI 解析 + 配置写回
# ---------------------------------------------------------------------------
def test_parser_label_extract_flags():
    parser = build_parser()
    args = parser.parse_args(["pred", "--datadir", "imgs"])
    assert args.func is _cmd_pred
    assert args.label_extract is False
    assert args.label_extract_url is None and args.label_extract_token is None

    args2 = parser.parse_args([
        "pred", "--datadir", "imgs", "--label-extract",
        "--label-extract-url", "http://h/", "--label-extract-token", "bearer T",
    ])
    assert args2.label_extract is True
    assert args2.label_extract_url == "http://h/" and args2.label_extract_token == "bearer T"


def test_pred_datadir_runs_label_extract(tmp_path, monkeypatch):
    """pred --datadir --label-extract(fake 后端):描述后抽取标签,产出 label.jsonl。"""
    import argparse

    monkeypatch.setenv("EVAL_VLM_CONFIG", str(tmp_path / "global.yaml"))
    imgs = tmp_path / "imgs"
    imgs.mkdir()
    for n in ("a.jpg", "b.jpg"):
        (imgs / n).write_bytes(b"")
    ws = tmp_path / "ws"

    # 不联网:抽取阶段整体替换 extract_one。
    monkeypatch.setattr(label_extract, "extract_one", lambda text, le: ["标签1", "标签2"])

    ns = argparse.Namespace(
        datadir=str(imgs), dataset=None, name=None, prompt=None, system_prompt=None,
        backend="fake", force=False, overwrite=False, mnn_config=None,
        mnn_image_max_side=None, base_url=None, model=None, workspace=str(ws),
        label_extract=True, label_extract_url=None, label_extract_token=None,
    )
    assert _cmd_pred(ns) == 0

    run_dir = ws / imgs.name / "trained-vlm"                 # fake -> openai.model 默认
    label_path = run_dir / "label.jsonl"
    assert label_path.exists()
    rows = [json.loads(ln) for ln in
            label_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(rows) == 2
    assert all(r["labels"] == ["标签1", "标签2"] and "image" in r for r in rows)
