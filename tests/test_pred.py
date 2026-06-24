"""pred 命令:可自定义 vLLM API(folder config.yaml)+ 声明式多轮对话模板。"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval_vlm import workspace
from eval_vlm.cli import build_parser, _cmd_pred
from eval_vlm.config import DEFAULT_PROMPT, PredConfig, load_config
from eval_vlm.data.loader import load_samples
from eval_vlm.predict import build_context


@pytest.fixture
def temp_global(tmp_path, monkeypatch):
    """全局配置指向临时文件,避免污染真实 ~/.eval_vlm。返回 (workspace, 图片文件夹)。"""
    monkeypatch.setenv("EVAL_VLM_CONFIG", str(tmp_path / "global.yaml"))
    imgs = tmp_path / "imgs"
    imgs.mkdir()
    for n in ("a.jpg", "b.jpg", "c.png"):        # fake 后端不打开图,空文件即可被 list_images 发现
        (imgs / n).write_bytes(b"")
    return tmp_path / "ws", imgs


# ---------------------------------------------------------------------------
# PredConfig 解析 + build_context / 校验(纯逻辑)
# ---------------------------------------------------------------------------
def test_predconfig_parsing(tmp_path):
    """pred: 块被 _build 解析:prompt/system_prompt 标量,template 为 list[dict]。"""
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(
        "pred:\n"
        "  prompt: 描述一下\n"
        "  system_prompt: 你是助手\n"
        "  template:\n"
        "    - {role: user, content: 'hi'}\n"
        "    - {role: assistant, content: 'ok'}\n"
        "    - {role: user, content: '<image>go'}\n",
        encoding="utf-8",
    )
    cfg = load_config(cfg_path)
    assert cfg.pred.prompt == "描述一下"
    assert cfg.pred.system_prompt == "你是助手"
    assert isinstance(cfg.pred.template, list) and len(cfg.pred.template) == 3
    assert cfg.pred.template[0] == {"role": "user", "content": "hi"}


def test_build_context_single_turn_autoprepend():
    """单轮:不含 <image> 自动前置;已含则不重复。"""
    assert build_context(PredConfig(prompt="描述"))[0].content == "<image>描述"
    assert build_context(PredConfig(prompt="先看<image>"))[0].content == "先看<image>"


def test_build_context_prompt_override():
    """prompt_override(CLI --prompt)优先于 pred.prompt。"""
    turns = build_context(PredConfig(prompt="X"), prompt_override="Y")
    assert turns[0].content == "<image>Y"


def test_build_context_template_overrides_prompt():
    """设了 template 就用它(覆盖 prompt),原样返回各轮。"""
    tpl = [{"role": "user", "content": "<image>看图"}]
    turns = build_context(PredConfig(prompt="忽略我", template=tpl))
    assert len(turns) == 1 and turns[0].content == "<image>看图"


@pytest.mark.parametrize("tpl, frag", [
    ([{"role": "user", "content": "无图"}], "恰好出现 1 次"),
    ([{"role": "user", "content": "<image><image>"}], "恰好出现 1 次"),
    ([{"role": "assistant", "content": "<image>"}, {"role": "user", "content": "x"}],
     "必须位于 user"),
    ([{"role": "user", "content": "<image>x"}, {"role": "assistant", "content": "y"}],
     "最后一轮必须是 user"),
])
def test_build_context_validation_errors(tpl, frag):
    with pytest.raises(ValueError) as e:
        build_context(PredConfig(template=tpl))
    assert frag in str(e.value)


def test_build_context_bad_template_entry():
    with pytest.raises(ValueError):
        build_context(PredConfig(template=[{"role": "user"}]))   # 缺 content


# ---------------------------------------------------------------------------
# CLI 解析:--prompt 默认 None(否则永远压过 config);新 flag 可解析
# ---------------------------------------------------------------------------
def test_parser_pred_prompt_default_none_and_flags():
    parser = build_parser()
    args = parser.parse_args(["pred", "--datadir", "imgs"])
    assert args.func is _cmd_pred
    assert args.prompt is None and args.system_prompt is None
    assert args.backend is None and args.force is False and args.overwrite is False

    args2 = parser.parse_args([
        "pred", "-d", "imgs", "--prompt", "P", "--system-prompt", "S",
        "--backend", "fake", "--force", "--overwrite",
        "--base-url", "http://h/v1", "--model", "m",
    ])
    assert args2.prompt == "P" and args2.system_prompt == "S"
    assert args2.backend == "fake" and args2.force is True and args2.overwrite is True
    assert args2.base_url == "http://h/v1" and args2.model == "m"


# ---------------------------------------------------------------------------
# 生成/读取 config.yaml + 端到端(fake 后端)
# ---------------------------------------------------------------------------
# pred 产物落 工作目录/<图片夹名>/<inference.model>;fake 流程模型名取模板默认。
PRED_MODEL = "trained-vlm"


def _run_pred(ws: Path, imgs: Path, **kw):
    """构造 argparse.Namespace 直接调 _cmd_pred(默认 fake 后端)。"""
    import argparse
    ns = argparse.Namespace(
        datadir=str(imgs), name=None, prompt=None, system_prompt=None,
        backend="fake", force=False, overwrite=False, mnn_config=None,
        base_url=None, model=None, workspace=str(ws),
    )
    for k, v in kw.items():
        setattr(ns, k, v)
    return _cmd_pred(ns)


def _pred_out(ws: Path, imgs: Path) -> Path:
    """pred 产物目录(模型子目录);config.yaml 仍在其父级(数据集文件夹)。"""
    return ws / imgs.name / PRED_MODEL


def test_pred_generates_then_reads_config(temp_global):
    """首跑生成 config.yaml;再跑沿用(不重建);--force 重建。"""
    ws, imgs = temp_global
    config_path = ws / imgs.name / "config.yaml"

    _run_pred(ws, imgs)
    assert config_path.exists()
    assert "pred:" in config_path.read_text(encoding="utf-8")

    # 手改 config,再跑不带 --force -> 内容保持(沿用,不重建)
    marker = config_path.read_text(encoding="utf-8") + "\n# user-edit\n"
    config_path.write_text(marker, encoding="utf-8")
    _run_pred(ws, imgs)
    assert "# user-edit" in config_path.read_text(encoding="utf-8")

    # --force -> 重建,手改被覆盖
    _run_pred(ws, imgs, force=True)
    assert "# user-edit" not in config_path.read_text(encoding="utf-8")


def test_pred_end_to_end_default_single_turn(temp_global):
    """默认单轮:产物为 [user(<image>请描述图片), assistant] 的 LlamaFactory 记录。"""
    ws, imgs = temp_global
    _run_pred(ws, imgs)
    out = _pred_out(ws, imgs)
    lines = (out / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3                                  # 3 张图
    rec = json.loads(lines[0])
    assert rec["images"] and rec["messages"][0]["content"] == f"<image>{DEFAULT_PROMPT}"
    assert rec["messages"][-1]["role"] == "assistant"


def test_pred_end_to_end_multiturn_template(temp_global):
    """多轮模板(预置 config.yaml):记录含全部模板轮 + 末 assistant;system_prompt 映射。"""
    ws, imgs = temp_global
    ds_dir = ws / imgs.name                                 # 数据集文件夹(放 config.yaml)
    ds_dir.mkdir(parents=True)
    (ds_dir / "config.yaml").write_text(
        "data:\n  media_root: " + str(imgs) + "\n"
        "inference:\n  backend: fake\n"
        "pred:\n"
        "  system_prompt: 你是图像描述助手\n"
        "  template:\n"
        "    - {role: user, content: '我会给你一张图,请客观描述。'}\n"
        "    - {role: assistant, content: '好的。'}\n"
        "    - {role: user, content: '<image>请描述这张图片'}\n",
        encoding="utf-8",
    )
    _run_pred(ws, imgs)                                     # config 已存在 -> 沿用

    out = _pred_out(ws, imgs)                               # 产物在模型子目录
    rec = json.loads((out / "predictions.jsonl").read_text(encoding="utf-8").splitlines()[0])
    roles = [m["role"] for m in rec["messages"]]
    assert roles == ["user", "assistant", "user", "assistant"]   # 3 模板轮 + 末答案
    # fake 后端回显最后一个 user 文本(去 <image>)
    assert rec["messages"][-1]["content"] == "请描述这张图片"

    meta = json.loads((out / "pred_meta.json").read_text(encoding="utf-8"))
    assert meta["num_context_turns"] == 3 and meta["system_prompt"] == "你是图像描述助手"
    assert meta["prompt"] is None                           # 多轮时 prompt 记 null


def test_pred_record_roundtrips_as_llamafactory(temp_global):
    """产出的记录是合法 LlamaFactory(<image> 数==images 数),可被 loader 解析回流。"""
    ws, imgs = temp_global
    _run_pred(ws, imgs)
    out = _pred_out(ws, imgs)
    # predictions.jsonl 是逐行 JSON;转成 JSON 数组喂回 loader(校验 <image> 数==images 数)。
    rows = [json.loads(ln) for ln in
            (out / "predictions.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip()]
    arr = out / "as_dataset.json"
    arr.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")

    cfg = load_config(ws / imgs.name / "config.yaml")
    samples = load_samples(cfg, source=arr)                # 能解析即说明记录是合法 LlamaFactory
    assert len(samples) == 3
    assert all(s.images and any("<image>" in t.content for t in s.turns) for s in samples)


def test_pred_resume_skips_done(temp_global):
    """断点续跑:已成功的图片再次运行被跳过(不重复追加)。"""
    ws, imgs = temp_global
    _run_pred(ws, imgs)
    out = _pred_out(ws, imgs)
    n_first = len((out / "predictions.jsonl").read_text(encoding="utf-8").splitlines())
    _run_pred(ws, imgs)                                     # 再跑一次
    n_second = len((out / "predictions.jsonl").read_text(encoding="utf-8").splitlines())
    assert n_first == n_second == 3                         # 没有重复追加


def test_pred_overwrite_reinfers_all(temp_global):
    """--overwrite:无视已有结果整份重跑(截断重写),不做断点续跑跳过。"""
    ws, imgs = temp_global
    _run_pred(ws, imgs)
    out = _pred_out(ws, imgs)
    pred_path = out / "predictions.jsonl"
    assert len(pred_path.read_text(encoding="utf-8").splitlines()) == 3

    # 续跑(默认):全部已完成 -> 跳过,无新增
    _run_pred(ws, imgs)
    assert len(pred_path.read_text(encoding="utf-8").splitlines()) == 3

    # --overwrite:截断重写,重新描述 3 张(仍是 3 行,但是本轮全部 newly_completed)
    _run_pred(ws, imgs, overwrite=True)
    lines = pred_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3                                  # 截断重写,无重复累加


def test_pred_persists_model_into_config(temp_global):
    """pred 的 --model 永久写回 config.yaml,产物落到对应模型子目录。"""
    ws, imgs = temp_global
    _run_pred(ws, imgs, model="vlm_v2")
    cfg_text = (ws / imgs.name / "config.yaml").read_text(encoding="utf-8")
    assert "vlm_v2" in cfg_text                             # 写回 inference.model
    assert (ws / imgs.name / "vlm_v2" / "predictions.jsonl").exists()
