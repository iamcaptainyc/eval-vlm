"""MNN(pymnn)后端:用注入的假 MNN 模块验证多模态 prompt 构造与文本捕获,
不依赖真实 pymnn(CI 无 GPU/无 MNN 也能跑)。另测并发降级与 CLI/工厂分发。"""
from __future__ import annotations

import sys
import types

import pytest

from eval_vlm.config import Config
from eval_vlm.data.schema import Turn
from eval_vlm.inference import build_backend, worker_count
from eval_vlm.inference.base import InferenceBackend


# ---------------------------------------------------------------------------
# 假 MNN 模块(MNN.llm / MNN.cv),记录调用以便断言
# ---------------------------------------------------------------------------
class _FakeImg:
    def __init__(self, h: int, w: int):
        self.shape = [h, w, 3]


class _FakeModel:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.loaded = False
        self.reset_calls = 0
        self.last_prompt = None
        self.last_stream = None
        self.last_max_tokens = None

    def load(self):
        self.loaded = True
        return True

    def reset(self):
        self.reset_calls += 1

    def response(self, content, stream=0, max_new_tokens=-1):
        self.last_prompt = content
        self.last_stream = stream
        self.last_max_tokens = max_new_tokens
        return "这是一张图片的描述"

    def get_context(self):
        return {"prompt_len": 7, "gen_seq_len": 5, "vision_us": 1234, "pixels_mp": 0.17}


@pytest.fixture
def fake_mnn(monkeypatch):
    """把假的 MNN.llm / MNN.cv 注入 sys.modules;返回 create 出来的模型句柄列表。"""
    created: list[_FakeModel] = []

    llm_mod = types.ModuleType("MNN.llm")
    cv_mod = types.ModuleType("MNN.cv")

    def _create(path, *a, **k):
        m = _FakeModel(path)
        created.append(m)
        return m

    llm_mod.create = _create
    cv_mod.imread = lambda p, *a, **k: _FakeImg(420, 420)

    mnn = types.ModuleType("MNN")
    mnn.llm = llm_mod
    mnn.cv = cv_mod

    monkeypatch.setitem(sys.modules, "MNN", mnn)
    monkeypatch.setitem(sys.modules, "MNN.llm", llm_mod)
    monkeypatch.setitem(sys.modules, "MNN.cv", cv_mod)
    return created


def _mnn_cfg(tmp_path) -> Config:
    imgs = tmp_path / "imgs"
    imgs.mkdir()
    (imgs / "a.jpg").write_bytes(b"")        # cv.imread 被 mock,空文件足够通过 exists()
    cfg = Config()
    cfg.inference.backend = "mnn"
    cfg.inference.mnn_config_path = str(tmp_path / "model" / "config.json")
    cfg.inference.max_tokens = 128
    cfg.data.media_root = str(imgs)
    return cfg


# ---------------------------------------------------------------------------
# 后端核心:prompt 构造 + 文本捕获
# ---------------------------------------------------------------------------
def test_mnn_complete_builds_multimodal_prompt(fake_mnn, tmp_path):
    from eval_vlm.inference.mnn_backend import MNNBackend

    cfg = _mnn_cfg(tmp_path)
    backend = MNNBackend(cfg)
    assert backend.thread_safe is False
    assert fake_mnn[0].loaded is True               # __init__ 里 load() 被调用

    ctx = [Turn(role="user", content="<image>请描述图片")]
    pred = backend.complete(ctx, ["a.jpg"], "a.jpg")

    assert pred.error is None
    assert pred.prediction == "这是一张图片的描述"
    model = fake_mnn[0]
    # stream=False 才能拿到返回文本;max_tokens 透传为 max_new_tokens。
    assert model.last_stream is False
    assert model.last_max_tokens == 128
    # <image> -> <img>image_0</img>;图片走 data/height/width dict。
    prompt = model.last_prompt
    assert prompt["text"] == "<img>image_0</img>请描述图片"
    assert len(prompt["images"]) == 1
    img = prompt["images"][0]
    assert img["height"] == 420 and img["width"] == 420
    assert isinstance(img["data"], _FakeImg)
    # 独立单图:每次推理前 reset 清状态。
    assert model.reset_calls == 1
    # 统计信息落到 raw。
    assert pred.raw["backend"] == "mnn" and pred.raw["prompt_len"] == 7


def test_mnn_missing_config_path_raises(fake_mnn, tmp_path):
    from eval_vlm.inference.mnn_backend import MNNBackend

    cfg = _mnn_cfg(tmp_path)
    cfg.inference.mnn_config_path = None
    with pytest.raises(ValueError) as e:
        MNNBackend(cfg)
    assert "mnn_config_path" in str(e.value)


def test_mnn_missing_image_records_error(fake_mnn, tmp_path):
    from eval_vlm.inference.mnn_backend import MNNBackend

    cfg = _mnn_cfg(tmp_path)
    backend = MNNBackend(cfg)
    ctx = [Turn(role="user", content="<image>请描述图片")]
    pred = backend.complete(ctx, ["missing.jpg"], "missing.jpg")
    assert pred.prediction == "" and pred.error is not None
    assert "build_prompt" in pred.error


def test_mnn_requires_single_image(fake_mnn, tmp_path):
    from eval_vlm.inference.mnn_backend import MNNBackend

    cfg = _mnn_cfg(tmp_path)
    backend = MNNBackend(cfg)
    ctx = [Turn(role="user", content="<image>请描述图片")]
    pred = backend.complete(ctx, ["a.jpg", "b.jpg"], "x")
    assert pred.error is not None and "单图" in pred.error


# ---------------------------------------------------------------------------
# 工厂分发:mnn / vllm 别名
# ---------------------------------------------------------------------------
def test_build_backend_dispatches_mnn(fake_mnn, tmp_path):
    from eval_vlm.inference.mnn_backend import MNNBackend

    cfg = _mnn_cfg(tmp_path)
    assert isinstance(build_backend(cfg), MNNBackend)


def test_build_backend_vllm_is_openai_alias():
    from eval_vlm.inference.openai_backend import OpenAIBackend

    cfg = Config()
    cfg.inference.backend = "vllm"
    assert isinstance(build_backend(cfg), OpenAIBackend)


def test_build_backend_unknown_raises():
    cfg = Config()
    cfg.inference.backend = "nope"
    with pytest.raises(ValueError) as e:
        build_backend(cfg)
    assert "nope" in str(e.value)


# ---------------------------------------------------------------------------
# 并发降级:非线程安全后端 -> 串行
# ---------------------------------------------------------------------------
class _Unsafe(InferenceBackend):
    thread_safe = False

    def complete(self, *a, **k):  # pragma: no cover - 不会被调用
        raise NotImplementedError


class _Safe(InferenceBackend):
    def complete(self, *a, **k):  # pragma: no cover
        raise NotImplementedError


def test_worker_count_serializes_unsafe_backend():
    cfg = Config()
    assert worker_count(_Unsafe(cfg), 8) == 1        # 强制串行
    assert worker_count(_Safe(cfg), 8) == 8          # 线程安全用配置值
    assert worker_count(_Safe(cfg), 0) == 1          # 至少 1


# ---------------------------------------------------------------------------
# CLI 解析:--backend mnn/vllm + --mnn-config
# ---------------------------------------------------------------------------
def test_parser_pred_mnn_flags():
    from eval_vlm.cli import build_parser, _cmd_pred

    parser = build_parser()
    args = parser.parse_args([
        "pred", "-d", "imgs", "--backend", "mnn",
        "--mnn-config", "/m/config.json",
    ])
    assert args.func is _cmd_pred
    assert args.backend == "mnn"
    assert args.mnn_config == "/m/config.json"

    # vllm 别名也能解析
    args2 = parser.parse_args(["pred", "-d", "imgs", "--backend", "vllm"])
    assert args2.backend == "vllm" and args2.mnn_config is None
