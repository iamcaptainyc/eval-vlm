"""MNN(pymnn)后端:用注入的假 MNN 模块验证多模态 prompt 构造与文本捕获,
不依赖真实 pymnn(CI 无 GPU/无 MNN 也能跑)。另测并发降级与 CLI/工厂分发。"""
from __future__ import annotations

import sys
import types
from pathlib import Path

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
    cfg.inference.mnn.config_path = str(tmp_path / "model" / "config.json")
    cfg.inference.mnn.max_tokens = 128
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


def test_mnn_webp_transcodes_via_pillow(fake_mnn, tmp_path):
    """.webp 等原生解码器不认的格式:先经 Pillow 转码成临时 PNG 再 imread,
    避免 MNN.cv.imread 打不开图后返回非法 Var、读 .shape 触发 Segfault。"""
    from PIL import Image

    from eval_vlm.inference.mnn_backend import MNNBackend

    cfg = _mnn_cfg(tmp_path)
    webp = Path(cfg.data.media_root) / "城市.webp"
    try:
        Image.new("RGB", (8, 8), (10, 20, 30)).save(webp, format="WEBP")
    except Exception as e:  # noqa: BLE001 - 本环境 Pillow 无 webp 支持则跳过
        pytest.skip(f"Pillow 无 WEBP 支持: {e}")

    backend = MNNBackend(cfg)
    calls: list[str] = []

    def _recording_imread(p, *a, **k):
        calls.append(p)
        return _FakeImg(8, 8)

    backend._cv.imread = _recording_imread

    ctx = [Turn(role="user", content="<image>请描述图片")]
    pred = backend.complete(ctx, ["城市.webp"], "城市.webp")

    assert pred.error is None
    assert pred.prediction == "这是一张图片的描述"
    # 走了转码分支:imread 收到的是临时 .png,而非原始 .webp。
    assert calls and calls[0].lower().endswith(".png")
    assert not calls[0].lower().endswith(".webp")
    # 临时文件用完即删,不留垃圾。
    assert not Path(calls[0]).exists()


def test_mnn_jpg_uses_native_imread_directly(fake_mnn, tmp_path):
    """原生支持的扩展名(.jpg)直接走 imread 快路,不经 Pillow 转码。"""
    from eval_vlm.inference.mnn_backend import MNNBackend

    cfg = _mnn_cfg(tmp_path)
    backend = MNNBackend(cfg)
    calls: list[str] = []

    def _recording_imread(p, *a, **k):
        calls.append(p)
        return _FakeImg(420, 420)

    backend._cv.imread = _recording_imread

    ctx = [Turn(role="user", content="<image>请描述图片")]
    pred = backend.complete(ctx, ["a.jpg"], "a.jpg")

    assert pred.error is None
    # 直接拿到原始 .jpg 路径,无临时 png。
    assert calls and calls[0].lower().endswith("a.jpg")


def test_mnn_response_falls_back_to_two_arg_signature(fake_mnn, tmp_path):
    """旧版 pymnn 绑定 response 仅接受 (content, stream),三参调用抛 TypeError;
    后端应捕获并退化为两参调用,而非整批 148 张全失败。"""
    from eval_vlm.inference.mnn_backend import MNNBackend

    cfg = _mnn_cfg(tmp_path)
    backend = MNNBackend(cfg)
    assert backend._response_takes_max_tokens is True   # 初始乐观假设新版

    calls: list[tuple] = []

    def _old_response(content, stream=0):   # 没有 max_new_tokens 形参
        calls.append((content, stream))
        return "旧绑定的描述"

    backend.model.response = _old_response

    ctx = [Turn(role="user", content="<image>请描述图片")]
    pred = backend.complete(ctx, ["a.jpg"], "a.jpg")

    assert pred.error is None
    assert pred.prediction == "旧绑定的描述"
    # 只成功调用了一次(两参);记住退化,后续不再尝试三参。
    assert len(calls) == 1 and calls[0][1] is False
    assert backend._response_takes_max_tokens is False


def test_mnn_matches_latest_pymnn_wrapper(fake_mnn, tmp_path):
    """对齐最新 pymnn Python 包装(MNN/llm/__init__.py)的真实形态:
      - response(self, prompt, stream=False):仅两参,多传 max_new_tokens 抛 TypeError;
      - set_config(dict):收 dict(内部 json.dumps);
      - context 属性:Context 对象(非 dict),按属性读统计。
    断言后端能退化为两参、经 set_config(dict) 设上限、并从 context 属性收集统计。"""
    from eval_vlm.inference.mnn_backend import MNNBackend

    class _Ctx:
        prompt_len = 11
        gen_seq_len = 9
        vision_us = 2000
        prefill_us = 5000
        decode_us = 8000
        pixels_mp = 0.18

    class _WrapperModel:
        def __init__(self):
            self.set_config_args: list = []
            self.response_calls: list = []

        def load(self):
            return True

        def reset(self):
            pass

        def response(self, prompt, stream=False):   # 两参,无 max_new_tokens
            self.response_calls.append((prompt, stream))
            return "最新包装的描述"

        def set_config(self, config):                # 收 dict
            self.set_config_args.append(config)
            return True

        @property
        def context(self):
            return _Ctx()

    cfg = _mnn_cfg(tmp_path)
    backend = MNNBackend(cfg)
    backend.model = _WrapperModel()

    ctx = [Turn(role="user", content="<image>请描述图片")]
    pred = backend.complete(ctx, ["a.jpg"], "a.jpg")

    assert pred.error is None
    assert pred.prediction == "最新包装的描述"
    # 退化为两参调用(stream=False),且只成功调一次。
    assert len(backend.model.response_calls) == 1
    assert backend.model.response_calls[0][1] is False
    assert backend._response_takes_max_tokens is False
    # 此处 model 在 __init__ 后才换成 _WrapperModel,故采样配置(在 __init__ 时下发到
    # 原 fake model)不计入这里;_WrapperModel 只收到 complete 退化后下发的 max_new_tokens。
    # 采样/重复抑制设置的下发由 test_mnn_applies_sampler_config_on_init 覆盖。
    assert backend.model.set_config_args == [{"max_new_tokens": 128}]
    # 统计从 context 属性对象收集(非 get_context dict)。
    assert pred.raw["prompt_len"] == 11 and pred.raw["decode_us"] == 8000
    assert pred.raw["pixels_mp"] == 0.18


def test_mnn_applies_sampler_config_on_init(fake_mnn, tmp_path):
    """默认启用 penalty 采样器抑制小模型的 "\\n\\n\\n…" 退化:__init__ 时经 set_config
    下发 sampler_type/penalty_sampler/repetition_penalty 等;temperature/top_k/top_p
    默认 None 不下发(沿用 MNN 默认)。"""
    from eval_vlm.inference.mnn_backend import MNNBackend

    pushed: list = []

    class _CfgModel:
        def load(self):
            return True

        def reset(self):
            pass

        def response(self, prompt, stream=False):
            return "ok"

        def set_config(self, config):
            pushed.append(config)
            return True

    cfg = _mnn_cfg(tmp_path)
    backend = MNNBackend(cfg)
    backend.model = _CfgModel()
    backend._apply_sampler_config()

    assert len(pushed) == 1
    sent = pushed[0]
    assert sent["sampler_type"] == "penalty"
    assert sent["penalty_sampler"] == "greedy"
    assert sent["repetition_penalty"] == 1.1
    assert sent["penalty_window"] == 0
    # 未设置的随机采样项不下发,避免覆盖 MNN 默认。
    assert "temperature" not in sent and "top_k" not in sent and "top_p" not in sent


def test_mnn_sampler_config_disabled_when_type_empty(fake_mnn, tmp_path):
    """sampler_type="" 时完全不下发采样配置,沿用模型 config.json 自带设置。"""
    from eval_vlm.inference.mnn_backend import MNNBackend

    pushed: list = []

    class _CfgModel:
        def load(self):
            return True

        def reset(self):
            pass

        def response(self, prompt, stream=False):
            return "ok"

        def set_config(self, config):
            pushed.append(config)
            return True

    cfg = _mnn_cfg(tmp_path)
    cfg.inference.mnn.sampler_type = ""
    backend = MNNBackend(cfg)
    backend.model = _CfgModel()
    backend._apply_sampler_config()

    assert pushed == []


def test_mnn_missing_config_path_raises(fake_mnn, tmp_path):
    from eval_vlm.inference.mnn_backend import MNNBackend

    cfg = _mnn_cfg(tmp_path)
    cfg.inference.mnn.config_path = None
    with pytest.raises(ValueError) as e:
        MNNBackend(cfg)
    assert "config_path" in str(e.value)


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
        "pred", "--datadir", "imgs", "--backend", "mnn",
        "--mnn-config", "/m/config.json",
    ])
    assert args.func is _cmd_pred
    assert args.backend == "mnn"
    assert args.mnn_config == "/m/config.json"

    # vllm 别名也能解析
    args2 = parser.parse_args(["pred", "--datadir", "imgs", "--backend", "vllm"])
    assert args2.backend == "vllm" and args2.mnn_config is None
