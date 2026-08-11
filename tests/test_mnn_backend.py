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
    from PIL import Image

    imgs = tmp_path / "imgs"
    imgs.mkdir()
    # 真实小图:新版 _imread 用 Pillow 打开原图取尺寸(对齐训练预处理),空文件会打不开。
    Image.new("RGB", (420, 420), (10, 20, 30)).save(imgs / "a.jpg", format="JPEG")
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
    """原生支持且无需缩放的图(.jpg,尺寸在训练预处理范围内)直接走 imread 快路。"""
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


def test_mnn_target_size_matches_llamafactory_rules(fake_mnn, tmp_path):
    """_target_size 复刻 LlamaFactory mm_plugin 的训练预处理:
    超 image_max_pixels 按 sqrt 因子缩小、低于 image_min_pixels 放大、
    最小边钳 28、极端长宽比钳 180 倍、image_max_side 兜底。"""
    from eval_vlm.inference.mnn_backend import MNNBackend

    cfg = _mnn_cfg(tmp_path)
    backend = MNNBackend(cfg)

    # 1920x1080 = 2073600 px > 768*768 -> factor=sqrt(589824/2073600)=0.5333…
    w, h = backend._target_size(1920, 1080)
    assert (w, h) == (1024, 576)
    # 范围内的图不动。
    assert backend._target_size(420, 420) == (420, 420)
    # 过小的图放大到 image_min_pixels(32*32):8x8 -> 32x32。
    assert backend._target_size(8, 8) == (32, 32)
    # 最小边钳 28:100x16(1600px 在 min/max 之间)-> 高钳到 28。
    assert backend._target_size(100, 16) == (100, 28)
    # 长宽比 > 200 -> 长边钳到短边×180(先经 max_pixels 缩小后判断)。
    w, h = backend._target_size(30000, 100)
    assert w / h <= 200
    # image_max_side 兜底缩放后短边重新钳 28(否则 MNN 按 28 对齐会取整到 0 个 patch)。
    cfg.inference.mnn.image_max_pixels = 0
    cfg.inference.mnn.image_min_pixels = 0
    w, h = backend._target_size(30000, 150)
    assert max(w, h) <= cfg.inference.mnn.image_max_side and min(w, h) >= 28
    # 关闭各限制(<=0)则原样返回。
    cfg.inference.mnn.image_max_side = 0
    assert backend._target_size(5000, 5000) == (5000, 5000)


def test_mnn_downscales_oversized_image_like_training(fake_mnn, tmp_path):
    """超过 image_max_pixels 的大图:经 Pillow 按训练规则缩小后转临时 PNG 再 imread,
    而不是原尺寸直喂(那会产生远超训练分布的视觉 token,答案偏离训练效果)。"""
    from PIL import Image

    from eval_vlm.inference.mnn_backend import MNNBackend

    cfg = _mnn_cfg(tmp_path)
    big = Path(cfg.data.media_root) / "big.jpg"
    Image.new("RGB", (1920, 1080), (1, 2, 3)).save(big, format="JPEG")

    backend = MNNBackend(cfg)
    sizes: list[tuple] = []
    calls: list[str] = []

    def _recording_imread(p, *a, **k):
        calls.append(p)
        with Image.open(p) as im:
            sizes.append(im.size)
        return _FakeImg(im.size[1], im.size[0])

    backend._cv.imread = _recording_imread

    ctx = [Turn(role="user", content="<image>请描述图片")]
    pred = backend.complete(ctx, ["big.jpg"], "big.jpg")

    assert pred.error is None
    assert calls[0].lower().endswith(".png")     # 走了 Pillow 缩放转码
    assert sizes[0] == (1024, 576)               # 1920x1080 -> ~768*768 总像素


def test_mnn_system_prompt_pushed_on_init(fake_mnn, tmp_path):
    """inference.mnn.system_prompt 设置时,__init__ 经 set_config 下发给引擎
    (由 MNN apply_chat_template 拼进对话模板,与训练的 system 轮对齐)。"""
    from eval_vlm.inference.mnn_backend import MNNBackend

    pushed: list = []
    _FakeModel.set_config = lambda self, config: pushed.append(config) or True
    try:
        cfg = _mnn_cfg(tmp_path)
        cfg.inference.mnn.system_prompt = "You are a helpful assistant."
        MNNBackend(cfg)
    finally:
        del _FakeModel.set_config

    assert {"system_prompt": "You are a helpful assistant."} in pushed


def test_mnn_no_system_prompt_not_pushed(fake_mnn, tmp_path):
    """system_prompt 未设(None)时不下发,沿用模型 config.json 自带值。"""
    from eval_vlm.inference.mnn_backend import MNNBackend

    pushed: list = []
    _FakeModel.set_config = lambda self, config: pushed.append(config) or True
    try:
        cfg = _mnn_cfg(tmp_path)
        MNNBackend(cfg)
    finally:
        del _FakeModel.set_config

    assert not any(isinstance(c, dict) and "system_prompt" in c for c in pushed)


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
    # 从引擎计时派生的 vLLM 风格指标(prompt_len=11, decode_len=9,
    # vision=2000us, prefill=5000us, decode=8000us):
    assert pred.raw["ttft_ms"] == 7.0                       # (2000+5000)/1000
    assert pred.raw["e2e_ms"] == 15.0                       # (2000+5000+8000)/1000
    assert pred.raw["tpot_ms"] == round(8000 / 1000.0 / 9, 3)
    assert pred.raw["prefill_toks_per_s"] == round(11 / (5000 / 1e6), 2)
    assert pred.raw["decode_toks_per_s"] == round(9 / (8000 / 1e6), 2)
    assert pred.raw["total_toks_per_s"] == round((11 + 9) / (15000 / 1e6), 2)


def test_mnn_perf_metrics_skipped_when_timings_absent(fake_mnn, tmp_path):
    """context 只给 prompt_len/gen_seq_len/vision_us(无 prefill_us/decode_us)时,
    需 prefill/decode 的派生指标应缺省不产出,而非写 0 或崩溃(best-effort)。"""
    from eval_vlm.inference.mnn_backend import MNNBackend

    cfg = _mnn_cfg(tmp_path)
    backend = MNNBackend(cfg)          # _FakeModel.get_context 无 prefill_us/decode_us

    ctx = [Turn(role="user", content="<image>请描述图片")]
    pred = backend.complete(ctx, ["a.jpg"], "a.jpg")

    assert pred.error is None
    # vision_us=1234 存在 -> ttft 可算;prefill/decode 缺失 -> 相关指标不产出。
    assert pred.raw["ttft_ms"] == round(1234 / 1000.0, 3)
    assert "tpot_ms" not in pred.raw
    assert "decode_toks_per_s" not in pred.raw
    assert "prefill_toks_per_s" not in pred.raw


class _CfgModel:
    """记录 set_config 下发内容的假模型,用于验证采样配置翻译。"""

    def __init__(self):
        self.pushed: list = []

    def load(self):
        return True

    def reset(self):
        pass

    def response(self, prompt, stream=False):
        return "ok"

    def set_config(self, config):
        self.pushed.append(config)
        return True


def _sampler_sent(cfg, tmp_path):
    """在给定 cfg 下构造后端并触发采样配置下发,返回下发的 dict 列表。"""
    from eval_vlm.inference.mnn_backend import MNNBackend

    backend = MNNBackend(cfg)
    backend.model = _CfgModel()
    backend._apply_sampler_config()
    return backend.model.pushed


def test_mnn_default_sampler_is_penalty_plus_greedy(fake_mnn, tmp_path):
    """默认(repetition_penalty=1.1、temperature 未设):翻译成 mixed=[penalty, greedy]——
    重复惩罚打断 \\n 复读 + 确定性 argmax(可复现)。top_k/top_p/temperature 不下发。"""
    cfg = _mnn_cfg(tmp_path)
    pushed = _sampler_sent(cfg, tmp_path)

    assert len(pushed) == 1
    sent = pushed[0]
    assert sent["sampler_type"] == "mixed"
    assert sent["mixed_samplers"] == ["penalty", "greedy"]
    assert sent["repetition_penalty"] == 1.1
    assert sent["penalty_window"] == 0
    assert "temperature" not in sent and "top_k" not in sent and "top_p" not in sent


def test_mnn_value_gated_standard_sampling(fake_mnn, tmp_path):
    """设了 temperature>0 + top_k + top_p → 翻译成标准 topK+topP+temperature 采样,
    并自动带上 penalty(默认 1.1 仍开)。末步为 temperature(随机)。"""
    cfg = _mnn_cfg(tmp_path)
    cfg.inference.mnn.temperature = 0.7
    cfg.inference.mnn.top_k = 40
    cfg.inference.mnn.top_p = 0.9
    pushed = _sampler_sent(cfg, tmp_path)

    assert len(pushed) == 1
    sent = pushed[0]
    assert sent["sampler_type"] == "mixed"
    assert sent["mixed_samplers"] == ["penalty", "topK", "topP", "temperature"]
    assert sent["temperature"] == 0.7
    assert sent["top_k"] == 40 and sent["top_p"] == 0.9
    assert sent["repetition_penalty"] == 1.1


def test_mnn_penalty_off_yields_bare_greedy(fake_mnn, tmp_path):
    """全部惩罚关闭(repetition_penalty<=1、无 freq/presence)且无 temperature →
    mixed=[greedy],不下发任何 penalty 键(纯确定性 argmax)。"""
    cfg = _mnn_cfg(tmp_path)
    cfg.inference.mnn.repetition_penalty = 1.0
    pushed = _sampler_sent(cfg, tmp_path)

    assert len(pushed) == 1
    sent = pushed[0]
    assert sent["mixed_samplers"] == ["greedy"]
    assert "repetition_penalty" not in sent and "penalty" not in sent["mixed_samplers"]


def test_mnn_sampler_config_escape_hatch_verbatim(fake_mnn, tmp_path):
    """sampler_config 非空:原样下发 MNN 原生键,跳过 value-gated 翻译。"""
    cfg = _mnn_cfg(tmp_path)
    cfg.inference.mnn.sampler_config = {"sampler_type": "greedy"}
    pushed = _sampler_sent(cfg, tmp_path)

    assert pushed == [{"sampler_type": "greedy"}]


def test_mnn_sampler_config_empty_dict_disables(fake_mnn, tmp_path):
    """sampler_config={} 视作「不下发」→ 完全沿用模型 config.json 自带采样。"""
    cfg = _mnn_cfg(tmp_path)
    cfg.inference.mnn.sampler_config = {}
    pushed = _sampler_sent(cfg, tmp_path)

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
