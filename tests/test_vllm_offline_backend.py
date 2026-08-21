"""离线 vLLM 后端(vllm_offline):不装 vllm 时用假 vllm 模块验证 config/dispatch/推理路径。

全程不真的加载模型:注入假 `vllm` 模块(LLM/SamplingParams),断言构建参数、消息组装、
complete 产出。仿 test_mnn_backend 的「monkeypatch 原生依赖」思路。
"""
from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import pytest

from eval_vlm.config import Config
from eval_vlm.data.schema import Turn
from eval_vlm.inference import build_backend


# ---------------------------------------------------------------------------
# 假 vllm 模块
# ---------------------------------------------------------------------------
class _FakeSamplingParams:
    def __init__(self, **kw):
        self.kw = kw


class _FakeGen:
    def __init__(self, text):
        self.text = text
        self.token_ids = [1, 2, 3]


class _FakeOut:
    def __init__(self, text):
        self.outputs = [_FakeGen(text)]
        self.prompt_token_ids = [0] * 7


class _FakeLLM:
    last_init = None
    last_chat = None

    def __init__(self, **kw):
        _FakeLLM.last_init = kw

    def chat(self, messages, sampling, use_tqdm=False):
        _FakeLLM.last_chat = messages
        # 批处理:messages 是"对话列表"(首元素是 list)-> 每条对话一个输出;
        # 单条:messages 是一条对话(首元素是 dict)-> 一个输出。
        if messages and isinstance(messages[0], list):
            return [_FakeOut(f"pred_{i}") for i in range(len(messages))]
        return [_FakeOut("主辅路：主路")]


@pytest.fixture
def fake_vllm(monkeypatch):
    mod = types.ModuleType("vllm")
    mod.LLM = _FakeLLM
    mod.SamplingParams = _FakeSamplingParams
    monkeypatch.setitem(sys.modules, "vllm", mod)
    _FakeLLM.last_init = None
    _FakeLLM.last_chat = None
    return mod


def _cfg(model_path: str) -> Config:
    cfg = Config()
    cfg.inference.backend = "vllm_offline"
    cfg.inference.vllm_offline.model_path = model_path
    return cfg


# ---------------------------------------------------------------------------
# config / dispatch
# ---------------------------------------------------------------------------
def test_result_name_and_active():
    cfg = _cfg("/models/trial_3_merged")
    assert cfg.inference.active is cfg.inference.vllm_offline
    assert cfg.inference.result_name == "trial_3_merged"
    # run_dir 三级:dataset / result_name / backend
    assert cfg.inference.result_name == "trial_3_merged"
    assert cfg.inference.backend == "vllm_offline"


def test_result_name_fallback_when_no_path():
    cfg = Config()
    cfg.inference.backend = "vllm_offline"
    assert cfg.inference.result_name == "vllm-offline-model"


def test_config_yaml_builds_vllm_offline_dataclass(tmp_path):
    """config.yaml 里的 inference.vllm_offline 块要被 _build 构造成 dataclass(而非留成 dict)。"""
    import yaml
    from eval_vlm.config import load_config
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump({"inference": {"backend": "vllm_offline", "vllm_offline": {
        "model_path": "/m/merged", "gpu_memory_utilization": 0.8, "max_model_len": 8192,
        "image_max_pixels": 564480, "max_tokens": 256, "vllm_kwargs": {"enforce_eager": True}}}}),
        encoding="utf-8")
    cfg = load_config(str(p))
    vo = cfg.inference.vllm_offline
    from eval_vlm.config import VLLMOfflineBackendConfig
    assert isinstance(vo, VLLMOfflineBackendConfig)
    assert vo.model_path == "/m/merged" and vo.max_model_len == 8192
    assert vo.vllm_kwargs == {"enforce_eager": True}
    assert vo.image_min_pixels == 784           # 缺省项回落默认
    assert cfg.inference.result_name == "merged"


def test_missing_model_path_raises(fake_vllm):
    cfg = Config()
    cfg.inference.backend = "vllm_offline"          # 无 model_path
    with pytest.raises(ValueError):
        build_backend(cfg)


def test_build_backend_dispatches_vllm_offline(fake_vllm):
    backend = build_backend(_cfg("/models/m"))
    from eval_vlm.inference.vllm_offline_backend import VLLMOfflineBackend
    assert isinstance(backend, VLLMOfflineBackend)
    assert backend.thread_safe is False
    # LLM 构建参数透传正确
    init = _FakeLLM.last_init
    assert init["model"] == "/models/m"
    assert init["limit_mm_per_prompt"] == {"image": 4}
    assert init["mm_processor_kwargs"]["max_pixels"] == cfg_max_pixels()


def cfg_max_pixels() -> int:
    return Config().inference.vllm_offline.image_max_pixels


def test_engine_kwargs_from_config(fake_vllm):
    cfg = _cfg("/models/m")
    cfg.inference.vllm_offline.gpu_memory_utilization = 0.75
    cfg.inference.vllm_offline.max_model_len = 8192
    cfg.inference.vllm_offline.image_max_pixels = 111111
    cfg.inference.vllm_offline.vllm_kwargs = {"enforce_eager": True}
    build_backend(cfg)
    init = _FakeLLM.last_init
    assert init["gpu_memory_utilization"] == 0.75
    assert init["max_model_len"] == 8192
    assert init["mm_processor_kwargs"]["max_pixels"] == 111111
    assert init["enforce_eager"] is True            # vllm_kwargs 逃生口合并


def test_env_applied_before_llm_build(fake_vllm, monkeypatch):
    """vllm_offline.env 在建 LLM 前写入 os.environ(用于禁用/绕过 flashinfer 等)。"""
    for k in ("FLASHINFER_DISABLE_VERSION_CHECK", "VLLM_ATTENTION_BACKEND"):
        monkeypatch.delenv(k, raising=False)   # 注册清理,测试结束移除
    cfg = _cfg("/models/m")
    cfg.inference.vllm_offline.env = {"FLASHINFER_DISABLE_VERSION_CHECK": "1",
                                      "VLLM_ATTENTION_BACKEND": "FLASH_ATTN"}
    build_backend(cfg)
    assert os.environ["FLASHINFER_DISABLE_VERSION_CHECK"] == "1"
    assert os.environ["VLLM_ATTENTION_BACKEND"] == "FLASH_ATTN"


# ---------------------------------------------------------------------------
# complete
# ---------------------------------------------------------------------------
def test_complete_builds_messages_and_returns_text(fake_vllm, tmp_path):
    img = tmp_path / "a.jpg"
    img.write_bytes(b"\xff\xd8\xff\xe0fakejpeg")
    backend = build_backend(_cfg("/models/m"))

    context = [Turn(role="user", content="<image>请描述这张图片。")]
    pred = backend.complete(context, [str(img)], "sample-1")

    assert pred.error is None
    assert pred.prediction == "主辅路：主路"
    assert pred.raw["backend"] == "vllm_offline"
    assert pred.raw["prompt_len"] == 7
    # 消息:user 轮拆成 image_url(data URI)+ text
    msg = _FakeLLM.last_chat
    content = msg[0]["content"]
    assert content[0]["type"] == "image_url"
    assert content[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert content[1] == {"type": "text", "text": "请描述这张图片。"}


def test_complete_system_prompt_prepended(fake_vllm, tmp_path):
    img = tmp_path / "a.png"
    img.write_bytes(b"\x89PNGfake")
    cfg = _cfg("/models/m")
    cfg.inference.vllm_offline.system_prompt = "你是助手"
    backend = build_backend(cfg)
    backend.complete([Turn(role="user", content="<image>描述")], [str(img)], "s")
    msg = _FakeLLM.last_chat
    assert msg[0] == {"role": "system", "content": [{"type": "text", "text": "你是助手"}]}
    assert msg[1]["content"][0]["image_url"]["url"].startswith("data:image/png;base64,")


def test_complete_multi_image_takes_needed(fake_vllm, tmp_path):
    """多图样本 + 单 <image> context → 只取第 1 张图,不报错。"""
    img1 = tmp_path / "a.jpg"
    img1.write_bytes(b"\xff\xd8\xff\xe0fakejpeg")
    img2 = tmp_path / "b.jpg"
    img2.write_bytes(b"\xff\xd8\xff\xe0fakejpeg2")
    backend = build_backend(_cfg("/models/m"))
    pred = backend.complete([Turn(role="user", content="<image>x")],
                            [str(img1), str(img2)], "s")
    assert pred.error is None
    assert pred.prediction == "主辅路：主路"
    # 消息中只有 1 个 image_url 块(只消费了 1 张图)
    msg = _FakeLLM.last_chat
    img_parts = [p for m in msg for p in m["content"] if isinstance(p, dict) and p.get("type") == "image_url"]
    assert len(img_parts) == 1


def test_complete_missing_image_records_error(fake_vllm, tmp_path):
    backend = build_backend(_cfg("/models/m"))
    pred = backend.complete([Turn(role="user", content="<image>x")],
                            [str(tmp_path / "nope.jpg")], "s")
    assert pred.error and "build_prompt" in pred.error


def test_complete_fail_fast_reraises(fake_vllm, tmp_path):
    backend = build_backend(_cfg("/models/m"))
    backend.cfg.inference.fail_fast = True
    with pytest.raises(FileNotFoundError):
        backend.complete([Turn(role="user", content="<image>x")],
                         [str(tmp_path / "nope.jpg")], "s")


# ---------------------------------------------------------------------------
# complete_batch:真·批处理
# ---------------------------------------------------------------------------
def test_supports_batch_flag(fake_vllm):
    backend = build_backend(_cfg("/models/m"))
    assert backend.supports_batch is True


def test_complete_batch_one_chat_call_ordered(fake_vllm, tmp_path):
    """整批一次 llm.chat(多对话);返回与 items 等长同序;成功条有 raw。"""
    from eval_vlm.inference.base import BatchItem
    imgs = []
    for n in ("a.jpg", "b.jpg", "c.jpg"):
        p = tmp_path / n
        p.write_bytes(b"\xff\xd8\xffdata")
        imgs.append(str(p))
    backend = build_backend(_cfg("/models/m"))
    items = [BatchItem([Turn("user", "<image>描述")], [im], f"s{i}") for i, im in enumerate(imgs)]
    preds = backend.complete_batch(items)

    assert [p.id for p in preds] == ["s0", "s1", "s2"]
    assert [p.prediction for p in preds] == ["pred_0", "pred_1", "pred_2"]
    assert all(p.error is None and p.raw["backend"] == "vllm_offline" for p in preds)
    # 一次 chat,传入的是"对话列表"(批处理)
    assert isinstance(_FakeLLM.last_chat[0], list)
    assert len(_FakeLLM.last_chat) == 3


def test_complete_batch_build_error_isolated(fake_vllm, tmp_path):
    """单条构造失败(缺图)记 error 且不进批,其余仍正常;顺序/长度不变。"""
    from eval_vlm.inference.base import BatchItem
    ok = tmp_path / "ok.jpg"
    ok.write_bytes(b"x")
    items = [
        BatchItem([Turn("user", "<image>x")], [str(tmp_path / "missing.jpg")], "bad"),
        BatchItem([Turn("user", "<image>x")], [str(ok)], "good"),
    ]
    backend = build_backend(_cfg("/models/m"))
    preds = backend.complete_batch(items)
    assert preds[0].id == "bad" and preds[0].error and "build_prompt" in preds[0].error
    assert preds[1].id == "good" and preds[1].error is None
    # 只有 1 条进了批(good)
    assert len(_FakeLLM.last_chat) == 1


def test_complete_multi_turn_multi_image(fake_vllm, tmp_path):
    """多轮多图:3 张图 / 3 个 <image>,逐轮 rollout 时 context 逐渐变长。"""
    imgs = []
    for n in ("1.jpg", "2.jpg", "3.jpg"):
        p = tmp_path / n
        p.write_bytes(b"\xff\xd8\xff\xe0fake")
        imgs.append(str(p))
    backend = build_backend(_cfg("/models/m"))

    # 第 1 轮: context 含 1 个 <image> → 取 imgs[0]
    ctx1 = [Turn("user", "<image>描述")]
    pred1 = backend.complete(ctx1, imgs, "s1")
    assert pred1.error is None
    msg1 = _FakeLLM.last_chat
    img_parts1 = [p for m in msg1 for p in m["content"] if isinstance(p, dict) and p.get("type") == "image_url"]
    assert len(img_parts1) == 1

    # 第 2 轮: context 含 2 个 <image> → 取 imgs[0:2]
    ctx2 = [
        Turn("user", "<image>描述"),
        Turn("assistant", "雪景"),
        Turn("user", "<image>天气"),
    ]
    pred2 = backend.complete(ctx2, imgs, "s2")
    assert pred2.error is None
    msg2 = _FakeLLM.last_chat
    img_parts2 = [p for m in msg2 for p in m["content"] if isinstance(p, dict) and p.get("type") == "image_url"]
    assert len(img_parts2) == 2

    # 第 3 轮: context 含 3 个 <image> → 取全部
    ctx3 = [
        Turn("user", "<image>描述"),
        Turn("assistant", "雪景"),
        Turn("user", "<image>天气"),
        Turn("assistant", "下雪"),
        Turn("user", "<image>查看天气"),
    ]
    pred3 = backend.complete(ctx3, imgs, "s3")
    assert pred3.error is None
    msg3 = _FakeLLM.last_chat
    img_parts3 = [p for m in msg3 for p in m["content"] if isinstance(p, dict) and p.get("type") == "image_url"]
    assert len(img_parts3) == 3


def test_complete_placeholder_exceeds_images_errors(fake_vllm, tmp_path):
    """<image> 占位符数 > images 数 → 报错。"""
    img = tmp_path / "a.jpg"
    img.write_bytes(b"x")
    backend = build_backend(_cfg("/models/m"))
    ctx = [Turn("user", "<image>x"), Turn("assistant", "y"), Turn("user", "<image>z")]
    pred = backend.complete(ctx, [str(img)], "s")
    assert pred.error and "占位符" in pred.error


def test_complete_batch_multi_image(fake_vllm, tmp_path):
    """批处理路径也能正确处理多图多轮。"""
    from eval_vlm.inference.base import BatchItem
    imgs = []
    for n in ("a.jpg", "b.jpg"):
        p = tmp_path / n
        p.write_bytes(b"\xff\xd8\xffdata")
        imgs.append(str(p))
    backend = build_backend(_cfg("/models/m"))
    # 2 个样本:每个样本 2 张图 / 2 个 <image>
    ctx = [Turn("user", "<image>描述"), Turn("assistant", "x"), Turn("user", "<image>天气")]
    items = [
        BatchItem(ctx, imgs, "s0"),
        BatchItem(ctx, imgs, "s1"),
    ]
    preds = backend.complete_batch(items)
    assert len(preds) == 2
    assert all(p.error is None for p in preds)


def test_max_images_per_prompt_config(fake_vllm):
    """max_images_per_prompt 配置透传到 limit_mm_per_prompt。"""
    cfg = _cfg("/models/m")
    cfg.inference.vllm_offline.max_images_per_prompt = 8
    build_backend(cfg)
    assert _FakeLLM.last_init["limit_mm_per_prompt"] == {"image": 8}
