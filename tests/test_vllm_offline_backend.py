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
    assert init["limit_mm_per_prompt"] == {"image": 1}
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


def test_complete_rejects_multi_image(fake_vllm, tmp_path):
    img = tmp_path / "a.jpg"
    img.write_bytes(b"x")
    backend = build_backend(_cfg("/models/m"))
    pred = backend.complete([Turn(role="user", content="<image>x")],
                            [str(img), str(img)], "s")
    assert pred.error and "单图" in pred.error


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
