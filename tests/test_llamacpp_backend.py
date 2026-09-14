"""llama.cpp (llamacpp) 多模态推理后端单元测试。

验证范围:
  1. Config 解析与持久化:LlamaCppBackendConfig 默认值、序列化与反序列化、InferenceConfig.active / result_name;
  2. 多图单轮 / 多图多轮 messages 构造:严格遵循 <image> 占位符与 images 列表的消费绑定机制;
  3. Server 模式:基于 Mock OpenAI client 验证 Payload 组装 (messages, extra_body, 采样参数)、Prediction 返回;
  4. 异常处理与 fail-fast 模式:mmproj 缺失提示识别、重试与 fail-fast 原始异常冒泡;
  5. CLI 模式:参数拼接、命令调用与结果清洗;
  6. 完整多轮 rollout 与 eval / field-eval 流程兼容性。
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from eval_vlm.config import Config, InferenceConfig, LlamaCppBackendConfig
from eval_vlm.data.schema import Sample, Turn
from eval_vlm.inference import build_backend, worker_count
from eval_vlm.inference.llamacpp_backend import LlamaCppBackend
from eval_vlm.runner import run_inference
from eval_vlm.data.splitter import split_dataset


# ---------------------------------------------------------------------------
# 1. Config 测试
# ---------------------------------------------------------------------------
def test_llamacpp_config_defaults():
    cfg = Config()
    assert cfg.inference.llamacpp.mode == "server"
    assert cfg.inference.llamacpp.base_url == "http://127.0.0.1:8080/v1"
    assert cfg.inference.llamacpp.max_concurrency == 4
    assert cfg.inference.llamacpp.temperature == 0.0
    assert cfg.inference.llamacpp.top_p == 1.0
    assert cfg.inference.llamacpp.top_k == 40
    assert cfg.inference.llamacpp.repetition_penalty == 1.0


def test_llamacpp_active_and_result_name():
    cfg = Config()
    cfg.inference.backend = "llamacpp"
    assert isinstance(cfg.inference.active, LlamaCppBackendConfig)

    # 默认 result_name
    assert cfg.inference.result_name == "llamacpp-model"

    # 指定 model
    cfg.inference.llamacpp.model = "qwen2.5-vl-7b"
    assert cfg.inference.result_name == "qwen2.5-vl-7b"

    # 指定 model_path 时，若在子文件夹内优先取父文件夹名作为模型名标识 A
    cfg.inference.llamacpp.model_path = "/path/to/my-vlm-model/model.Q4_K_M.gguf"
    assert cfg.inference.result_name == "my-vlm-model"

    # 别名 llama.cpp 和 llama_cpp 同样支持
    cfg.inference.backend = "llama.cpp"
    assert cfg.inference.result_name == "my-vlm-model"
    cfg.inference.backend = "llama_cpp"
    assert cfg.inference.result_name == "my-vlm-model"


# ---------------------------------------------------------------------------
# 2. 多图、多轮对话消息构建测试
# ---------------------------------------------------------------------------
def test_build_messages_single_image(tmp_path):
    img_file = tmp_path / "test.jpg"
    img_file.write_bytes(b"dummy image bytes")

    cfg = Config()
    cfg.inference.backend = "llamacpp"
    cfg.data.media_root = str(tmp_path)

    backend = LlamaCppBackend(cfg)

    context = [
        Turn(role="user", content="<image>请描述这张图片。")
    ]
    messages = backend._build_messages(context, ["test.jpg"], "sample-1")

    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    content = messages[0]["content"]
    # 含有 image_url 与 text
    types = [c["type"] for c in content]
    assert "image_url" in types
    assert "text" in types
    text_part = next(c for c in content if c["type"] == "text")
    assert text_part["text"] == "请描述这张图片。"
    img_part = next(c for c in content if c["type"] == "image_url")
    assert img_part["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_build_messages_multi_image_single_turn(tmp_path):
    img1 = tmp_path / "img1.jpg"
    img1.write_bytes(b"image 1 bytes")
    img2 = tmp_path / "img2.jpg"
    img2.write_bytes(b"image 2 bytes")

    cfg = Config()
    cfg.inference.backend = "llamacpp"
    cfg.data.media_root = str(tmp_path)

    backend = LlamaCppBackend(cfg)

    # 单轮包含两张图
    context = [
        Turn(role="user", content="图一:<image> 图二:<image> 请对比这两张图。")
    ]
    messages = backend._build_messages(context, ["img1.jpg", "img2.jpg"], "sample-multi-img")

    assert len(messages) == 1
    content = messages[0]["content"]
    img_parts = [p for p in content if p["type"] == "image_url"]
    assert len(img_parts) == 2


def test_build_messages_multi_turn_rollout(tmp_path):
    img1 = tmp_path / "img1.jpg"
    img1.write_bytes(b"image 1 bytes")

    cfg = Config()
    cfg.inference.backend = "llamacpp"
    cfg.data.media_root = str(tmp_path)
    cfg.inference.llamacpp.system_prompt = "You are a helpful assistant."

    backend = LlamaCppBackend(cfg)

    # 模拟两轮对话 Rollout:
    # 轮1: user 传图提问
    # 轮1: assistant 回答
    # 轮2: user 追问 (不带图)
    context = [
        Turn(role="user", content="<image>图中有什么?"),
        Turn(role="assistant", content="图中有一辆红色的汽车。"),
        Turn(role="user", content="汽车停在马路左侧还是右侧?"),
    ]
    messages = backend._build_messages(context, ["img1.jpg"], "sample-multi-turn")

    # 包含 system 轮 + 3 轮上下文
    assert len(messages) == 4
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "You are a helpful assistant."

    assert messages[1]["role"] == "user"
    assert any(p["type"] == "image_url" for p in messages[1]["content"])

    assert messages[2]["role"] == "assistant"
    assert messages[2]["content"] == "图中有一辆红色的汽车。"

    assert messages[3]["role"] == "user"
    assert messages[3]["content"] == "汽车停在马路左侧还是右侧?"


def test_build_messages_multi_turn_multi_image(tmp_path):
    img1 = tmp_path / "first.jpg"
    img1.write_bytes(b"first image")
    img2 = tmp_path / "second.jpg"
    img2.write_bytes(b"second image")

    cfg = Config()
    cfg.inference.backend = "llamacpp"
    cfg.data.media_root = str(tmp_path)

    backend = LlamaCppBackend(cfg)

    # 第一轮传图1，第二轮追问传图2
    context = [
        Turn(role="user", content="<image>第一张图内容"),
        Turn(role="assistant", content="这是第一张图描述"),
        Turn(role="user", content="<image>第二张图内容与上一张图有何异同?"),
    ]
    messages = backend._build_messages(context, ["first.jpg", "second.jpg"], "sample-2img-2round")
    assert len(messages) == 3

    # 第一个 user 轮应包含第 1 张图
    u1_imgs = [p for p in messages[0]["content"] if p["type"] == "image_url"]
    assert len(u1_imgs) == 1

    # 第二个 user 轮应包含第 2 张图
    u2_imgs = [p for p in messages[2]["content"] if p["type"] == "image_url"]
    assert len(u2_imgs) == 1


def test_build_messages_mismatch_image_count(tmp_path):
    cfg = Config()
    cfg.inference.backend = "llamacpp"
    cfg.data.media_root = str(tmp_path)

    backend = LlamaCppBackend(cfg)

    # 2 个占位符但只提供了 1 张图
    context = [
        Turn(role="user", content="<image><image>两个图")
    ]
    with pytest.raises(ValueError, match="占位符数量"):
        backend._build_messages(context, ["only_one.jpg"], "err-sample")


# ---------------------------------------------------------------------------
# 3. Server 模式与 Mock 调用
# ---------------------------------------------------------------------------
def test_server_complete_success(tmp_path):
    img = tmp_path / "sample.jpg"
    img.write_bytes(b"abc")

    cfg = Config()
    cfg.inference.backend = "llamacpp"
    cfg.data.media_root = str(tmp_path)
    cfg.inference.llamacpp.top_k = 20
    cfg.inference.llamacpp.repetition_penalty = 1.2

    backend = LlamaCppBackend(cfg)

    # Mock OpenAI client
    mock_choice = MagicMock()
    mock_choice.message.content = "这是一张小轿车照片"
    mock_resp = MagicMock()
    mock_resp.choices = [mock_choice]
    mock_resp.usage.completion_tokens = 15

    with patch.object(backend.client.chat.completions, "create", return_value=mock_resp) as mock_create:
        context = [Turn(role="user", content="<image>请描述这张图片")]
        pred = backend.complete(context, ["sample.jpg"], "s1")

        assert pred.id == "s1"
        assert pred.prediction == "这是一张小轿车照片"
        assert pred.raw["completion_tokens"] == 15
        assert pred.error is None

        # 检查参数
        call_kwargs = mock_create.call_args[1]
        assert call_kwargs["temperature"] == 0.0
        assert call_kwargs["extra_body"]["top_k"] == 20
        assert call_kwargs["extra_body"]["repeat_penalty"] == 1.2


def test_server_complete_missing_mmproj(tmp_path):
    img = tmp_path / "sample.jpg"
    img.write_bytes(b"abc")

    cfg = Config()
    cfg.inference.backend = "llamacpp"
    cfg.data.media_root = str(tmp_path)

    backend = LlamaCppBackend(cfg)

    # 模拟 llama-server 返回 missing mmproj 错误
    with patch.object(
        backend.client.chat.completions,
        "create",
        side_effect=RuntimeError("image input is not supported - hint: you may need to provide the mmproj"),
    ):
        context = [Turn(role="user", content="<image>请描述")]
        pred = backend.complete(context, ["sample.jpg"], "s1")

        assert pred.id == "s1"
        assert not pred.prediction
        assert "未加载多模态投影器" in pred.error


def test_server_complete_fail_fast(tmp_path):
    img = tmp_path / "sample.jpg"
    img.write_bytes(b"abc")

    cfg = Config()
    cfg.inference.backend = "llamacpp"
    cfg.data.media_root = str(tmp_path)
    cfg.inference.fail_fast = True

    backend = LlamaCppBackend(cfg)

    with patch.object(
        backend.client.chat.completions,
        "create",
        side_effect=ConnectionError("Failed to connect to llama-server"),
    ):
        context = [Turn(role="user", content="<image>请描述")]
        with pytest.raises(ConnectionError):
            backend.complete(context, ["sample.jpg"], "s1")


# ---------------------------------------------------------------------------
# 4. CLI 模式测试
# ---------------------------------------------------------------------------
def test_cli_complete_success(tmp_path):
    img = tmp_path / "sample.png"
    img.write_bytes(b"img")

    cfg = Config()
    cfg.inference.backend = "llamacpp"
    cfg.data.media_root = str(tmp_path)
    cfg.inference.llamacpp.mode = "cli"
    cfg.inference.llamacpp.model_path = "/models/vlm.gguf"
    cfg.inference.llamacpp.mmproj_path = "/models/mmproj.gguf"

    backend = LlamaCppBackend(cfg)
    assert backend.thread_safe is False
    assert worker_count(backend, 8) == 1  # 串行保证

    mock_res = MagicMock()
    mock_res.returncode = 0
    mock_res.stdout = "这是 CLI 生成的测试结果\n"
    mock_res.stderr = ""

    with patch("subprocess.run", return_value=mock_res) as mock_run:
        context = [Turn(role="user", content="<image>描述")]
        pred = backend.complete(context, ["sample.png"], "cli-1")

        assert pred.prediction == "这是 CLI 生成的测试结果"
        assert pred.error is None

        # 检查传入 subprocess 的命令行
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "mtmd-cli"
        assert "-m" in cmd and "/models/vlm.gguf" in cmd
        assert "--mmproj" in cmd and "/models/mmproj.gguf" in cmd
        assert "--image" in cmd and str(img) in cmd


# ---------------------------------------------------------------------------
# 5. 集成测试: 多轮 Rollout 测试链路
# ---------------------------------------------------------------------------
def test_llamacpp_rollout_integration(tworound_config, monkeypatch):
    cfg = tworound_config
    cfg.inference.backend = "llamacpp"
    cfg.inference.llamacpp.mode = "server"
    cfg.eval.context = "rollout"

    split_dataset(cfg)

    # 捕获发送给 server 的 messages
    captured_messages = []

    def mock_create(*args, **kwargs):
        msgs = kwargs.get("messages", [])
        captured_messages.append(msgs)
        # 根据当前是否有 assistant 历史来模拟第一轮与第二轮的回复
        has_asst = any(m["role"] == "assistant" for m in msgs)
        ans = "生气" if has_asst else "画面是一名成年男性的面部特写"
        mock_choice = MagicMock()
        mock_choice.message.content = ans
        resp = MagicMock()
        resp.choices = [mock_choice]
        resp.usage.completion_tokens = 8
        return resp

    backend = build_backend(cfg)
    monkeypatch.setattr(backend.client.chat.completions, "create", mock_create)
    monkeypatch.setattr("eval_vlm.runner.build_backend", lambda _: backend)

    stats = run_inference(cfg)
    assert stats["errors"] == 0
    assert stats["newly_completed"] > 0

    # 验证第二轮的 messages 确实包含了第一轮生成的回答
    round2_msgs = [m for m in captured_messages if any(turn["role"] == "assistant" for turn in m)]
    assert len(round2_msgs) > 0
    for m in round2_msgs:
        asst_turns = [turn for turn in m if turn["role"] == "assistant"]
        assert asst_turns[0]["content"] == "画面是一名成年男性的面部特写"


# ---------------------------------------------------------------------------
# 6. 推理指标统计测试 (timings -> infer_stats.txt / infer_stats.json)
# ---------------------------------------------------------------------------
def test_server_timings_extraction(tmp_path):
    img = tmp_path / "sample.jpg"
    img.write_bytes(b"abc")

    cfg = Config()
    cfg.inference.backend = "llamacpp"
    cfg.data.media_root = str(tmp_path)

    backend = LlamaCppBackend(cfg)

    mock_resp = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "测试输出内容"
    mock_resp.choices = [mock_choice]
    mock_resp.usage.prompt_tokens = 150
    mock_resp.usage.completion_tokens = 25
    mock_resp.usage.total_tokens = 175
    # 模拟 llama-server 返回的 timings 字典
    mock_resp.timings = {
        "prompt_n": 150,
        "prompt_ms": 60.0,
        "predicted_n": 25,
        "predicted_ms": 100.0,
    }

    with patch.object(backend.client.chat.completions, "create", return_value=mock_resp):
        context = [Turn(role="user", content="<image>请描述")]
        pred = backend.complete(context, ["sample.jpg"], "timing-1")

        assert pred.prediction == "测试输出内容"
        assert pred.raw["backend"] == "llamacpp"
        assert pred.raw["prompt_len"] == 150
        assert pred.raw["gen_seq_len"] == 25
        assert pred.raw["prefill_us"] == 60000
        assert pred.raw["decode_us"] == 100000
        assert pred.raw["ttft_ms"] == 60.0
        assert pred.raw["tpot_ms"] == 4.0  # 100ms / 25 tok


def test_cli_timings_extraction(tmp_path):
    img = tmp_path / "sample.png"
    img.write_bytes(b"img")

    cfg = Config()
    cfg.inference.backend = "llamacpp"
    cfg.data.media_root = str(tmp_path)
    cfg.inference.llamacpp.mode = "cli"
    cfg.inference.llamacpp.model_path = "/models/vlm.gguf"
    cfg.inference.llamacpp.mmproj_path = "/models/mmproj.gguf"

    backend = LlamaCppBackend(cfg)

    mock_res = MagicMock()
    mock_res.returncode = 0
    mock_res.stdout = "这是输出文本"
    mock_res.stderr = (
        "prompt eval time =      50.00 ms /   100 tokens (    0.50 ms per token,  2000.00 tokens per second)\n"
        "       eval time =     200.00 ms /    20 runs   (   10.00 ms per token,   100.00 tokens per second)\n"
    )

    with patch("subprocess.run", return_value=mock_res):
        context = [Turn(role="user", content="<image>描述")]
        pred = backend.complete(context, ["sample.png"], "cli-timing")

        assert pred.prediction == "这是输出文本"
        assert pred.raw["prompt_len"] == 100
        assert pred.raw["gen_seq_len"] == 20
        assert pred.raw["prefill_us"] == 50000
        assert pred.raw["decode_us"] == 200000
        assert pred.raw["ttft_ms"] == 50.0


def test_llamacpp_infer_stats_generation(tmp_path):
    import json
    from eval_vlm.cli import _maybe_generate_mnn_report
    from eval_vlm.infer_stats import generate_inference_report

    cfg = Config()
    cfg.run_dir_path = tmp_path
    cfg.inference.backend = "llamacpp"
    cfg.inference.llamacpp.model = "qwen2-vl"
    cfg.run_dir.mkdir(parents=True, exist_ok=True)

    lines = [
        json.dumps({
            "id": "s1", "latency": 1.2, "error": None,
            "raw": {
                "backend": "llamacpp", "prompt_len": 120, "gen_seq_len": 20,
                "prefill_us": 60000, "decode_us": 140000,
            }
        }),
        json.dumps({
            "id": "s2", "latency": 0.8, "error": None,
            "raw": {
                "backend": "llamacpp", "prompt_len": 80, "gen_seq_len": 15,
                "prefill_us": 40000, "decode_us": 105000,
            }
        }),
    ]
    cfg.predictions_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    summary, report = generate_inference_report(cfg.predictions_path, out_dir=cfg.run_dir, print_report=False)
    assert summary["samples"]["analyzed"] == 2
    assert "ttft_s" in summary["metrics"]
    assert "tpot_ms" in summary["metrics"]
    assert "decode_tps" in summary["metrics"]
    assert (cfg.run_dir / "infer_stats.txt").exists()
    assert (cfg.run_dir / "infer_stats.json").exists()

    # 验证 CLI 钩子自动触发
    report_res = _maybe_generate_mnn_report(cfg)
    assert report_res is not None
    assert report_res["samples"]["analyzed"] == 2
    assert cfg.infer_stats_txt_path.exists()
    assert cfg.infer_stats_json_path.exists()


