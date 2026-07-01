"""cmnn(C++ 原生库批量)后端:用注入的假 cmnn_native 扩展验证批量分发、prompt 构造、
图片净化、采样/上限下发、退化检测与错误隔离,不依赖真实 MNN/已编译扩展(CI 也能跑)。

与 test_mnn_backend.py 的套路一致:把假的 cmnn_native 注入 sys.modules。区别是 cmnn 的
Python 层不依赖 pymnn(MNN.*),图片净化只用 Pillow,故测试里用真实小图片文件。
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest
from PIL import Image

from eval_vlm.config import Config
from eval_vlm.data.schema import Turn
from eval_vlm.inference import build_backend
from eval_vlm.inference.base import BatchItem


# ---------------------------------------------------------------------------
# 假 cmnn_native 扩展:记录调用,generate_batch 回显可配置的输出
# ---------------------------------------------------------------------------
class _FakeEngine:
    # 类级开关:让个别用例定制 generate_batch 的输出而不改构造流程。
    def __init__(self, config_path: str, num_workers: int):
        self.config_path = config_path
        self.num_workers = num_workers
        self.set_config_args: list[str] = []
        self.last_requests: list[dict] | None = None
        self.closed = False
        # 默认输出工厂:i -> 一条结果 dict(可被用例替换)。
        self.output_for = lambda i, req: {
            "text": f"描述{i}",
            "prompt_len": 7,
            "gen_seq_len": 5,
            "vision_us": 100,
            "pixels_mp": 0.1,
            "latency": 0.01,
        }

    def set_config(self, content: str):
        self.set_config_args.append(content)
        return True

    def generate_batch(self, requests):
        self.last_requests = list(requests)
        return [self.output_for(i, req) for i, req in enumerate(requests)]

    def close(self):
        self.closed = True


@pytest.fixture
def fake_cmnn(monkeypatch):
    """注入假的 cmnn_native;返回被创建的 Engine 列表(用例可断言其记录)。"""
    created: list[_FakeEngine] = []

    mod = types.ModuleType("cmnn_native")

    def _engine(config_path, num_workers):
        e = _FakeEngine(config_path, num_workers)
        created.append(e)
        return e

    mod.Engine = _engine
    monkeypatch.setitem(sys.modules, "cmnn_native", mod)
    return created


def _write_img(path: Path, size=(32, 32), fmt="JPEG", color=(90, 120, 150)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path, format=fmt)


def _cmnn_cfg(tmp_path, *, num_workers=4, batch_size=16, max_tokens=128) -> Config:
    imgs = tmp_path / "imgs"
    imgs.mkdir()
    _write_img(imgs / "a.jpg")
    _write_img(imgs / "b.jpg", color=(10, 20, 30))
    cfg = Config()
    cfg.inference.backend = "cmnn"
    cfg.inference.cmnn.config_path = str(tmp_path / "model" / "config.json")
    cfg.inference.cmnn.num_workers = num_workers
    cfg.inference.cmnn.batch_size = batch_size
    cfg.inference.cmnn.max_tokens = max_tokens
    cfg.data.media_root = str(imgs)
    return cfg


def _ctx() -> list[Turn]:
    return [Turn(role="user", content="<image>请描述图片")]


# ---------------------------------------------------------------------------
# 基本能力:声明 / 分发 / 构造
# ---------------------------------------------------------------------------
def test_cmnn_declares_batch_and_not_thread_safe(fake_cmnn, tmp_path):
    from eval_vlm.inference.cmnn_backend import CMNNBackend

    backend = CMNNBackend(_cmnn_cfg(tmp_path))
    assert backend.supports_batch is True
    assert backend.thread_safe is False
    # Engine 以 config_path + num_workers 构造。
    assert fake_cmnn[0].num_workers == 4
    assert fake_cmnn[0].config_path.endswith("config.json")


def test_build_backend_dispatches_cmnn(fake_cmnn, tmp_path):
    from eval_vlm.inference.cmnn_backend import CMNNBackend

    assert isinstance(build_backend(_cmnn_cfg(tmp_path)), CMNNBackend)


def test_cmnn_batch_builds_requests_and_maps_outputs(fake_cmnn, tmp_path):
    from eval_vlm.inference.cmnn_backend import CMNNBackend

    backend = CMNNBackend(_cmnn_cfg(tmp_path))
    items = [
        BatchItem(context=_ctx(), images=["a.jpg"], sample_id="a.jpg"),
        BatchItem(context=_ctx(), images=["b.jpg"], sample_id="b.jpg"),
    ]
    preds = backend.complete_batch(items)

    assert len(preds) == 2
    assert [p.id for p in preds] == ["a.jpg", "b.jpg"]      # 等长同序
    assert preds[0].error is None and preds[0].prediction == "描述0"
    assert preds[1].prediction == "描述1"
    # 请求内容:<image> -> <img>image_0</img>;jpg 在尺寸内 -> 原生路径直传;带 max_new_tokens。
    reqs = fake_cmnn[0].last_requests
    assert reqs[0]["text"] == "<img>image_0</img>请描述图片"
    assert reqs[0]["image_path"].replace("\\", "/").endswith("imgs/a.jpg")
    assert reqs[0]["max_new_tokens"] == 128
    # 统计落到 raw(backend 标记 + 透传的原生统计)。
    assert preds[0].raw["backend"] == "cmnn" and preds[0].raw["prompt_len"] == 7


def test_cmnn_complete_single_delegates_to_batch(fake_cmnn, tmp_path):
    from eval_vlm.inference.cmnn_backend import CMNNBackend

    backend = CMNNBackend(_cmnn_cfg(tmp_path))
    pred = backend.complete(_ctx(), ["a.jpg"], "a.jpg")
    assert pred.error is None and pred.prediction == "描述0"


# ---------------------------------------------------------------------------
# 配置下发:采样 + max_new_tokens
# ---------------------------------------------------------------------------
def test_cmnn_pushes_sampler_and_max_tokens_at_init(fake_cmnn, tmp_path):
    from eval_vlm.inference.cmnn_backend import CMNNBackend

    CMNNBackend(_cmnn_cfg(tmp_path))              # 默认 repetition_penalty=1.1
    sent = fake_cmnn[0].set_config_args
    assert len(sent) == 1
    cfg = json.loads(sent[0])
    # 默认:mixed=[penalty, greedy] + repetition_penalty + max_new_tokens 合并下发。
    assert cfg["sampler_type"] == "mixed"
    assert cfg["mixed_samplers"] == ["penalty", "greedy"]
    assert cfg["repetition_penalty"] == 1.1
    assert cfg["max_new_tokens"] == 128


def test_cmnn_value_gated_sampling_matches_mnn(fake_cmnn, tmp_path):
    """cmnn 与 mnn 共用同一套 value-gated 翻译:temperature+top_k+top_p 生效。"""
    from eval_vlm.inference.cmnn_backend import CMNNBackend

    cfg = _cmnn_cfg(tmp_path)
    cfg.inference.cmnn.temperature = 0.7
    cfg.inference.cmnn.top_k = 40
    cfg.inference.cmnn.top_p = 0.9
    CMNNBackend(cfg)
    sent = json.loads(fake_cmnn[0].set_config_args[0])
    assert sent["mixed_samplers"] == ["penalty", "topK", "topP", "temperature"]
    assert sent["temperature"] == 0.7 and sent["top_k"] == 40 and sent["top_p"] == 0.9


# ---------------------------------------------------------------------------
# 图片净化:webp 转码 / 超大图缩放
# ---------------------------------------------------------------------------
def test_cmnn_webp_transcodes_to_temp_png(fake_cmnn, tmp_path):
    from eval_vlm.inference.cmnn_backend import CMNNBackend

    cfg = _cmnn_cfg(tmp_path)
    webp = Path(cfg.data.media_root) / "城市.webp"
    try:
        Image.new("RGB", (16, 16), (1, 2, 3)).save(webp, format="WEBP")
    except Exception as e:  # noqa: BLE001 - 无 webp 支持则跳过
        pytest.skip(f"Pillow 无 WEBP 支持: {e}")

    backend = CMNNBackend(cfg)
    preds = backend.complete_batch(
        [BatchItem(context=_ctx(), images=["城市.webp"], sample_id="城市.webp")]
    )
    assert preds[0].error is None
    req_path = fake_cmnn[0].last_requests[0]["image_path"]
    # 非原生格式走转码:传给 C++ 的是临时 .png,且用完即删。
    assert req_path.lower().endswith(".png")
    assert not Path(req_path).exists()


def test_cmnn_oversized_image_downscaled_to_temp(fake_cmnn, tmp_path):
    from eval_vlm.inference.cmnn_backend import CMNNBackend

    cfg = _cmnn_cfg(tmp_path)
    cfg.inference.cmnn.image_max_side = 64
    big = Path(cfg.data.media_root) / "big.jpg"
    _write_img(big, size=(200, 120))
    backend = CMNNBackend(cfg)
    preds = backend.complete_batch(
        [BatchItem(context=_ctx(), images=["big.jpg"], sample_id="big.jpg")]
    )
    assert preds[0].error is None
    req_path = fake_cmnn[0].last_requests[0]["image_path"]
    assert req_path.lower().endswith(".png")     # 缩放走临时 png
    assert not Path(req_path).exists()            # 用完即删


# ---------------------------------------------------------------------------
# 错误隔离与退化
# ---------------------------------------------------------------------------
def test_cmnn_build_error_isolated_within_batch(fake_cmnn, tmp_path):
    """一条缺图不影响同批其它条:缺图记 build_prompt error,好图仍成功。"""
    from eval_vlm.inference.cmnn_backend import CMNNBackend

    backend = CMNNBackend(_cmnn_cfg(tmp_path))
    items = [
        BatchItem(context=_ctx(), images=["missing.jpg"], sample_id="missing.jpg"),
        BatchItem(context=_ctx(), images=["a.jpg"], sample_id="a.jpg"),
    ]
    preds = backend.complete_batch(items)
    assert preds[0].error is not None and "build_prompt" in preds[0].error
    assert preds[0].prediction == ""
    assert preds[1].error is None and preds[1].prediction == "描述0"   # 好图是本次 requests 第 0 条
    # 只有 1 条有效请求送到原生库(缺图不入 requests)。
    assert len(fake_cmnn[0].last_requests) == 1
    assert fake_cmnn[0].last_requests[0]["image_path"].replace("\\", "/").endswith("imgs/a.jpg")


def test_cmnn_multi_image_records_error(fake_cmnn, tmp_path):
    from eval_vlm.inference.cmnn_backend import CMNNBackend

    backend = CMNNBackend(_cmnn_cfg(tmp_path))
    preds = backend.complete_batch(
        [BatchItem(context=_ctx(), images=["a.jpg", "b.jpg"], sample_id="x")]
    )
    assert preds[0].error is not None and "单图" in preds[0].error


def test_cmnn_degenerate_output_recorded_as_error(fake_cmnn, tmp_path):
    from eval_vlm.inference.cmnn_backend import CMNNBackend

    backend = CMNNBackend(_cmnn_cfg(tmp_path))
    fake_cmnn[0].output_for = lambda i, req: {"text": "\n" * 200, "gen_seq_len": 200}
    preds = backend.complete_batch(
        [BatchItem(context=_ctx(), images=["a.jpg"], sample_id="a.jpg")]
    )
    assert preds[0].prediction == ""
    assert preds[0].error is not None and "degenerate_output" in preds[0].error


def test_cmnn_native_error_propagated_per_item(fake_cmnn, tmp_path):
    """原生层单条 error 透传为该条失败(不影响批内其它条的契约)。"""
    from eval_vlm.inference.cmnn_backend import CMNNBackend

    backend = CMNNBackend(_cmnn_cfg(tmp_path))
    fake_cmnn[0].output_for = lambda i, req: {"error": "OOM in vision encoder"}
    preds = backend.complete_batch(
        [BatchItem(context=_ctx(), images=["a.jpg"], sample_id="a.jpg")]
    )
    assert preds[0].error is not None and "OOM" in preds[0].error


def test_cmnn_generate_batch_raises_marks_all_error(fake_cmnn, tmp_path):
    """整批原生调用抛异常:所有有效请求条记为 error,不抛穿、不丢条。"""
    from eval_vlm.inference.cmnn_backend import CMNNBackend

    backend = CMNNBackend(_cmnn_cfg(tmp_path))

    def _boom(requests):
        raise RuntimeError("native crash")

    fake_cmnn[0].generate_batch = _boom
    preds = backend.complete_batch([
        BatchItem(context=_ctx(), images=["a.jpg"], sample_id="a.jpg"),
        BatchItem(context=_ctx(), images=["b.jpg"], sample_id="b.jpg"),
    ])
    assert len(preds) == 2
    assert all(p.error is not None and "native crash" in p.error for p in preds)


def test_cmnn_count_mismatch_marks_all_error(fake_cmnn, tmp_path):
    """原生库返回条数与请求不符:整批判失败,避免错位回填。"""
    from eval_vlm.inference.cmnn_backend import CMNNBackend

    backend = CMNNBackend(_cmnn_cfg(tmp_path))
    fake_cmnn[0].generate_batch = lambda requests: [{"text": "只回一条"}]
    preds = backend.complete_batch([
        BatchItem(context=_ctx(), images=["a.jpg"], sample_id="a.jpg"),
        BatchItem(context=_ctx(), images=["b.jpg"], sample_id="b.jpg"),
    ])
    assert len(preds) == 2
    assert all(p.error is not None and "返回条数" in p.error for p in preds)


# ---------------------------------------------------------------------------
# 构造期校验
# ---------------------------------------------------------------------------
def test_cmnn_missing_config_path_raises(fake_cmnn, tmp_path):
    from eval_vlm.inference.cmnn_backend import CMNNBackend

    cfg = _cmnn_cfg(tmp_path)
    cfg.inference.cmnn.config_path = None
    with pytest.raises(ValueError) as e:
        CMNNBackend(cfg)
    assert "config_path" in str(e.value)


def test_cmnn_bad_num_workers_raises(fake_cmnn, tmp_path):
    from eval_vlm.inference.cmnn_backend import CMNNBackend

    cfg = _cmnn_cfg(tmp_path, num_workers=0)
    with pytest.raises(ValueError) as e:
        CMNNBackend(cfg)
    assert "num_workers" in str(e.value)


def test_cmnn_over_limit_warns_once(fake_cmnn, tmp_path, recwarn):
    from eval_vlm.inference.cmnn_backend import CMNNBackend

    backend = CMNNBackend(_cmnn_cfg(tmp_path, max_tokens=128))
    # 正常文本但 gen_seq_len 远超 max_tokens。
    fake_cmnn[0].output_for = lambda i, req: {"text": "正常的一段描述文本。", "gen_seq_len": 999}
    p1 = backend.complete_batch([BatchItem(context=_ctx(), images=["a.jpg"], sample_id="a.jpg")])
    p2 = backend.complete_batch([BatchItem(context=_ctx(), images=["b.jpg"], sample_id="b.jpg")])
    assert p1[0].error is None and p2[0].error is None
    msgs = [str(w.message) for w in recwarn.list if "max_new_tokens" in str(w.message)]
    assert len(msgs) == 1


def test_cmnn_missing_native_extension_raises(monkeypatch, tmp_path):
    """未编译安装 cmnn_native 时给出可操作的报错(指向 native/cmnn/README)。"""
    from eval_vlm.inference.cmnn_backend import CMNNBackend

    monkeypatch.setitem(sys.modules, "cmnn_native", None)  # import 触发 ImportError
    with pytest.raises(ImportError) as e:
        CMNNBackend(_cmnn_cfg(tmp_path))
    assert "cmnn_native" in str(e.value)


# ---------------------------------------------------------------------------
# 集成:predict_folder 走批量路径(验证 predict.py 的 supports_batch 分支接线)
# ---------------------------------------------------------------------------
def test_predict_folder_uses_batch_path(fake_cmnn, tmp_path):
    """pred --datadir 对 cmnn 走 complete_batch 分块路径:两张图各得一条成功描述,落盘。"""
    import json as _json

    from eval_vlm.predict import predict_folder

    cfg = _cmnn_cfg(tmp_path, num_workers=2, batch_size=8)
    datadir = Path(cfg.data.media_root)              # imgs/,内含 a.jpg + b.jpg
    cfg.run_dir_path = tmp_path / "out"              # 钉死产物目录

    stats = predict_folder(cfg, datadir)
    assert stats["newly_completed"] == 2 and stats["errors"] == 0

    # 落盘:每张一条 LlamaFactory 记录(messages + images + assistant 描述)。
    lines = [
        _json.loads(ln)
        for ln in cfg.predictions_path.read_text(encoding="utf-8").splitlines() if ln.strip()
    ]
    assert {r["id"] for r in lines} == {"a.jpg", "b.jpg"}
    assert all(r["messages"][-1]["role"] == "assistant" for r in lines)
    assert all(r["messages"][-1]["content"].startswith("描述") for r in lines)
    # 只起了一个 engine(整个 predict_folder 复用同一后端),分块内并行度=num_workers。
    assert fake_cmnn[0].num_workers == 2


def test_predict_folder_chunks_by_batch_size(fake_cmnn, tmp_path):
    """batch_size 小于图片数时按块多次调用 generate_batch(增量落盘)。"""
    from eval_vlm.predict import predict_folder

    cfg = _cmnn_cfg(tmp_path, batch_size=1)          # 每块 1 张 -> 两次 generate_batch
    datadir = Path(cfg.data.media_root)
    cfg.run_dir_path = tmp_path / "out"

    calls: list[int] = []
    orig = _FakeEngine.generate_batch

    def _counting(self, requests):
        calls.append(len(requests))
        return orig(self, requests)

    # 在实例上包裹计数(engine 在 predict_folder 内构造,故 patch 类方法)。
    _FakeEngine.generate_batch = _counting
    try:
        stats = predict_folder(cfg, datadir)
    finally:
        _FakeEngine.generate_batch = orig

    assert stats["newly_completed"] == 2
    assert calls == [1, 1]                            # 两块各 1 张
