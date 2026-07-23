"""precision(mnn 转换后 vs hf 转换前 行为级对比)与 hf 参考后端的单元测试。

precision 部分不依赖任何模型:手写两份 predictions.jsonl 验证对齐/指标/报告。
hf 后端部分用注入的假 torch/transformers 验证 prompt 构造与 raw 元数据(CI 无 GPU 也能跑)。
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from eval_vlm.config import Config
from eval_vlm.data.schema import Prediction, Turn
from eval_vlm.results import store


# ---------------------------------------------------------------------------
# 工具:构造一个钉住 dataset_dir 的 Config,并写两份预测
# ---------------------------------------------------------------------------
def _cfg(tmp_path: Path) -> Config:
    cfg = Config()
    cfg.run_dir_path = tmp_path            # dataset_dir 直接钉到 tmp
    cfg.precision.candidate_dir = "cand"
    cfg.precision.reference_dir = "ref"
    return cfg


def _write_preds(cfg: Config, model: str, preds: list[Prediction],
                 backend: str = "mnn") -> None:
    path = cfg.predictions_path_for(model, backend)
    with store.PredictionWriter(path) as w:
        for p in preds:
            w.write(p)


def _p(pid, pred, *, turn=0, prompt_tokens=None, image_pixels=None, error=None) -> Prediction:
    raw = {}
    if prompt_tokens is not None:
        raw["prompt_token_count"] = prompt_tokens
    if image_pixels is not None:
        raw["image_pixels"] = image_pixels
    return Prediction(id=pid, turn=turn, prediction=pred, raw=raw or None, error=error)


# ---------------------------------------------------------------------------
# 文本级指标 helper
# ---------------------------------------------------------------------------
def test_first_divergence_and_edit_similarity():
    from eval_vlm.precision import _edit_similarity, _first_divergence

    assert _first_divergence("红色的汽车", "红色的汽车") is None      # 完全一致
    assert _first_divergence("一只猫", "一只狗") == 2                 # 「一只」相同,第3字发散
    assert _first_divergence("abc", "abcdef") == 3                    # 前缀:较短串长度
    assert _first_divergence("xyz", "abc") == 0                       # 一开始就发散
    assert _edit_similarity("同样文本", "同样文本") == 1.0
    assert _edit_similarity("", "") == 1.0
    assert 0.0 <= _edit_similarity("完全不同aaa", "毫不相干bbb") < 0.5


# ---------------------------------------------------------------------------
# 对比:指标 + 报告落盘
# ---------------------------------------------------------------------------
def test_compare_precision_basic_metrics(tmp_path):
    from eval_vlm.precision import compare_precision

    cfg = _cfg(tmp_path)
    # A 完全一致;B 尾部发散;C 差异大且预处理不对齐(prompt/pixel delta≠0)。
    _write_preds(cfg, "cand", [
        _p("A", "红色的汽车", prompt_tokens=100, image_pixels=500),
        _p("B", "一只猫", prompt_tokens=100, image_pixels=500),
        _p("C", "完全不同的很长的输出内容", prompt_tokens=120, image_pixels=600),
    ])
    _write_preds(cfg, "ref", [
        _p("A", "红色的汽车", prompt_tokens=100, image_pixels=500),
        _p("B", "一只狗", prompt_tokens=100, image_pixels=500),
        _p("C", "短", prompt_tokens=100, image_pixels=500),
    ], backend="hf")

    summary = compare_precision(cfg, candidate="cand", reference="ref")

    assert summary["num_compared"] == 3
    b = summary["behavior"]
    assert b["agreement_rate"] == pytest.approx(1 / 3, abs=1e-3)   # 仅 A 完全一致
    assert b["num_identical"] == 1
    assert b["num_diverged"] == 2
    # 对齐审计:C 的 prompt token 与图片像素都不一致。
    assert summary["alignment"]["prompt_tokens"]["available"] == 3
    assert summary["alignment"]["prompt_tokens"]["mismatches"] == 1
    assert summary["alignment"]["prompt_tokens"]["max_abs_delta"] == 20
    assert summary["alignment"]["image_pixels"]["mismatches"] == 1
    # 报告落盘到候选目录。
    assert (cfg.model_run_dir("cand", "mnn") / "precision.json").exists()
    md = (cfg.model_run_dir("cand", "mnn") / "precision.md").read_text(encoding="utf-8")
    assert "精度对比报告" in md and "最差样本" in md
    assert summary["report_md"].endswith("precision.md")


def test_compare_precision_flags_preprocessing_mismatch(tmp_path):
    """prompt token / 图片像素 delta≠0 时,报告 flags 标注「预处理不对齐」。"""
    from eval_vlm.precision import compare_precision

    cfg = _cfg(tmp_path)
    _write_preds(cfg, "cand", [_p("A", "猫", prompt_tokens=130, image_pixels=800)])
    _write_preds(cfg, "ref", [_p("A", "狗", prompt_tokens=100, image_pixels=500)], backend="hf")

    summary = compare_precision(cfg, candidate="cand", reference="ref")
    joined = " ".join(summary["flags"])
    assert "prompt token 数不对齐" in joined
    assert "图片 resize 像素不一致" in joined


def test_compare_precision_skips_errors_and_counts_only_sides(tmp_path):
    from eval_vlm.precision import compare_precision

    cfg = _cfg(tmp_path)
    _write_preds(cfg, "cand", [
        _p("A", "x", prompt_tokens=10, image_pixels=10),
        _p("B", "", error="boom"),          # 候选报错 -> 跳过
        _p("D", "only-cand"),               # 仅候选有
    ])
    _write_preds(cfg, "ref", [
        _p("A", "x", prompt_tokens=10, image_pixels=10),
        _p("B", "ok"),
        _p("E", "only-ref"),                # 仅参考有
    ], backend="hf")
    summary = compare_precision(cfg, candidate="cand", reference="ref")
    assert summary["num_compared"] == 1       # 只有 A(B 因候选报错跳过)
    assert summary["num_errors_skipped"] == 1
    assert summary["num_only_candidate"] == 1  # D
    assert summary["num_only_reference"] == 1  # E


def test_compare_precision_missing_predictions_raises(tmp_path):
    from eval_vlm.precision import compare_precision

    cfg = _cfg(tmp_path)
    _write_preds(cfg, "cand", [_p("A", "x")])   # 只有候选,没有参考
    with pytest.raises(FileNotFoundError):
        compare_precision(cfg, candidate="cand", reference="ref")


def _write_scored(cfg, model, backend, rows):
    store.write_jsonl(cfg.model_run_dir(model, backend) / "scored.jsonl", rows)


def test_compare_precision_quality_delta_present(tmp_path):
    """两端都跑过 score(scored.jsonl 齐全)-> summary 含净质量Δ + 净回归 flag。"""
    from eval_vlm.precision import compare_precision

    cfg = _cfg(tmp_path)
    cfg.precision.quality_regression_max = 0.4          # 让 0.5 净回归触发 flag
    _write_preds(cfg, "cand", [_p("A", "x"), _p("B", "y")])
    _write_preds(cfg, "ref", [_p("A", "x"), _p("B", "y")], backend="hf")
    # 候选:A对 B错;参考(HF):A对 B对 -> B 净回归
    _write_scored(cfg, "cand", "mnn", [
        {"id": "A", "turn": 0, "scorer": "exact_match", "score": 1.0},
        {"id": "B", "turn": 0, "scorer": "exact_match", "score": 0.0}])
    _write_scored(cfg, "ref", "hf", [
        {"id": "A", "turn": 0, "scorer": "exact_match", "score": 1.0},
        {"id": "B", "turn": 0, "scorer": "exact_match", "score": 1.0}])

    summary = compare_precision(cfg, candidate="cand", reference="ref")
    q = summary["quality"]
    assert q["available"] is True
    assert q["binary"]["net_regression_rate"] == 0.5
    assert any("净质量回归" in f for f in summary["flags"])
    assert "净质量Δ" in (cfg.model_run_dir("cand", "mnn") / "precision.md").read_text(encoding="utf-8")


def test_compare_precision_quality_delta_degrades_without_scored(tmp_path):
    """任一端缺 scored.jsonl -> 净质量Δ 优雅降级(available=False),不报错。"""
    from eval_vlm.precision import compare_precision

    cfg = _cfg(tmp_path)
    _write_preds(cfg, "cand", [_p("A", "x")])
    _write_preds(cfg, "ref", [_p("A", "x")], backend="hf")   # 没写任何 scored.jsonl

    summary = compare_precision(cfg, candidate="cand", reference="ref")
    assert summary["quality"]["available"] is False
    assert "score" in summary["quality"]["reason"]


# ---------------------------------------------------------------------------
# hf 参考后端(注入假 torch / transformers)
# ---------------------------------------------------------------------------
class _Batch:
    """二维 id 张量的极简替身:支持 .shape 与 [:, start:] 切片。"""
    def __init__(self, rows):
        self.rows = [list(r) for r in rows]

    @property
    def shape(self):
        return (len(self.rows), len(self.rows[0]) if self.rows else 0)

    def __getitem__(self, key):
        if isinstance(key, tuple):
            _, col = key
            return _Batch([r[col] for r in self.rows])
        return _Row(self.rows[key])


class _Row(list):
    def tolist(self):
        return list(self)


class _Inputs(dict):
    """BatchFeature 替身:既能 **inputs 解包,又能属性访问与 .to()。"""
    def to(self, device):
        return self

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e


class _FakeProcessor:
    def __init__(self):
        self.image_processor = types.SimpleNamespace(merge_size=2, patch_size=14)
        self.apply_calls = []

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        self.apply_calls.append(messages)
        return "PROMPT"

    def __call__(self, text=None, images=None, padding=True, return_tensors=None):
        # prompt 3 个 token;image_grid_thw=[1,10,10] -> 像素 (10*14)^2=19600、视觉 token 25。
        return _Inputs(input_ids=_Batch([[1, 2, 3]]),
                       image_grid_thw=[[1, 10, 10]],
                       pixel_values=[0])

    def batch_decode(self, seqs, skip_special_tokens=True, clean_up_tokenization_spaces=False):
        return ["这是参考模型的描述"]


class _FakeModel:
    device = "cpu"

    def eval(self):
        return self

    def generate(self, **kw):
        prompt = kw["input_ids"].rows[0]
        return _Batch([prompt + [7, 8, 9]])   # 追加 3 个生成 token


@pytest.fixture
def fake_hf(monkeypatch):
    import contextlib

    torch_mod = types.ModuleType("torch")
    torch_mod.inference_mode = lambda: contextlib.nullcontext()
    torch_mod.bfloat16 = "bfloat16"
    torch_mod.float16 = "float16"

    tf_mod = types.ModuleType("transformers")
    proc = _FakeProcessor()
    model = _FakeModel()
    tf_mod.AutoProcessor = types.SimpleNamespace(
        from_pretrained=lambda path, **k: proc)
    tf_mod.AutoModelForVision2Seq = types.SimpleNamespace(
        from_pretrained=lambda path, **k: model)

    monkeypatch.setitem(sys.modules, "torch", torch_mod)
    monkeypatch.setitem(sys.modules, "transformers", tf_mod)
    return proc, model


def _hf_cfg(tmp_path) -> Config:
    from PIL import Image
    imgs = tmp_path / "imgs"
    imgs.mkdir()
    Image.new("RGB", (64, 64), (10, 20, 30)).save(imgs / "a.jpg", format="JPEG")
    cfg = Config()
    cfg.inference.backend = "hf"
    cfg.inference.hf.model_path = str(tmp_path / "hf_ckpt")
    cfg.inference.hf.max_tokens = 32
    cfg.data.media_root = str(imgs)
    return cfg


def test_hf_backend_captures_alignment_raw(fake_hf, tmp_path):
    from eval_vlm.inference.hf_backend import HFBackend

    cfg = _hf_cfg(tmp_path)
    backend = HFBackend(cfg)
    assert backend.thread_safe is False

    ctx = [Turn(role="user", content="<image>请描述图片")]
    pred = backend.complete(ctx, ["a.jpg"], "a.jpg")

    assert pred.error is None
    assert pred.prediction == "这是参考模型的描述"
    # 对齐元数据落 raw:prompt token 数、图片像素/视觉 token、输出 token ids。
    assert pred.raw["backend"] == "hf"
    assert pred.raw["prompt_token_count"] == 3
    assert pred.raw["image_pixels"] == (10 * 14) * (10 * 14)
    assert pred.raw["image_tokens"] == (1 * 10 * 10) // 4
    assert pred.raw["output_token_ids"] == [7, 8, 9]


def test_hf_backend_requires_single_image(fake_hf, tmp_path):
    from eval_vlm.inference.hf_backend import HFBackend

    cfg = _hf_cfg(tmp_path)
    backend = HFBackend(cfg)
    ctx = [Turn(role="user", content="<image>请描述图片")]
    pred = backend.complete(ctx, ["a.jpg", "b.jpg"], "x")
    assert pred.error is not None and "单图" in pred.error


def test_hf_backend_missing_model_path_raises(fake_hf, tmp_path):
    from eval_vlm.inference.hf_backend import HFBackend

    cfg = _hf_cfg(tmp_path)
    cfg.inference.hf.model_path = None
    with pytest.raises(ValueError) as e:
        HFBackend(cfg)
    assert "model_path" in str(e.value)


# ---------------------------------------------------------------------------
# 工厂分发 & 配置解析
# ---------------------------------------------------------------------------
def test_build_backend_dispatches_hf(fake_hf, tmp_path):
    from eval_vlm.inference import build_backend
    from eval_vlm.inference.hf_backend import HFBackend

    cfg = _hf_cfg(tmp_path)
    assert isinstance(build_backend(cfg), HFBackend)


def test_hf_result_name_from_model_path():
    cfg = Config()
    cfg.inference.backend = "hf"
    cfg.inference.hf.model_path = "/models/qwen2-vl-7b-emotion"
    assert cfg.inference.result_name == "qwen2-vl-7b-emotion"
    assert cfg.inference.active is cfg.inference.hf


def test_parser_pred_hf_flag_and_precision_subcommand():
    from eval_vlm.cli import build_parser, _cmd_pred, _cmd_precision

    parser = build_parser()
    a = parser.parse_args(["pred", "--datadir", "imgs", "--backend", "hf",
                           "--hf-model", "/m/hf"])
    assert a.func is _cmd_pred and a.backend == "hf" and a.hf_model == "/m/hf"

    a2 = parser.parse_args(["precision", "--dataset", "ds",
                            "--candidate-dir", "c", "--reference-dir", "r"])
    assert a2.func is _cmd_precision
    assert a2.candidate_dir == "c" and a2.reference_dir == "r"
