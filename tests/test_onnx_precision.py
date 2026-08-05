"""onnx-precision(torch/safetensors ↔ ONNX 逐层激活数值级校验)的单元测试。

分三档,依赖从轻到重:
  1) 纯 numpy 指标核心(cosine / rel_l2 / max_abs / 逐层聚合 + 首发散层)—— 无需 torch,始终跑。
  2) config 构建(onnx_precision 建成 dataclass 而非 raw dict)+ CLI 参数解析 —— 无需 torch,始终跑。
  3) probe 引擎端到端(玩具 nn.Sequential → torch.onnx.export → onnxruntime 逐层对比)——
     未装 torch/onnx/onnxruntime 时按现有惯例 skip(仿 PIL/MNN 缺失)。
"""
from __future__ import annotations

import numpy as np
import pytest

from eval_vlm import onnx_precision as op


# ---------------------------------------------------------------------------
# 1) 纯 numpy 指标核心
# ---------------------------------------------------------------------------
def test_cosine_math():
    a = np.array([1.0, 2.0, 3.0])
    assert op.cosine(a, a) == pytest.approx(1.0)
    # 反向 -> -1
    assert op.cosine(a, -a) == pytest.approx(-1.0)
    # 正交 -> 0
    assert op.cosine(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(0.0)
    # 零向量约定:双零=1.0,单零=0.0
    z = np.zeros(3)
    assert op.cosine(z, z) == 1.0
    assert op.cosine(a, z) == 0.0


def test_rel_l2_and_abs_math():
    ref = np.array([3.0, 4.0])                 # ‖ref‖ = 5
    cand = np.array([3.0, 4.0 + 0.5])          # 差向量 (0, 0.5) -> ‖diff‖ = 0.5
    assert op.rel_l2(cand, ref) == pytest.approx(0.5 / 5.0)
    assert op.max_abs_err(cand, ref) == pytest.approx(0.5)
    assert op.mean_abs_err(cand, ref) == pytest.approx(0.25)
    # 参考为零向量:rel_l2 退化为绝对 L2
    assert op.rel_l2(np.array([0.0, 3.0]), np.zeros(2)) == pytest.approx(3.0)


def test_layer_metric_shape():
    a = np.arange(6, dtype=float).reshape(2, 3)
    m = op.layer_metric(a, a)
    assert m["cosine"] == pytest.approx(1.0)
    assert m["rel_l2"] == pytest.approx(0.0)
    assert m["max_abs_err"] == 0.0


def test_aggregate_locates_first_divergence():
    """逐图×逐层聚合:某中间层在某一张图上发散 -> 首发散层 == 该层下标(跨图取最差)。"""
    good = np.array([1.0, 2.0, 3.0])
    bad = np.array([1.0, 2.0, -3.0])           # 与 good 的 cosine 明显 <1
    # 3 层、2 图:仅第 1 层在第 2 张图上发散
    per_image = [
        [op.layer_metric(good, good), op.layer_metric(good, good), op.layer_metric(good, good)],
        [op.layer_metric(good, good), op.layer_metric(bad, good), op.layer_metric(good, good)],
    ]
    layers, first_div = op._aggregate_layers(
        per_image, ["l0", "l1", "l2"], cosine_min=0.9999, rel_l2_max=1e-2)
    assert first_div == 1
    assert layers[0]["diverged"] is False
    assert layers[1]["diverged"] is True
    assert layers[2]["diverged"] is False
    # 跨图取最差:第 1 层 min_cosine 来自第 2 张图
    assert layers[1]["min_cosine"] < 0.9999


def test_aggregate_all_good_no_divergence():
    a = np.array([1.0, 2.0, 3.0])
    per_image = [[op.layer_metric(a, a), op.layer_metric(a, a)]]
    layers, first_div = op._aggregate_layers(
        per_image, ["l0", "l1"], cosine_min=0.9999, rel_l2_max=1e-2)
    assert first_div is None
    assert all(not ly["diverged"] for ly in layers)


def test_diverged_predicate():
    assert op._diverged(0.99, 0.0, cosine_min=0.9999, rel_l2_max=1e-2) is True   # cosine 不达标
    assert op._diverged(1.0, 0.5, cosine_min=0.9999, rel_l2_max=1e-2) is True    # rel_l2 超标
    assert op._diverged(1.0, 0.0, cosine_min=0.9999, rel_l2_max=1e-2) is False


def test_call_rope_index_adapts_to_signature():
    """get_rope_index 签名跨代不同:Qwen2/2.5-VL 无 mm_token_type_ids,Qwen3/3.5-VL 必须传。
    _call_rope_index 应按签名反射选参——两代都不报 TypeError,且新代确实把 mm 透传进去。"""
    pytest.importorskip("torch")
    seen = {}

    # Qwen3/3.5-VL 风格:mm_token_type_ids 是必需位置参
    def rope_new(input_ids, mm_token_type_ids, image_grid_thw=None, attention_mask=None):
        seen["mm"] = mm_token_type_ids
        seen["gthw"] = image_grid_thw
        return ("POS_NEW",)          # 返回元组 -> helper 取 [0]

    # Qwen2/2.5-VL 风格:没有 mm_token_type_ids 形参
    def rope_old(input_ids, image_grid_thw=None, attention_mask=None):
        seen["old_called"] = True
        return "POS_OLD"

    inputs = {"input_ids": "IDS", "image_grid_thw": "GTHW",
              "mm_token_type_ids": "MM"}

    pos_new = op._call_rope_index(rope_new, model=None, inputs=inputs, attn="ATTN", gthw="GTHW")
    assert pos_new == "POS_NEW"      # 元组被解包
    assert seen["mm"] == "MM"        # processor 的 mm_token_type_ids 被透传
    assert seen["gthw"] == "GTHW"

    # 老签名不含 mm 形参:不应因多喂 mm 而报 TypeError
    pos_old = op._call_rope_index(rope_old, model=None, inputs=inputs, attn="ATTN", gthw="GTHW")
    assert pos_old == "POS_OLD"
    assert seen.get("old_called") is True


# ---------------------------------------------------------------------------
# 2) config 构建 + CLI 参数解析
# ---------------------------------------------------------------------------
def test_config_onnx_precision_is_dataclass():
    """onnx_precision 块必须被构造成 ONNXPrecisionConfig(而非留成 raw dict——vllm_offline 踩过的坑)。"""
    from eval_vlm.config import Config, ONNXPrecisionConfig, _build

    cfg = _build(Config, {"onnx_precision": {"cosine_min": 0.5, "targets": ["vit"],
                                             "num_samples": 3, "keep_onnx": True}})
    assert isinstance(cfg.onnx_precision, ONNXPrecisionConfig)
    assert cfg.onnx_precision.cosine_min == 0.5
    assert cfg.onnx_precision.targets == ["vit"]
    assert cfg.onnx_precision.num_samples == 3
    assert cfg.onnx_precision.keep_onnx is True
    # 缺省字段仍取默认
    assert cfg.onnx_precision.rel_l2_max == 0.01


def test_config_onnx_precision_defaults():
    from eval_vlm.config import Config

    cfg = Config()
    assert cfg.onnx_precision.targets == ["vit", "llm"]
    assert cfg.onnx_precision.dtype == "float32"
    assert cfg.onnx_precision.device == "cpu"


def test_cli_onnx_precision_parses():
    from eval_vlm.cli import build_parser, _cmd_onnx_precision

    p = build_parser()
    ns = p.parse_args(["onnx-precision", "-d", "ds", "--hf-model", "/w/ckpt",
                       "--targets", "vit,llm", "--num-samples", "4",
                       "--dtype", "float32", "--cosine-min", "0.999",
                       "--rel-l2-max", "0.02", "--keep-onnx"])
    assert ns.func is _cmd_onnx_precision
    assert ns.dataset == "ds"
    assert ns.hf_model == "/w/ckpt"
    assert ns.targets == "vit,llm"
    assert ns.num_samples == 4
    assert ns.cosine_min == 0.999
    assert ns.rel_l2_max == 0.02
    assert ns.keep_onnx is True


def test_report_render_smoke():
    """报告渲染不依赖 torch:喂一个手造 summary,确认 md 含判定与逐层表关键字段。"""
    from eval_vlm.config import Config

    cfg = Config()
    summary = {
        "model": "toy", "reference_model_path": "/w/ckpt", "dtype": "float32",
        "device": "cpu", "opset": 17, "cosine_min": 0.9999, "rel_l2_max": 0.01,
        "num_samples": 2, "sample_ids": ["a", "b"], "compared_at": "now",
        "targets": {
            "vit": {"available": True, "target": "vit", "num_layers": 2, "num_images": 2,
                    "first_divergence_layer": 1, "first_divergence_name": "vit_block_1",
                    "worst_layer": {"name": "vit_block_1", "min_cosine": 0.9,
                                    "max_rel_l2": 0.3, "index": 1, "mean_cosine": 0.95,
                                    "mean_rel_l2": 0.2, "mean_max_abs_err": 0.1,
                                    "diverged": True},
                    "layers": [
                        {"index": 0, "name": "vit_block_0", "mean_cosine": 1.0,
                         "min_cosine": 1.0, "mean_rel_l2": 0.0, "max_rel_l2": 0.0,
                         "mean_max_abs_err": 0.0, "diverged": False},
                        {"index": 1, "name": "vit_block_1", "mean_cosine": 0.95,
                         "min_cosine": 0.9, "mean_rel_l2": 0.2, "max_rel_l2": 0.3,
                         "mean_max_abs_err": 0.1, "diverged": True},
                    ]},
            "llm": {"available": False, "reason": "未知 target"},
        },
    }
    summary["flags"] = op._build_flags(summary["targets"], 0.9999, 0.01)
    md = op._render_md(summary, cfg)
    assert "ONNX 逐层激活精度报告" in md
    assert "vit_block_1" in md          # 首发散层出现在表里
    assert "🔴" in md                    # ViT 发散判定
    assert "未对比" in md                # llm 不可用小节
    # flags 里 ViT 应标发散、llm 应标跳过
    assert any("视觉塔" in f and "🔴" in f for f in summary["flags"])
    assert any("LLM 解码器" in f and "⏭️" in f for f in summary["flags"])


# ---------------------------------------------------------------------------
# 3) probe 引擎端到端(玩具模块;未装 torch/onnx/onnxruntime 则 skip)
# ---------------------------------------------------------------------------
def test_probe_export_roundtrip_toy_model(tmp_path):
    """玩具 Linear+ReLU 网络包装成 probe -> 真 torch.onnx.export -> onnxruntime:
    逐层 cosine≈1、rel_l2≈0、首发散层为 None(验证 probe/导出/ORT/指标全链路)。"""
    torch = pytest.importorskip("torch")
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = torch.nn.Linear(4, 6)
            self.l2 = torch.nn.Linear(6, 4)

        def forward(self, x):
            return self.l2(torch.relu(self.l1(x)))

    net = Net().eval()
    layers = [net.l1, net.l2]
    names = ["l1", "l2"]
    invoke = lambda base, x: base(x)  # noqa: E731

    # 固定输入(不用随机,保证可复现)
    x = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    args = (x,)

    probe = op.build_layer_probe(net, layers, invoke)
    ref = op.torch_reference_activations(probe, args)
    assert len(ref) == 2

    onnx_path = tmp_path / "toy.onnx"
    op.export_probe_onnx(probe, args, onnx_path,
                         input_names=["x"], output_names=names, opset=17)
    cand = op.ort_activations(onnx_path, names, {"x": x.numpy()})
    assert len(cand) == 2

    per_image = [[op.layer_metric(c, r) for c, r in zip(cand, ref)]]
    layers_agg, first_div = op._aggregate_layers(
        per_image, names, cosine_min=0.9999, rel_l2_max=1e-3)
    assert first_div is None, layers_agg
    for ly in layers_agg:
        assert ly["min_cosine"] > 0.9999
        assert ly["max_rel_l2"] < 1e-3


def test_probe_close_removes_hooks():
    """反复建 probe 必须能摘干净 hook,否则会累加到共享层上(O(N²)+ 激活泄漏)。"""
    torch = pytest.importorskip("torch")

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = torch.nn.Linear(4, 4)

        def forward(self, x):
            return self.l1(x)

    net = Net().eval()
    invoke = lambda base, x: base(x)  # noqa: E731
    assert len(net.l1._forward_hooks) == 0
    for _ in range(5):
        probe = op.build_layer_probe(net, [net.l1], invoke)
        assert len(net.l1._forward_hooks) == 1   # 建 probe 后恰好 1 个
        probe.close()
        assert len(net.l1._forward_hooks) == 0   # close 后归零,不累加


def test_ort_feed_filters_pruned_input(tmp_path):
    """未被图使用的输入会被导出器裁掉;ort_activations 应过滤多余 feed 键而非报错
    (复刻 Qwen 视觉塔 grid_thw 被折成常量后遭裁剪的场景)。"""
    torch = pytest.importorskip("torch")
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = torch.nn.Linear(4, 4)

        def forward(self, x, unused):   # unused 从不参与计算 -> 导出时被裁成死输入
            return self.l1(x)

    net = Net().eval()
    invoke = lambda base, x, u: base(x, u)  # noqa: E731
    x = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    unused = torch.zeros(2, 4)
    probe = op.build_layer_probe(net, [net.l1], invoke)
    try:
        onnx_path = tmp_path / "pruned.onnx"
        op.export_probe_onnx(probe, (x, unused), onnx_path,
                             input_names=["x", "unused"], output_names=["l1"], opset=17)
    finally:
        probe.close()
    # 同时喂 x 与被裁的 unused:过滤后仍能正常取回激活,不抛 Invalid Feed Input Name。
    cand = op.ort_activations(onnx_path, ["l1"], {"x": x.numpy(), "unused": unused.numpy()})
    assert len(cand) == 1
    assert cand[0].shape == (2, 4)
