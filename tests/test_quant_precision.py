"""quant-precision(float 权重 ↔ MNN 量化权重的逐层激活对比)的单元测试。

分档,依赖从轻到重:
  1) 量化数学(dequant_weight_affine / dequant_weight_hqq / _shrink_lp / quantize_linears_ 等)——
     需 torch(缺则 skip,仿 onnx-precision 惯例);用玩具张量/模块,不加载真模型。
  2) config 构建(quant_precision 建成 dataclass 而非 raw dict)+ CLI 参数解析 —— 无需 torch。
  3) 报告渲染(手造 summary → md)—— 无需 torch。
  4) 就地量化→前向两遍→逐层对比 的引擎逻辑(玩具 nn.Sequential,复现引擎循环)—— 需 torch。
"""
from __future__ import annotations

import numpy as np
import pytest

from eval_vlm import quant_precision as qp


# ---------------------------------------------------------------------------
# 1) 量化数学
# ---------------------------------------------------------------------------
def test_affine_constant_block_zero_error():
    """常量块(所有值相同)反量化应精确还原 -> 零误差(scale==0 兜底)。"""
    torch = pytest.importorskip("torch")
    W = torch.full((3, 8), 2.5)
    dq = qp.dequant_weight_affine(W, bit=4, block=0, sym=False)
    assert torch.allclose(dq, W)
    dq_sym = qp.dequant_weight_affine(W, bit=4, block=0, sym=True)
    assert torch.allclose(dq_sym, W)


def test_affine_lower_bit_larger_error():
    """位宽越低,反量化误差越大(4bit 误差 > 8bit)。"""
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    W = torch.randn(4, 32) * 0.5          # 随机(非均匀)权重,不落在规则网格上
    e4 = qp.rel_l2(qp.dequant_weight_affine(W, 4, 0, False).numpy(), W.numpy())
    e8 = qp.rel_l2(qp.dequant_weight_affine(W, 8, 0, False).numpy(), W.numpy())
    assert e8 < e4
    assert e4 > 0.0


def test_affine_blocking_and_tail_block():
    """分块量化:block 越小、每块动态范围越窄 -> 误差不增;且 ic 不整除 block 的尾块也被处理(形状不变、有限)。"""
    torch = pytest.importorskip("torch")
    torch.manual_seed(4)
    W = torch.randn(4, 18) * 0.4                                 # 18 不被 8 整除 -> 有尾块
    dq_full = qp.dequant_weight_affine(W, 4, 0, False)            # 整行一块
    dq_blk = qp.dequant_weight_affine(W, 4, 8, False)             # block=8 -> 2 满块 + 尾块(2)
    assert dq_blk.shape == W.shape
    assert torch.isfinite(dq_blk).all()
    e_full = qp.rel_l2(dq_full.numpy(), W.numpy())
    e_blk = qp.rel_l2(dq_blk.numpy(), W.numpy())
    assert e_blk <= e_full + 1e-6                                 # 更细分块不会更差


def test_shrink_lp_math():
    """_shrink_lp 逐字:lp=1 -> sign(x)·relu(|x|-1/β);lp=2 -> sign(x)·relu(|x|-(1/β)|x|)。"""
    torch = pytest.importorskip("torch")
    x = torch.tensor([2.0, -2.0, 0.05, -0.05])
    # lp_norm==1, beta=10 -> 阈值 0.1:2->1.9, 0.05->0(被收缩掉)
    out1 = qp._shrink_lp(x, beta=10.0, lp_norm=1.0)
    assert out1[0] == pytest.approx(1.9)
    assert out1[1] == pytest.approx(-1.9)
    assert out1[2] == pytest.approx(0.0)
    assert out1[3] == pytest.approx(0.0)
    # lp_norm==2, beta=10 -> |x|-(1/10)|x| = 0.9|x|
    out2 = qp._shrink_lp(x, beta=10.0, lp_norm=2.0)
    assert out2[0] == pytest.approx(1.8)
    assert out2[2] == pytest.approx(0.045)


def test_hqq_iters0_equals_minmax_reconstruction():
    """HQQ iters=0 == 纯 min/max 非对称(无符号)网格重构(可独立用 numpy 复算)。"""
    torch = pytest.importorskip("torch")
    W = torch.linspace(-1.0, 2.0, steps=32).reshape(2, 16)
    dq = qp.dequant_weight_hqq(W, bit=4, block=0, lp_norm=0.7, beta=10.0,
                               kappa=1.01, iters=0, scale_only=False)
    # 独立复算 iters=0 路径
    Wn = W.numpy()
    qmax = (1 << 4) - 1
    mx = Wn.max(axis=1, keepdims=True)
    mn = Wn.min(axis=1, keepdims=True)
    scale = (mx - mn) / qmax
    zero = np.round(-mn / scale)
    wq = np.clip(np.round(Wn / scale + zero), 0, qmax)
    expect = (wq - zero) * scale
    assert np.allclose(dq.numpy(), expect, atol=1e-5)


def test_hqq_finite_and_not_worse_than_affine():
    """HQQ 迭代:输出有限、形状不变,且 L2 误差不显著劣于同参基础仿射(通常更优)。"""
    torch = pytest.importorskip("torch")
    torch.manual_seed(1)
    W = torch.randn(6, 32) * 0.3
    aff = qp.dequant_weight_affine(W, 4, 0, False)
    hqq = qp.dequant_weight_hqq(W, 4, 0, lp_norm=0.7, beta=10.0, kappa=1.01,
                                iters=30, scale_only=False)
    assert hqq.shape == W.shape
    assert torch.isfinite(hqq).all()
    e_aff = qp.rel_l2(aff.numpy(), W.numpy())
    e_hqq = qp.rel_l2(hqq.numpy(), W.numpy())
    assert e_hqq <= e_aff * 1.5          # 不催崩(不显著劣化)


def test_hqq_constant_block_zero_error():
    torch = pytest.importorskip("torch")
    W = torch.full((2, 8), -1.3)
    dq = qp.dequant_weight_hqq(W, 4, 0, lp_norm=0.7, beta=10.0, kappa=1.01, iters=10)
    assert torch.allclose(dq, W, atol=1e-5)


def test_dequant_dispatch_hqq_vs_affine():
    """_dequant_weight:hqq=True 且 iters>0 走 HQQ,否则走仿射(iters=0 亦回落仿射)。"""
    torch = pytest.importorskip("torch")
    W = torch.linspace(-1, 1, 32).reshape(2, 16)
    spec_aff = {"bit": 4, "block": 0, "sym": False, "hqq": False,
                "lp_norm": 0.7, "beta": 10.0, "kappa": 1.01, "iters": 20, "scale_only": False}
    got = qp._dequant_weight(W, spec_aff)
    exp = qp.dequant_weight_affine(W, 4, 0, False)
    assert torch.allclose(got, exp)
    # hqq=True 但 iters=0 -> 仍走仿射
    spec_hqq0 = dict(spec_aff, hqq=True, iters=0)
    assert torch.allclose(qp._dequant_weight(W, spec_hqq0), exp)


def test_quantize_and_restore_linears():
    """quantize_linears_ 就地改 Linear.weight、备份可逐位还原;bias 与非 Linear 不被动。"""
    torch = pytest.importorskip("torch")
    torch.manual_seed(2)
    net = torch.nn.Sequential(
        torch.nn.Linear(8, 8),
        torch.nn.LayerNorm(8),      # 非 Linear:不应被量化
        torch.nn.Linear(8, 4),
    )
    spec = {"bit": 4, "block": 0, "sym": False, "hqq": False,
            "lp_norm": 0.7, "beta": 10.0, "kappa": 1.01, "iters": 0, "scale_only": False}
    w0 = net[0].weight.detach().clone()
    b0 = net[0].bias.detach().clone()
    ln_w0 = net[1].weight.detach().clone()

    backup = qp.quantize_linears_(net, spec)
    assert set(backup.keys()) == {"0", "2"}            # 只备份两个 Linear
    assert not torch.allclose(net[0].weight, w0)       # 权重被量化(改变)
    assert torch.allclose(net[0].bias, b0)             # bias 未动
    assert torch.allclose(net[1].weight, ln_w0)        # LayerNorm 未动

    qp.restore_linears_(net, backup)
    assert torch.equal(net[0].weight, w0)              # 逐位还原
    assert torch.equal(net[2].weight.detach(), backup["2"])


def test_per_layer_weight_error_positive():
    torch = pytest.importorskip("torch")
    torch.manual_seed(3)
    layer = torch.nn.Sequential(torch.nn.Linear(16, 16), torch.nn.Linear(16, 16))
    spec = {"bit": 4, "block": 0, "sym": False, "hqq": False,
            "lp_norm": 0.7, "beta": 10.0, "kappa": 1.01, "iters": 0, "scale_only": False}
    err = qp.per_layer_weight_error(layer, spec)
    assert err > 0.0 and np.isfinite(err)


# ---------------------------------------------------------------------------
# 2) config 构建 + CLI 参数解析
# ---------------------------------------------------------------------------
def test_config_quant_precision_is_dataclass():
    """quant_precision 块必须被构造成 QuantPrecisionConfig(而非留成 raw dict)。"""
    from eval_vlm.config import Config, QuantPrecisionConfig, _build

    cfg = _build(Config, {"quant_precision": {"quant_bit": 8, "targets": ["llm", "vit"],
                                              "hqq": False, "visual_quant_bit": 4}})
    assert isinstance(cfg.quant_precision, QuantPrecisionConfig)
    assert cfg.quant_precision.quant_bit == 8
    assert cfg.quant_precision.targets == ["llm", "vit"]
    assert cfg.quant_precision.hqq is False
    assert cfg.quant_precision.visual_quant_bit == 4
    # 缺省字段仍取默认
    assert cfg.quant_precision.quant_block == 128
    assert cfg.quant_precision.rel_l2_max == 0.1


def test_config_quant_precision_defaults():
    from eval_vlm.config import Config

    cfg = Config()
    assert cfg.quant_precision.targets == ["llm"]
    assert cfg.quant_precision.quant_bit == 4
    assert cfg.quant_precision.hqq is True
    assert cfg.quant_precision.visual_quant_bit is None
    assert cfg.quant_precision.dtype == "float32"


def test_cli_quant_precision_parses():
    from eval_vlm.cli import build_parser, _cmd_quant_precision

    p = build_parser()
    ns = p.parse_args(["quant-precision", "-d", "ds", "--hf-model", "/w/ckpt",
                       "--targets", "llm,vit", "--num-samples", "4",
                       "--quant-bit", "8", "--quant-block", "64", "--sym",
                       "--no-hqq", "--visual-quant-bit", "4",
                       "--cosine-min", "0.98", "--rel-l2-max", "0.2"])
    assert ns.func is _cmd_quant_precision
    assert ns.dataset == "ds"
    assert ns.hf_model == "/w/ckpt"
    assert ns.targets == "llm,vit"
    assert ns.num_samples == 4
    assert ns.quant_bit == 8
    assert ns.quant_block == 64
    assert ns.sym is True
    assert ns.hqq is False            # --no-hqq
    assert ns.visual_quant_bit == 4
    assert ns.cosine_min == 0.98
    assert ns.rel_l2_max == 0.2


def test_cli_quant_precision_hqq_default_none():
    """未给 --hqq/--no-hqq 时 args.hqq 为 None(表示「用配置默认」)。"""
    from eval_vlm.cli import build_parser

    ns = build_parser().parse_args(["quant-precision", "-d", "ds"])
    assert ns.hqq is None
    assert ns.sym is False


# ---------------------------------------------------------------------------
# 3) 报告渲染(手造 summary,无需 torch)
# ---------------------------------------------------------------------------
def test_report_render_smoke():
    from eval_vlm.config import Config

    cfg = Config()
    qpc = cfg.quant_precision
    targets = {
        "llm": {"available": True, "target": "llm", "num_layers": 2, "num_images": 2,
                "first_divergence_layer": 1, "first_divergence_name": "dec_layer_1",
                "worst_layer": {"name": "dec_layer_1", "min_cosine": 0.8, "max_rel_l2": 0.3,
                                "index": 1, "mean_cosine": 0.9, "mean_rel_l2": 0.2,
                                "weight_rel_l2": 0.12, "diverged": True},
                "layers": [
                    {"index": 0, "name": "dec_layer_0", "mean_cosine": 0.999, "min_cosine": 0.999,
                     "mean_rel_l2": 0.01, "max_rel_l2": 0.02, "weight_rel_l2": 0.08, "diverged": False},
                    {"index": 1, "name": "dec_layer_1", "mean_cosine": 0.9, "min_cosine": 0.8,
                     "mean_rel_l2": 0.2, "max_rel_l2": 0.3, "weight_rel_l2": 0.12, "diverged": True},
                ]},
        "vit": {"available": False, "target": "vit",
                "reason": "ViT 未量化(visual_quant_bit=None)"},
    }
    flags = qp._build_flags(targets, qpc)
    summary = {
        "model": "toy", "reference_model_path": "/w/ckpt", "dtype": "float32", "device": "cpu",
        "quant": {"quant_bit": 4, "quant_block": 128, "sym": False, "hqq": True,
                  "hqq_lp_norm": 0.7, "hqq_beta": 10.0, "hqq_kappa": 1.01, "hqq_iters": 20,
                  "hqq_scale_only": False, "visual_quant_bit": None, "visual_quant_block": 128},
        "cosine_min": qpc.cosine_min, "rel_l2_max": qpc.rel_l2_max,
        "num_samples": 2, "sample_ids": ["a", "b"], "compared_at": "now",
        "targets": targets, "flags": flags,
    }
    md = qp._render_md(summary, cfg)
    assert "量化逐层激活精度报告" in md
    assert "dec_layer_1" in md
    assert "🔴" in md                       # LLM 发散判定
    assert "未对比" in md                    # vit 不可用小节
    assert "HQQ" in md
    assert any("LLM 解码器" in f and "🔴" in f for f in flags)
    assert any("视觉塔" in f and "⏭️" in f for f in flags)


def test_build_flags_all_good():
    from eval_vlm.config import Config

    qpc = Config().quant_precision
    targets = {
        "llm": {"available": True, "target": "llm", "num_layers": 2, "num_images": 1,
                "first_divergence_layer": None, "first_divergence_name": None,
                "worst_layer": {"name": "dec_layer_1", "min_cosine": 0.995,
                                "max_rel_l2": 0.05, "weight_rel_l2": 0.07},
                "layers": []},
    }
    flags = qp._build_flags(targets, qpc)
    assert any("在容差内" in f for f in flags)
    assert any(f.startswith("✅") for f in flags)


# ---------------------------------------------------------------------------
# 4) 引擎逻辑:就地量化 -> 两遍前向 -> 逐层对比(玩具网络复现引擎循环)
# ---------------------------------------------------------------------------
def test_two_pass_quantize_forward_locates_divergence():
    """复现引擎核心:玩具多层网,float 权重跑一遍存激活 -> 量化 -> 再跑 -> 逐层比 -> 定位首「崩」层。

    4bit 量化必然引入非零逐层误差;用紧阈值验证机器能定位到某「崩」层,且恢复后权重/激活精确复原。
    (注:仿射量化的相对误差与权重绝对尺度无关,故靠放大权重制造发散无效——用阈值控制。)"""
    torch = pytest.importorskip("torch")
    torch.manual_seed(7)

    layers = [torch.nn.Linear(16, 16) for _ in range(4)]
    net = torch.nn.Sequential(*layers)
    invoke = lambda base, x: base(x)  # noqa: E731

    x = torch.randn(3, 16)
    probe = qp.build_layer_probe(net, layers, invoke)
    try:
        float_acts = qp.torch_reference_activations(probe, (x,))
    finally:
        probe.close()

    spec = {"bit": 4, "block": 0, "sym": False, "hqq": False,
            "lp_norm": 0.7, "beta": 10.0, "kappa": 1.01, "iters": 0, "scale_only": False}
    backup = qp.quantize_linears_(net, spec)
    try:
        probe = qp.build_layer_probe(net, layers, invoke)
        try:
            quant_acts = qp.torch_reference_activations(probe, (x,))
        finally:
            probe.close()
    finally:
        qp.restore_linears_(net, backup)

    names = [f"dec_layer_{i}" for i in range(4)]
    per_image = [[qp.layer_metric(q, f) for q, f in zip(quant_acts, float_acts)]]
    # 紧阈值:4bit 误差必致某层「崩」
    agg, first_div = qp._aggregate_layers(per_image, names, cosine_min=0.99999, rel_l2_max=1e-5)
    assert len(agg) == 4
    assert first_div is not None            # 量化必致某层「崩」
    assert agg[0]["max_rel_l2"] > 0.0       # 首层已有非零量化误差
    # 恢复后权重逐位复原
    probe = qp.build_layer_probe(net, layers, invoke)
    try:
        restored_acts = qp.torch_reference_activations(probe, (x,))
    finally:
        probe.close()
    for r, f in zip(restored_acts, float_acts):
        assert np.allclose(r, f, atol=1e-5)
