"""quant-precision 命令:float 权重 ↔ MNN 量化权重的**逐层激活**对比。

背景:MNN 效果退化的诊断链路是 `safetensors →(llmexport)→ ONNX → MNN`。已有两条校验:
  - `precision`:**行为级**(MNN vs HF 文本一致率),端到端但受「MNN 高层 API 无 logits」限制;
  - `onnx-precision`:**导出保真度**(torch vs ONNX,float32),证明 `torch→ONNX` 这一段无损。
缺的一环是**权重量化本身对每层激活的影响**——也是本模块补的。

已查实 MNN 源码:量化在 `llmexport.py` 导出阶段对 torch 权重张量做(**weight-only**,`act_bit=16`
激活保持 fp16),调用 `torch_quant(w, quant_bit, quant_block, sym, awq, hqq)`;`--hqq` 是其中一个布尔,
走 `utils/hqq_quantizer.py` 的 proximal 迭代。MNN 导出代码里本就有 `fake_quant`:量化后当场反量化回
float 塞回 `linear.weight.data`——本模块即照抄这段:把 MNN 的量化(基础仿射 / HQQ)作用到参考
safetensors 权重上(量化再反量化),对同一组图跑两遍前向(float 权重 vs 反量化权重),**逐层比中间
激活**,定位量化从哪层起把激活拉崩、ViT 还是解码器更敏感。全程 torch,绕开 MNN 无-logits 限制。

诚实边界:测的是**权重量化**这一主因(MNN 激活本就是 fp16、误差主来源即权重 round),**不含** MNN
fp16 运行时累加 / 融合算子差异——那部分由行为级 `precision` 端到端覆盖。三者互补。基础仿射公式已
逐字锁定;HQQ 用经典 proximal 公式(与 MNN 同族),其精确超参 / 轴约定需在 GPU 机对照 MNN 源码 +
转换后某层存的 alpha 做一次 bit-level 对齐(见 dequant_weight_hqq 注释)。

设计要点:
  - torch **惰性 import**(仿 hf 后端),未装只在用本命令时报错;不需要 onnx / onnxruntime。
  - 复用 onnx_precision 的:指标(cosine/rel_l2/...)、逐层聚合、子模块发现、probe、取图、target 汇总。
  - LLM 用 `model(**inputs, output_hidden_states=True)` 原生逐层 hidden_states,无需 probe;
    ViT 视觉塔不吐 hidden_states,用 build_layer_probe 挂 hook 收集每 block + merger 输出。
  - 只量化 nn.Linear 权重(注意力 / MLP proj;ViT block linear + merger),不动 embedding/norm/bias
    (对齐 MNN:embed_bit=16、norm 不量化)。lm_head 不影响 hidden_states,天然跳过。
  - 两阶段:先跑完所有图的 float 激活并存下,再就地量化一次,跑第二遍即比即释(峰值只多一份 float 激活)。
  - 仅支持 Qwen3.5-VL 系列;参考权重复用 inference.hf.model_path。
  - 产物落 `<数据集>/<模型名>/quant-precision/quant_precision.{json,md}`。
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
from tqdm import tqdm

from .config import Config
from .results import store
# 复用 onnx_precision 已写好且测过的构件(指标 / 聚合 / 子模块发现 / probe / 取图 / target 汇总)。
from .onnx_precision import (
    layer_metric,
    rel_l2,
    _aggregate_layers,
    _find_visual,
    _find_decoder,
    build_layer_probe,
    torch_reference_activations,
    _param_dtype_device,
    _select_samples,
    _first_image_context,
    _target_summary,
)


# ---------------------------------------------------------------------------
# 量化数学(纯 torch,可脱离真模型单测)
#   基础仿射逐字复刻 MNN llmexport(transformers/llm/export/llmexport.py);
#   HQQ 复刻 utils/hqq_quantizer.py 的 proximal 迭代(经典 HQQ 公式,与 MNN 同族)。
# ---------------------------------------------------------------------------
def _quant_dequant_block_affine(w, bit: int, sym: bool):
    """对最后一维做「量化→反量化」的基础仿射(MNN hqq=False 路径),返回同形状反量化张量。

    非对称(sym=False,MNN 默认,带符号约定):
      offset=1<<(bit-1); clip_max=offset-1; clip_min=-offset   # 4bit->[-8,7]、8bit->[-128,127]
      scale=(max-min)/(clip_max-clip_min); q=clip(round((w-min)/scale)+clip_min, clip_min, clip_max)
      dq=(q-clip_min)*scale+min
    对称(sym=True):qmax=2^(bit-1)-1;scale=max(|w|)/qmax;q=clip(round(w/scale),-qmax,qmax);dq=q*scale
    scale==0 的块(常量权重)直接返回原值,避免除零。"""
    import torch

    if sym:
        qmax = (1 << (bit - 1)) - 1
        max_abs = w.abs().amax(dim=-1, keepdim=True)
        scale = max_abs / qmax
        safe = torch.where(scale == 0, torch.ones_like(scale), scale)
        q = torch.clamp(torch.round(w / safe), -qmax, qmax)
        dq = q * scale
    else:
        offset = 1 << (bit - 1)
        clip_max = offset - 1
        clip_min = -offset
        max_val = w.amax(dim=-1, keepdim=True)
        min_val = w.amin(dim=-1, keepdim=True)
        scale = (max_val - min_val) / (clip_max - clip_min)
        safe = torch.where(scale == 0, torch.ones_like(scale), scale)
        q = torch.round((w - min_val) / safe) + clip_min
        q = torch.clamp(q, clip_min, clip_max)
        dq = (q - clip_min) * scale + min_val
    return torch.where(scale == 0, w, dq)


def _iter_blocks(W, block: int):
    """按 (主体块, 尾块) 切分权重的输入通道维,产出 (切片对象, 视图) 供逐块量化后写回。

    W:[oc, ic];block<=0 或 >=ic => 整行一块(按输出通道)。主体 reshape 成 (oc, n_full, block)
    一次性向量化处理;ic % block != 0 的尾块单独处理(MNN 常见 block=128 能整除隐藏维,尾块是兜底)。"""
    oc, ic = W.shape
    bs = ic if (not block or block <= 0 or block >= ic) else block
    if bs >= ic:
        yield (slice(None), W)
        return
    n_full = ic // bs
    main_cols = n_full * bs
    yield ((slice(None), slice(0, main_cols)), W[:, :main_cols].reshape(oc, n_full, bs))
    if main_cols < ic:
        yield ((slice(None), slice(main_cols, ic)), W[:, main_cols:])


def dequant_weight_affine(W, bit: int, block: int, sym: bool):
    """逐字复刻 MNN llmexport 基础仿射(hqq=False):沿输入通道分块量化再反量化。W:[oc,ic] float。"""
    import torch

    dq = torch.empty_like(W)
    for sl, view in _iter_blocks(W, block):
        out = _quant_dequant_block_affine(view, bit, sym)
        dq[sl] = out.reshape(W[sl].shape)
    return dq


def _shrink_lp(x, beta: float, lp_norm: float):
    """HQQ 的 Lp 近端收缩算子(逐字对齐 MNN utils/hqq_quantizer.py `_shrink_lp_op`):
      lp_norm==1: sign(x)·relu(|x| - 1/β)
      否则:       sign(x)·relu(|x| - (1/β)·|x|^(lp_norm-1))"""
    import torch

    ax = x.abs()
    if lp_norm == 1:
        shrunk = torch.clamp(ax - 1.0 / beta, min=0.0)
    else:
        shrunk = torch.clamp(ax - (1.0 / beta) * ax.pow(lp_norm - 1), min=0.0)
    return torch.sign(x) * shrunk


def _hqq_block(w, bit: int, lp_norm: float, beta: float, kappa: float,
               iters: int, scale_only: bool):
    """对最后一维做 HQQ(半二次分裂)量化→反量化,返回反量化张量。

    经典 HQQ(mobiusml,MNN 的 hqq_quantizer.py 即据此):非对称无符号 [0, 2^bit-1],以 min/max 仿射
    为初值,迭代 iters 次:量化 W_q=clip(round(w/scale+zero)); 重构 W_r=(W_q-zero)*scale;
    对误差做 Lp 收缩 W_e=shrink(w-W_r,β); 更新 zero=mean(W_q-(w-W_e)/scale)(或 scale_only 变体用
    最小二乘更新 scale); β*=kappa。末次量化的反量化即结果。iters<=0 退化为纯 min/max 非对称仿射。

    注:MNN 内部把 scale 存成倒数(乘子)、zero 语义等价,本函数只产出**反量化权重**,与其数值一致;
    精确的轴 / clamp / 初值细节需在 GPU 机对照 MNN 源码 + 转换后某层存的 alpha 做 bit-level 对齐。"""
    import torch

    qmax = (1 << bit) - 1
    max_v = w.amax(dim=-1, keepdim=True)
    min_v = w.amin(dim=-1, keepdim=True)
    is_const = max_v == min_v          # 常量块:反量化应精确还原,不走量化网格(与仿射路径一致)
    scale = (max_v - min_v) / qmax
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    zero = torch.round(-min_v / scale)

    beta_i = beta
    for _ in range(max(0, iters)):
        w_q = torch.clamp(torch.round(w / scale + zero), 0, qmax)
        w_r = (w_q - zero) * scale
        w_e = _shrink_lp(w - w_r, beta_i, lp_norm)
        if scale_only:
            target = w - w_e
            wq0 = w_q - zero
            num = (target * wq0).sum(dim=-1, keepdim=True)
            den = (wq0 * wq0).sum(dim=-1, keepdim=True) + 1e-8
            new_scale = num / den
            scale = torch.where(new_scale.abs() < 1e-8, scale, new_scale)
        else:
            zero = torch.mean(w_q - (w - w_e) / scale, dim=-1, keepdim=True)
        beta_i *= kappa

    w_q = torch.clamp(torch.round(w / scale + zero), 0, qmax)
    dq = (w_q - zero) * scale
    return torch.where(is_const, w, dq)


def dequant_weight_hqq(W, bit: int, block: int, *, lp_norm: float, beta: float,
                       kappa: float, iters: int, scale_only: bool = False):
    """复刻 MNN HQQ:沿输入通道分块做 HQQ proximal 量化再反量化。W:[oc,ic] float。"""
    import torch

    dq = torch.empty_like(W)
    for sl, view in _iter_blocks(W, block):
        out = _hqq_block(view, bit, lp_norm, beta, kappa, iters, scale_only)
        dq[sl] = out.reshape(W[sl].shape)
    return dq


def _dequant_weight(W, spec: dict):
    """按 spec 分派:hqq=True 且 iters>0 走 HQQ,否则走基础仿射。W 会先转 float 再量化。"""
    Wf = W.float()
    if spec.get("hqq") and spec.get("iters", 0) > 0:
        return dequant_weight_hqq(
            Wf, spec["bit"], spec["block"],
            lp_norm=spec["lp_norm"], beta=spec["beta"], kappa=spec["kappa"],
            iters=spec["iters"], scale_only=spec.get("scale_only", False),
        )
    return dequant_weight_affine(Wf, spec["bit"], spec["block"], spec.get("sym", False))


# ---------------------------------------------------------------------------
# 就地量化 nn.Linear 权重 + 备份/恢复 + 逐层权重误差
# ---------------------------------------------------------------------------
def quantize_linears_(module: Any, spec: dict) -> dict:
    """就地把 module 内所有 nn.Linear.weight 换成反量化版本(按 spec 选仿射/HQQ)。

    返回 {子模块全限定名: 原 weight 克隆} 备份,供 restore_linears_ 精确还原。
    只动 nn.Linear.weight;embedding / norm / bias 不碰(对齐 MNN weight-only)。"""
    import torch

    backup: dict[str, Any] = {}
    with torch.no_grad():
        for name, m in module.named_modules():
            if isinstance(m, torch.nn.Linear):
                W = m.weight.data
                backup[name] = W.detach().clone()
                dq = _dequant_weight(W, spec).to(dtype=W.dtype, device=W.device)
                m.weight.data = dq
    return backup


def restore_linears_(module: Any, backup: dict) -> None:
    """用 quantize_linears_ 的备份把权重逐位还原(跑完必须调,否则污染共享模型)。"""
    import torch

    with torch.no_grad():
        for name, m in module.named_modules():
            if name in backup:
                m.weight.data = backup[name]


def per_layer_weight_error(layer_module: Any, spec: dict) -> float:
    """该层内各 nn.Linear 反量化后的 rel_l2 误差均值(解释「激活为何在此层发散」的权重侧指标)。"""
    import torch

    errs: list[float] = []
    with torch.no_grad():
        for m in layer_module.modules():
            if isinstance(m, torch.nn.Linear):
                W = m.weight.data.float()
                dq = _dequant_weight(W, spec)
                errs.append(rel_l2(dq.cpu().numpy(), W.cpu().numpy()))
    return float(sum(errs) / len(errs)) if errs else 0.0


def _llm_spec(qp) -> dict:
    return {
        "bit": qp.quant_bit, "block": qp.quant_block, "sym": qp.sym,
        "hqq": qp.hqq, "lp_norm": qp.hqq_lp_norm, "beta": qp.hqq_beta,
        "kappa": qp.hqq_kappa, "iters": qp.hqq_iters, "scale_only": qp.hqq_scale_only,
    }


def _vit_spec(qp) -> dict:
    return {
        "bit": qp.visual_quant_bit, "block": qp.visual_quant_block, "sym": qp.sym,
        "hqq": qp.hqq, "lp_norm": qp.hqq_lp_norm, "beta": qp.hqq_beta,
        "kappa": qp.hqq_kappa, "iters": qp.hqq_iters, "scale_only": qp.hqq_scale_only,
    }


def _attach_weight_err(layers: list[dict], weight_err: list[float]) -> None:
    for ly in layers:
        idx = ly["index"]
        ly["weight_rel_l2"] = round(weight_err[idx], 6) if idx < len(weight_err) else None


# ---------------------------------------------------------------------------
# 单 target(llm / vit)的 float vs 量化逐层对比
# ---------------------------------------------------------------------------
def _compare_target_llm(backend: Any, inputs_list: list, cfg: Config) -> dict:
    """解码器:两遍 `model(**inputs, output_hidden_states=True)`(float 权重 vs 量化权重),逐层比 hidden_states。"""
    import torch

    model = backend.model
    decoder = _find_decoder(model)
    layers = list(decoder.layers)
    names = [f"dec_layer_{i}" for i in range(len(layers))]
    _mdtype, mdevice = _param_dtype_device(decoder)
    qp = cfg.quant_precision
    spec = _llm_spec(qp)
    weight_err = [per_layer_weight_error(ly, spec) for ly in layers]

    def _forward_hs(inputs: dict) -> list["np.ndarray"]:
        inputs = {k: (v.to(mdevice) if torch.is_tensor(v) else v) for k, v in inputs.items()}
        with torch.inference_mode():
            out = model(**inputs, output_hidden_states=True, use_cache=False, return_dict=True)
        # hidden_states: (embeds, layer_0_out, ..., layer_{N-1}_out) -> [1:] 为逐层输出
        return [h.detach().to(torch.float32).cpu().numpy() for h in out.hidden_states[1:]]

    float_acts = [_forward_hs(inp)
                  for inp in tqdm(inputs_list, desc="quant-precision:llm(float)", unit="img", leave=False)]

    backup = quantize_linears_(decoder, spec)
    try:
        per_image: list[list[dict]] = []
        for i, inp in enumerate(tqdm(inputs_list, desc="quant-precision:llm(quant)", unit="img", leave=False)):
            qa = _forward_hs(inp)
            fa = float_acts[i]
            n = min(len(qa), len(fa), len(names))
            per_image.append([layer_metric(qa[l], fa[l]) for l in range(n)])
    finally:
        restore_linears_(decoder, backup)

    agg, first_div = _aggregate_layers(per_image, names, qp.cosine_min, qp.rel_l2_max)
    _attach_weight_err(agg, weight_err)
    return _target_summary("llm", agg, first_div, names, len(per_image))


def _compare_target_vit(backend: Any, inputs_list: list, cfg: Config) -> dict:
    """视觉塔:build_layer_probe 收集每 block + merger 输出,两遍(float vs 量化)逐层比。

    若 visual_quant_bit is None(MNN 默认视觉塔不量化)则跳过并说明,避免「全绿」误导。"""
    import torch

    qp = cfg.quant_precision
    if qp.visual_quant_bit is None:
        return {
            "available": False, "target": "vit",
            "reason": "ViT 未量化(visual_quant_bit=None);MNN 默认视觉塔不量化,无量化误差可测。"
                      "设 --visual-quant-bit 4/8 才对比。",
        }

    model = backend.model
    visual = _find_visual(model)
    layers = list(visual.blocks) + [visual.merger]
    names = [f"vit_block_{i}" for i in range(len(visual.blocks))] + ["vit_merger"]
    invoke = lambda base, pv, gthw: base(pv, gthw)  # noqa: E731
    mdtype, mdevice = _param_dtype_device(visual)
    spec = _vit_spec(qp)
    weight_err = [per_layer_weight_error(ly, spec) for ly in layers]

    def _args(inputs: dict) -> tuple:
        pv = inputs["pixel_values"].to(device=mdevice, dtype=mdtype)
        gthw = inputs["image_grid_thw"]
        gthw = gthw.to(mdevice) if torch.is_tensor(gthw) else gthw
        return (pv, gthw)

    def _probe_acts(inputs: dict) -> list["np.ndarray"]:
        probe = build_layer_probe(visual, layers, invoke)
        try:
            return torch_reference_activations(probe, _args(inputs))
        finally:
            probe.close()

    float_acts = [_probe_acts(inp)
                  for inp in tqdm(inputs_list, desc="quant-precision:vit(float)", unit="img", leave=False)]

    backup = quantize_linears_(visual, spec)
    try:
        per_image: list[list[dict]] = []
        for i, inp in enumerate(tqdm(inputs_list, desc="quant-precision:vit(quant)", unit="img", leave=False)):
            qa = _probe_acts(inp)
            fa = float_acts[i]
            n = min(len(qa), len(fa), len(names))
            per_image.append([layer_metric(qa[l], fa[l]) for l in range(n)])
    finally:
        restore_linears_(visual, backup)

    agg, first_div = _aggregate_layers(per_image, names, qp.cosine_min, qp.rel_l2_max)
    _attach_weight_err(agg, weight_err)
    return _target_summary("vit", agg, first_div, names, len(per_image))


# ---------------------------------------------------------------------------
# 入口:编排取图 -> 逐 target 对比 -> 聚合落盘
# ---------------------------------------------------------------------------
def _build_reference_backend(cfg: Config):
    """构造 float32/CPU 的 HFBackend 作参考(单次加载权重)。不强制 eager:本命令不导出 ONNX,
    float / 量化两遍用同一注意力实现,差异相减即抵消,故走 HF 默认(sdpa)即可、更快。"""
    from .inference.hf_backend import HFBackend

    ref_cfg = copy.deepcopy(cfg)
    ref_cfg.inference.backend = "hf"
    ref_cfg.inference.hf.dtype = cfg.quant_precision.dtype
    ref_cfg.inference.hf.device = cfg.quant_precision.device
    return HFBackend(ref_cfg)


def run_quant_precision(cfg: Config, hf_model: Optional[str] = None,
                        targets: Optional[list[str]] = None,
                        num_samples: Optional[int] = None) -> dict:
    """把 MNN 的权重量化模拟作用到参考模型,对一组图逐层对比 float 权重 vs 量化权重的激活。

    参考权重取 hf_model(覆盖)或 cfg.inference.hf.model_path。产物落
    <数据集>/<模型名>/quant-precision/quant_precision.{json,md},返回汇总 dict(含首「崩」层)。
    """
    if hf_model:
        cfg.inference.hf.model_path = hf_model
    qp = cfg.quant_precision
    targets = targets or list(qp.targets)
    num_samples = num_samples if num_samples is not None else qp.num_samples

    model_name = Path(cfg.inference.hf.model_path).expanduser().name if cfg.inference.hf.model_path else "hf-model"
    samples = _select_samples(cfg, num_samples)
    if not samples:
        raise RuntimeError(
            "未找到可用样本(需单图且含 <image>);请确认已 split 出 test.json 或数据源可读。"
        )

    backend = _build_reference_backend(cfg)
    # 复用 HFBackend.build_inputs 得到与行为级 precision 同源的预处理输入(CPU 张量)。
    # 逐样本容错:单张坏图不该整批崩溃(与 HFBackend.complete 一致);fail_fast 打开则照常抛。
    inputs_list = []
    used_ids = []
    skipped: list[dict] = []
    for s in samples:
        try:
            ctx, images = _first_image_context(s)
            inputs, _pil = backend.build_inputs(ctx, images, s.id)
        except Exception as e:  # noqa: BLE001
            if cfg.inference.fail_fast:
                raise
            skipped.append({"id": s.id, "reason": f"{type(e).__name__}: {e}"})
            continue
        inputs_list.append(inputs)
        used_ids.append(s.id)
    if not inputs_list:
        raise RuntimeError(
            f"所有样本预处理失败(共 {len(samples)} 个),无法逐层对比;"
            f"首个错误:{skipped[0]['reason'] if skipped else '未知'}")

    results: dict[str, dict] = {}
    for t in targets:
        try:
            if t == "llm":
                results[t] = _compare_target_llm(backend, inputs_list, cfg)
            elif t == "vit":
                results[t] = _compare_target_vit(backend, inputs_list, cfg)
            else:
                results[t] = {"available": False, "reason": f"未知 target: {t}(可选 llm/vit)"}
        except Exception as e:  # noqa: BLE001 - 某 target 失败不影响另一个
            if cfg.inference.fail_fast:
                raise
            results[t] = {"available": False, "reason": f"{type(e).__name__}: {e}"}

    summary = _assemble_summary(cfg, model_name, used_ids, results)
    if skipped:
        summary["skipped_samples"] = skipped
    report_dir = cfg.model_run_dir(model_name, "quant-precision")
    store.write_json(report_dir / "quant_precision.json", summary)
    store.write_text(report_dir / "quant_precision.md", _render_md(summary, cfg))
    summary["report_json"] = str(report_dir / "quant_precision.json")
    summary["report_md"] = str(report_dir / "quant_precision.md")
    return summary


def _assemble_summary(cfg: Config, model_name: str, sample_ids: list[str],
                      results: dict[str, dict]) -> dict:
    qp = cfg.quant_precision
    summary = {
        "model": model_name,
        "reference_model_path": cfg.inference.hf.model_path,
        "dtype": qp.dtype,
        "device": qp.device,
        "quant": {
            "quant_bit": qp.quant_bit, "quant_block": qp.quant_block, "sym": qp.sym,
            "hqq": qp.hqq, "hqq_lp_norm": qp.hqq_lp_norm, "hqq_beta": qp.hqq_beta,
            "hqq_kappa": qp.hqq_kappa, "hqq_iters": qp.hqq_iters,
            "hqq_scale_only": qp.hqq_scale_only,
            "visual_quant_bit": qp.visual_quant_bit, "visual_quant_block": qp.visual_quant_block,
        },
        "cosine_min": qp.cosine_min,
        "rel_l2_max": qp.rel_l2_max,
        "num_samples": len(sample_ids),
        "sample_ids": sample_ids,
        "compared_at": datetime.now(timezone.utc).isoformat(),
        "targets": results,
    }
    summary["flags"] = _build_flags(results, qp)
    return summary


def _quant_recipe_str(qp) -> str:
    base = f"{qp.quant_bit}bit/block={qp.quant_block}/{'sym' if qp.sym else 'asym'}"
    return f"HQQ({base}, iters={qp.hqq_iters})" if qp.hqq and qp.hqq_iters > 0 else f"affine({base})"


def _build_flags(results: dict[str, dict], qp) -> list[str]:
    """据各 target 首「崩」层生成人类可读判定(报告顶部醒目提示 + 误差定位)。"""
    flags: list[str] = []
    label = {"vit": "视觉塔(ViT)", "llm": "LLM 解码器"}
    recipe = _quant_recipe_str(qp)
    any_available = False
    any_diverged = False
    for t, r in results.items():
        name = label.get(t, t)
        if not r.get("available"):
            flags.append(f"⏭️ {name}:未对比({r.get('reason', '未知原因')})。")
            continue
        any_available = True
        fd = r.get("first_divergence_layer")
        worst = r.get("worst_layer") or {}
        if fd is not None:
            any_diverged = True
            flags.append(
                f"🔴 {name}[{recipe}]:第 {fd} 层(`{r.get('first_divergence_name')}`)起量化激活与 float 发散"
                f"(阈值 cosine≥{qp.cosine_min} 且 rel_l2≤{qp.rel_l2_max});"
                f"最差层 `{worst.get('name')}` min-cosine={worst.get('min_cosine')} / "
                f"max-rel_l2={worst.get('max_rel_l2')} / 权重误差={worst.get('weight_rel_l2')}。"
            )
        else:
            flags.append(
                f"✅ {name}[{recipe}]:全部 {r.get('num_layers')} 层在容差内"
                f"(最差层 `{worst.get('name')}` min-cosine={worst.get('min_cosine')} / "
                f"权重误差={worst.get('weight_rel_l2')})。"
            )
    if not any_available and not flags:
        flags.append("⚠️ 无可对比 target。")
    elif any_available and not any_diverged:
        flags.insert(0, "✅ 所有已对比子模型逐层激活均在容差内:该量化配方未见显著逐层退化。")
    return flags


# ---------------------------------------------------------------------------
# 人类可读报告
# ---------------------------------------------------------------------------
def _render_target_section(target: str, r: dict, cfg: Config) -> list[str]:
    label = {"vit": "视觉塔(ViT)", "llm": "LLM 解码器"}.get(target, target)
    out = [f"## {label}", ""]
    if not r.get("available"):
        out += [f"> ⏭️ 未对比:{r.get('reason', '未知原因')}", ""]
        return out
    fd = r.get("first_divergence_layer")
    out.append(
        f"- 层数: {r['num_layers']} · 对比图数: {r['num_images']} · "
        f"首「崩」层: {('第 ' + str(fd) + ' 层 `' + str(r.get('first_divergence_name')) + '`') if fd is not None else '无(全部达标)'}"
    )
    out.append("")
    layers = r.get("layers") or []
    cap = cfg.quant_precision.max_layers_in_report
    shown = layers[:cap]
    out += [
        "| 层 | 名称 | 均cosine | 最差cosine | 均rel_l2 | 最差rel_l2 | 权重rel_l2 | 判定 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for ly in shown:
        mark = "🔴" if ly["diverged"] else "✅"
        out.append(
            f"| {ly['index']} | `{ly['name']}` | {ly['mean_cosine']} | {ly['min_cosine']} "
            f"| {ly['mean_rel_l2']} | {ly['max_rel_l2']} | {ly.get('weight_rel_l2')} | {mark} |"
        )
    if len(layers) > cap:
        out.append(f"| … | (省略 {len(layers) - cap} 层;调大 max_layers_in_report 可展开) | | | | | | |")
    out.append("")
    return out


def _render_md(summary: dict, cfg: Config) -> str:
    q = summary["quant"]
    vit_q = f"{q['visual_quant_bit']}bit/block={q['visual_quant_block']}" if q["visual_quant_bit"] is not None else "未量化"
    lines = [
        f"# 量化逐层激活精度报告 — 模型 `{summary['model']}`",
        "",
        f"- 参考权重(safetensors): `{summary['reference_model_path']}`",
        f"- 精度/设备: {summary['dtype']} / {summary['device']}",
        f"- LLM 量化: {q['quant_bit']}bit · block={q['quant_block']} · {'对称' if q['sym'] else '非对称'} · "
        f"HQQ={'开' if q['hqq'] else '关'}(iters={q['hqq_iters']}, lp={q['hqq_lp_norm']}, β={q['hqq_beta']}, "
        f"κ={q['hqq_kappa']}, {'scale' if q['hqq_scale_only'] else 'zero'})",
        f"- ViT 量化: {vit_q}",
        f"- 达标阈值: cosine ≥ {summary['cosine_min']} 且 rel_l2 ≤ {summary['rel_l2_max']}",
        f"- 对比图数: {summary['num_samples']}",
        f"- 对比时间: {summary['compared_at']}",
        "",
        "> 模拟 MNN llmexport 的 **weight-only** 权重量化(基础仿射 / HQQ),对同一组图跑 float 权重 vs",
        "> 反量化权重两遍前向,逐层比中间激活。测的是**权重量化**这一主因(MNN 激活本为 fp16),",
        "> **不含** MNN fp16 运行时 / 融合算子差异——那部分由行为级 `precision` 端到端覆盖,三者互补。",
        "> `权重rel_l2` 列=该层各 Linear 反量化误差均值,解释「激活为何在此层发散」。",
        "",
        "## 判定",
        "",
    ]
    lines += [f"- {f}" for f in summary["flags"]]
    lines.append("")
    for t in ("llm", "vit"):
        if t in summary["targets"]:
            lines += _render_target_section(t, summary["targets"][t], cfg)
    for t, r in summary["targets"].items():
        if t not in ("llm", "vit"):
            lines += _render_target_section(t, r, cfg)
    return "\n".join(lines)
