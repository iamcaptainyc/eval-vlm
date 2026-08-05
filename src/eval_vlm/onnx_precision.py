"""onnx-precision 命令:torch/safetensors ↔ ONNX 的**逐层激活**数值级精度校验。

背景:MNN 效果退化的诊断链路是 `safetensors →(llmexport)→ ONNX → MNN`。行为级
`precision` 命令(读两份 predictions.jsonl 比文本)已覆盖 MNN vs safetensors,但受限于
「MNN 高层 API 无 logits」只能做到行为级。本模块补上管线里缺失的**数值级**一环:把模型的
**视觉塔(ViT)** 与 **LLM 解码器** 各自包装成「每层边界都是具名图输出」的 **probe ONNX**,
对一组图片做前向,逐层比对 onnxruntime(ONNX 候选)与 torch(safetensors 参考)的
**中间激活**,算 cosine / 相对 L2 误差,**定位第一个发散的层**。校验的是 torch→ONNX 导出
保真度(管线前半段)。

**probe 模式**(而非解析 llmexport 单块大图 + 名字映射):对**同一个** probe 模块,
① eager 跑一次得 torch 参考激活;② `torch.onnx.export` 导出、onnxruntime 跑一次得 ONNX
候选激活。两侧层定义完全同一,故差异纯粹来自导出/ORT 数值实现——对齐不脆弱、自包含。

**两端喂完全相同的预处理输入**(复用 HFBackend.build_inputs,即 precision 那套对齐
LlamaFactory 的预处理),否则逐层数值对比无意义。dtype 统一 **float32** 隔离 dtype 噪声。

设计要点:
  - torch / onnx / onnxruntime **惰性 import**(仿 hf/mnn 后端),未装只在用本命令时报错。
  - 指标(逐层逐图):cosine(展平)、rel_l2(‖候选-参考‖/‖参考‖)、max_abs_err、mean_abs_err。
  - **首发散层** = 最小的不达标层下标(仿 precision 的首发散字符);跨图取每层最差(任一图发散即算)。
  - 仅支持 Qwen 家族(visual.blocks[i] / 解码器 layers[i]);参考权重复用 inference.hf.model_path。
  - 产物落 `<数据集>/<模型名>/onnx-precision/` 下 onnx_precision.{json,md};probe onnx 默认写临时目录跑完删。
"""
from __future__ import annotations

import copy
import contextlib
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .config import Config
from .data.loader import load_samples
from .results import store

_INTERNAL_PLACEHOLDER = "<image>"


# ---------------------------------------------------------------------------
# 纯 numpy 指标核心(不依赖 torch/onnx,可单独单测)
# ---------------------------------------------------------------------------
def _flat(x: "np.ndarray") -> "np.ndarray":
    return np.asarray(x, dtype=np.float64).reshape(-1)


def cosine(cand: "np.ndarray", ref: "np.ndarray") -> float:
    """展平后的余弦相似度(1.0=方向完全一致)。任一为零向量则:两者都零=1.0,否则 0.0。"""
    a, b = _flat(cand), _flat(ref)
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 and nb == 0.0:
        return 1.0
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def rel_l2(cand: "np.ndarray", ref: "np.ndarray") -> float:
    """相对 L2 误差 ‖候选-参考‖ / ‖参考‖(参考=torch;越小越好)。参考为零向量时退化为绝对 L2。"""
    a, b = _flat(cand), _flat(ref)
    nb = float(np.linalg.norm(b))
    diff = float(np.linalg.norm(a - b))
    if nb == 0.0:
        return diff
    return diff / nb


def max_abs_err(cand: "np.ndarray", ref: "np.ndarray") -> float:
    a, b = _flat(cand), _flat(ref)
    return float(np.max(np.abs(a - b))) if a.size else 0.0


def mean_abs_err(cand: "np.ndarray", ref: "np.ndarray") -> float:
    a, b = _flat(cand), _flat(ref)
    return float(np.mean(np.abs(a - b))) if a.size else 0.0


def layer_metric(cand: "np.ndarray", ref: "np.ndarray") -> dict:
    """单层单图:候选(ONNX)vs 参考(torch)四项误差。"""
    return {
        "cosine": cosine(cand, ref),
        "rel_l2": rel_l2(cand, ref),
        "max_abs_err": max_abs_err(cand, ref),
        "mean_abs_err": mean_abs_err(cand, ref),
    }


def _diverged(cosine_v: float, rel_l2_v: float, cosine_min: float, rel_l2_max: float) -> bool:
    """达标判定的反面:cosine 低于阈值 或 相对 L2 高于阈值 即判该(层,图)发散。"""
    return (cosine_v < cosine_min) or (rel_l2_v > rel_l2_max)


def _aggregate_layers(per_image: list[list[dict]], names: list[str],
                      cosine_min: float, rel_l2_max: float) -> tuple[list[dict], Optional[int]]:
    """把「逐图 × 逐层」指标按层聚合(跨图),并定位首发散层。

    per_image[i][l] = 第 i 图第 l 层的 layer_metric。每层取跨图**最差**(min cosine / max rel_l2)
    作为该层是否发散的判据(任一图发散即算发散,便于抓最坏情况)。返回 (逐层汇总, 首发散层下标)。
    """
    if not per_image:
        return [], None
    num_layers = len(per_image[0])
    layers: list[dict] = []
    first_div: Optional[int] = None
    for l in range(num_layers):
        cos = [img[l]["cosine"] for img in per_image]
        rl2 = [img[l]["rel_l2"] for img in per_image]
        mxa = [img[l]["max_abs_err"] for img in per_image]
        mna = [img[l]["mean_abs_err"] for img in per_image]
        min_cos, max_rl2 = min(cos), max(rl2)
        diverged = _diverged(min_cos, max_rl2, cosine_min, rel_l2_max)
        layers.append({
            "index": l,
            "name": names[l] if l < len(names) else f"layer_{l}",
            "mean_cosine": round(sum(cos) / len(cos), 6),
            "min_cosine": round(min_cos, 6),
            "mean_rel_l2": round(sum(rl2) / len(rl2), 6),
            "max_rel_l2": round(max_rl2, 6),
            "mean_max_abs_err": round(sum(mxa) / len(mxa), 6),
            "mean_abs_err": round(sum(mna) / len(mna), 6),
            "diverged": diverged,
        })
        if diverged and first_div is None:
            first_div = l
    return layers, first_div


# ---------------------------------------------------------------------------
# Qwen 子模块发现(兼容不同 transformers 版本的层级布局)
# ---------------------------------------------------------------------------
def _resolve_path(root: Any, path: tuple[str, ...]) -> Any:
    obj = root
    for p in path:
        obj = getattr(obj, p, None)
        if obj is None:
            return None
    return obj


def _find_visual(model: Any) -> Any:
    """找视觉塔(有 .blocks):Qwen2-VL 是 model.visual;Qwen2.5/3-VL 是 model.model.visual。"""
    for path in (("visual",), ("model", "visual")):
        v = _resolve_path(model, path)
        if v is not None and hasattr(v, "blocks"):
            return v
    raise RuntimeError(
        "未找到含 .blocks 的视觉塔(model.visual / model.model.visual);"
        "onnx-precision 目前仅支持 Qwen2-VL / Qwen2.5-VL / Qwen3-VL 家族。"
    )


def _find_decoder(model: Any) -> Any:
    """找文本解码器(有 .layers):老版是 model.model;新版是 model.model.language_model。"""
    for path in (("model",), ("model", "language_model"), ("language_model",),
                 ("model", "model")):
        d = _resolve_path(model, path)
        if d is not None and hasattr(d, "layers"):
            return d
    raise RuntimeError(
        "未找到含 .layers 的文本解码器(model.model / model.model.language_model);"
        "onnx-precision 目前仅支持 Qwen 家族。"
    )


def _rope_index_fn(model: Any):
    """取 get_rope_index(算 Qwen 3D mrope 的 position_ids);老版在顶层、新版在 model.model 上。"""
    for owner in (model, getattr(model, "model", None)):
        fn = getattr(owner, "get_rope_index", None)
        if callable(fn):
            return fn
    return None


# ---------------------------------------------------------------------------
# probe 包装器 + 导出 + onnxruntime(惰性依赖 torch/onnx/onnxruntime)
# ---------------------------------------------------------------------------
def build_layer_probe(base: Any, layers: list, invoke):
    """把 base 子模块包装成「forward 返回各层输出元组」的 probe(torch.nn.Module)。

    对 base 的每个子层挂 forward hook 收集输出;forward 用 invoke(base, *args) 触发一次前向
    (invoke 负责把张量入参按各自 base 的签名映射成 kwargs,如解码器要 inputs_embeds=/use_cache=False)。
    同一个 probe 既给 eager 参考、又给 torch.onnx.export——两端层定义完全一致,差异只来自导出/ORT。
    layer 输出若为元组(如解码器层返回 (hidden, ...))取第 0 项。惰性 import torch。
    """
    import torch

    class _LayerProbe(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base = base
            self._invoke = invoke
            self._captured: list = []
            # 记录 hook 句柄,用完 close() 摘除;否则每图新建 probe 会把 hook 累加到
            # 共享的 blocks/layers 上(O(N²) 触发 + 中间激活迟迟不释放)。
            self._handles = [layer.register_forward_hook(self._hook) for layer in layers]

        def _hook(self, module, inputs, output):
            t = output[0] if isinstance(output, (tuple, list)) else output
            self._captured.append(t)

        def forward(self, *args):
            self._captured = []
            self._invoke(self.base, *args)
            return tuple(self._captured)

        def close(self):
            for h in self._handles:
                h.remove()
            self._handles = []

    return _LayerProbe()


def torch_reference_activations(probe: Any, args: tuple) -> list["np.ndarray"]:
    """eager 跑一次 probe,取各层输出为 numpy(torch/safetensors 参考激活)。"""
    import torch

    with torch.inference_mode():
        outs = probe(*args)
    return [o.detach().to(torch.float32).cpu().numpy() for o in outs]


def export_probe_onnx(probe: Any, args: tuple, path: Path,
                      input_names: list[str], output_names: list[str], opset: int) -> None:
    """把 probe 导成 ONNX,每层边界即具名输出。按当前图片形状固定导出(不设 dynamic axes),
    对 Qwen 视觉塔那种含动态控制流的子模块最稳(每张图重导一次,num_samples 通常很小)。"""
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        # dynamo=False:显式走经典 TorchScript 导出器。我们的 probe 靠 forward hook 收集每层输出、
        # 按 output_names 逐位命名边界张量——这正是经典导出器的行为;新版 dynamo 导出器
        # (torch≥2.9 默认)对 output_names 语义不同且需额外装 onnxscript。
        torch.onnx.export(
            probe, args, str(path),
            input_names=input_names, output_names=output_names,
            opset_version=opset, do_constant_folding=True,
            dynamo=False,
        )


def ort_activations(onnx_path: Path, output_names: list[str],
                    feed: dict) -> list["np.ndarray"]:
    """用 onnxruntime(CPU)跑 probe onnx,按 output_names 顺序取回各层激活。

    只喂图里真正声明的输入:Qwen 视觉塔 trace 时会把 grid_thw(乃至 position_ids/attention_mask)
    的值折成常量、该输入被导出器裁成死输入,若仍按名喂入 ORT 会报 Invalid Feed Input Name。
    按每图固定形状导出,被折叠的常量正好对应该图,故过滤掉多余 feed 键是正确且充分的。
    """
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    declared = {i.name for i in sess.get_inputs()}
    feed = {k: v for k, v in feed.items() if k in declared}
    outs = sess.run(output_names, feed)
    return [np.asarray(o) for o in outs]


# ---------------------------------------------------------------------------
# 单 target(vit / llm)的逐图导出 + 对比
# ---------------------------------------------------------------------------
def _param_dtype_device(module: Any):
    """取子模块首个参数的 (dtype, device),作为该 target 的输入统一目标。
    无参数(玩具兜底)时回落 float32/CPU。"""
    import torch

    try:
        p = next(module.parameters())
        return p.dtype, p.device
    except StopIteration:
        return torch.float32, torch.device("cpu")


def _compare_target_vit(backend: Any, inputs_list: list, cfg: Config,
                        tmpdir: Path, keep_onnx: bool) -> dict:
    """视觉塔:probe = model.visual,层 = blocks[i] + merger;输入 (pixel_values, grid_thw)。"""
    import torch

    model = backend.model
    visual = _find_visual(model)
    layers = list(visual.blocks) + [visual.merger]
    names = [f"vit_block_{i}" for i in range(len(visual.blocks))] + ["vit_merger"]
    invoke = lambda base, pv, gthw: base(pv, gthw)  # noqa: E731
    mdtype, mdevice = _param_dtype_device(visual)

    per_image: list[list[dict]] = []
    for si, inputs in enumerate(inputs_list):
        # 输入随模型 dtype/device 走:浮点张量转模型精度、整型(grid_thw)只搬设备,
        # 否则 dtype!=float32 或 device!=cpu 时会 matmul/设备不匹配崩掉。
        pv = inputs["pixel_values"].to(device=mdevice, dtype=mdtype)
        gthw = inputs["image_grid_thw"]
        gthw = gthw.to(mdevice) if torch.is_tensor(gthw) else gthw
        args = (pv, gthw)
        probe = build_layer_probe(visual, layers, invoke)
        try:
            ref = torch_reference_activations(probe, args)
            onnx_path = tmpdir / f"vit_{si}.onnx"
            export_probe_onnx(probe, args, onnx_path,
                              input_names=["pixel_values", "grid_thw"],
                              output_names=names, opset=cfg.onnx_precision.opset)
        finally:
            probe.close()
        feed = {"pixel_values": pv.cpu().numpy(),
                "grid_thw": gthw.cpu().numpy() if torch.is_tensor(gthw) else np.asarray(gthw)}
        cand = ort_activations(onnx_path, names, feed)
        if not keep_onnx:
            onnx_path.unlink(missing_ok=True)
        if not (len(cand) == len(ref) == len(names)):
            raise RuntimeError(
                f"vit 层数不一致:ONNX 输出 {len(cand)} / torch 参考 {len(ref)} / 命名 {len(names)}")
        per_image.append([layer_metric(c, r) for c, r in zip(cand, ref)])

    oc = cfg.onnx_precision
    agg, first_div = _aggregate_layers(per_image, names, oc.cosine_min, oc.rel_l2_max)
    return _target_summary("vit", agg, first_div, names, len(per_image))


def _compare_target_llm(backend: Any, inputs_list: list, cfg: Config,
                        tmpdir: Path, keep_onnx: bool) -> dict:
    """解码器:torch 端先算合并后的 inputs_embeds(output_hidden_states[0])与 3D mrope
    position_ids(get_rope_index),再把 model.model(解码器)包成 probe,逐层对比。"""
    import torch

    model = backend.model
    decoder = _find_decoder(model)
    rope_fn = _rope_index_fn(model)
    layers = list(decoder.layers)
    names = [f"dec_layer_{i}" for i in range(len(layers))]
    invoke = lambda base, emb, am, pid: base(  # noqa: E731
        inputs_embeds=emb, attention_mask=am, position_ids=pid,
        use_cache=False, return_dict=True)
    mdtype, mdevice = _param_dtype_device(decoder)

    per_image: list[list[dict]] = []
    for si, inputs in enumerate(inputs_list):
        # 先把整份输入搬到模型 device(build_inputs 产出的是 CPU 张量;device!=cpu 时不搬会崩)。
        inputs = {k: (v.to(mdevice) if torch.is_tensor(v) else v) for k, v in inputs.items()}
        with torch.inference_mode():
            out = model(**inputs, output_hidden_states=True, use_cache=False, return_dict=True)
        # hidden_states[0] 已是模型 dtype/device;转成解码器参数精度即可(默认 float32 时为 no-op)。
        embeds = out.hidden_states[0].to(dtype=mdtype)     # 合并视觉后、进第 0 层前的输入嵌入
        attn = inputs.get("attention_mask")
        if attn is None:
            attn = torch.ones(embeds.shape[:2], dtype=torch.long, device=mdevice)
        gthw = inputs.get("image_grid_thw")
        if rope_fn is not None:
            pos = rope_fn(input_ids=inputs["input_ids"], image_grid_thw=gthw,
                          attention_mask=attn)
            pos = pos[0] if isinstance(pos, (tuple, list)) else pos
        else:
            # 无 get_rope_index 时退化为 1D 顺序位置(仅玩具/非 Qwen 兜底)。
            seq = embeds.shape[1]
            pos = torch.arange(seq, device=mdevice).view(1, 1, seq).expand(
                3, embeds.shape[0], seq).contiguous()

        args = (embeds, attn, pos)
        probe = build_layer_probe(decoder, layers, invoke)
        try:
            ref = torch_reference_activations(probe, args)
            onnx_path = tmpdir / f"llm_{si}.onnx"
            export_probe_onnx(probe, args, onnx_path,
                              input_names=["inputs_embeds", "attention_mask", "position_ids"],
                              output_names=names, opset=cfg.onnx_precision.opset)
        finally:
            probe.close()
        feed = {
            "inputs_embeds": embeds.cpu().numpy(),
            "attention_mask": attn.cpu().numpy(),
            "position_ids": pos.cpu().numpy(),
        }
        cand = ort_activations(onnx_path, names, feed)
        if not keep_onnx:
            onnx_path.unlink(missing_ok=True)
        if not (len(cand) == len(ref) == len(names)):
            raise RuntimeError(
                f"llm 层数不一致:ONNX 输出 {len(cand)} / torch 参考 {len(ref)} / 命名 {len(names)}")
        per_image.append([layer_metric(c, r) for c, r in zip(cand, ref)])

    oc = cfg.onnx_precision
    agg, first_div = _aggregate_layers(per_image, names, oc.cosine_min, oc.rel_l2_max)
    return _target_summary("llm", agg, first_div, names, len(per_image))


def _target_summary(target: str, layers: list[dict], first_div: Optional[int],
                    names: list[str], num_images: int) -> dict:
    """单 target 的汇总:首发散层 + 最差层(按 min_cosine 升序)+ 逐层明细。"""
    worst = min(layers, key=lambda x: x["min_cosine"]) if layers else None
    return {
        "available": True,
        "target": target,
        "num_layers": len(layers),
        "num_images": num_images,
        "first_divergence_layer": first_div,
        "first_divergence_name": names[first_div] if first_div is not None and first_div < len(names) else None,
        "worst_layer": worst,
        "layers": layers,
    }


# ---------------------------------------------------------------------------
# 入口:编排取图 -> 逐 target 对比 -> 聚合落盘
# ---------------------------------------------------------------------------
def _first_image_context(sample) -> tuple[list, list[str]]:
    """取样本里第一个含 <image> 的 user 轮之前(含该轮)的上下文 + 图片列表(单图)。

    与 runner 逐轮 rollout 不同,这里只需要「喂进视觉塔/解码器」的一份前向输入,故取到含图那轮即可。
    """
    ctx: list = []
    for turn in sample.turns:
        ctx.append(turn)
        if _INTERNAL_PLACEHOLDER in turn.content:
            break
    return ctx, list(sample.images)


def _select_samples(cfg: Config, num_samples: int) -> list:
    """从 test.json 取前 num_samples 个「单图、含 <image>」样本。"""
    src = cfg.test_path if cfg.test_path.exists() else cfg.source_path
    samples = load_samples(cfg, source=src)
    picked = []
    for s in samples:
        if len(s.images) != 1:
            continue
        if not any(_INTERNAL_PLACEHOLDER in t.content for t in s.turns):
            continue
        picked.append(s)
        if len(picked) >= num_samples:
            break
    return picked


def _build_reference_backend(cfg: Config):
    """构造 float32/CPU 的 HFBackend 作 torch 参考 + ONNX 导出源(单次加载权重)。

    深拷贝 cfg 并把 backend 强制为 hf、dtype=float32、device 用配置里的(默认 cpu),避免污染调用方 cfg。
    """
    from .inference.hf_backend import HFBackend

    ref_cfg = copy.deepcopy(cfg)
    ref_cfg.inference.backend = "hf"
    ref_cfg.inference.hf.dtype = cfg.onnx_precision.dtype
    ref_cfg.inference.hf.device = cfg.onnx_precision.device
    return HFBackend(ref_cfg)


def run_onnx_precision(cfg: Config, hf_model: Optional[str] = None,
                       targets: Optional[list[str]] = None,
                       num_samples: Optional[int] = None,
                       keep_onnx: Optional[bool] = None) -> dict:
    """把参考模型的 ViT / LLM 解码器导成 probe ONNX,对一组图逐层对比 torch vs onnxruntime。

    参考权重取 hf_model(覆盖)或 cfg.inference.hf.model_path。产物落
    <数据集>/<模型名>/onnx-precision/onnx_precision.{json,md},返回汇总 dict(含首发散层)。
    """
    if hf_model:
        cfg.inference.hf.model_path = hf_model
    oc = cfg.onnx_precision
    targets = targets or list(oc.targets)
    num_samples = num_samples if num_samples is not None else oc.num_samples
    keep_onnx = oc.keep_onnx if keep_onnx is None else keep_onnx

    model_name = Path(cfg.inference.hf.model_path).expanduser().name if cfg.inference.hf.model_path else "hf-model"
    samples = _select_samples(cfg, num_samples)
    if not samples:
        raise RuntimeError(
            "未找到可用样本(需单图且含 <image>);请确认已 split 出 test.json 或数据源可读。"
        )

    backend = _build_reference_backend(cfg)
    # 复用 HFBackend.build_inputs 得到与行为级 precision 同源的预处理输入(CPU 张量)。
    # 逐样本容错:单张坏图(缺失/损坏)不该整批崩溃(与 HFBackend.complete 的逐条记错一致);
    # fail_fast 打开则照常抛出。
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
    if keep_onnx:
        persist = cfg.model_run_dir(model_name, "onnx-precision") / "probe_onnx"
        persist.mkdir(parents=True, exist_ok=True)
        dir_ctx = contextlib.nullcontext(str(persist))
    else:
        dir_ctx = tempfile.TemporaryDirectory(prefix="eval_vlm_onnx_")
    with dir_ctx as td:
        tmpdir = Path(td)
        for t in targets:
            try:
                if t == "vit":
                    results[t] = _compare_target_vit(backend, inputs_list, cfg, tmpdir, keep_onnx)
                elif t == "llm":
                    results[t] = _compare_target_llm(backend, inputs_list, cfg, tmpdir, keep_onnx)
                else:
                    results[t] = {"available": False, "reason": f"未知 target: {t}(可选 vit/llm)"}
            except Exception as e:  # noqa: BLE001 - 某 target 导出/对比失败不影响另一个
                if cfg.inference.fail_fast:
                    raise
                results[t] = {"available": False,
                              "reason": f"{type(e).__name__}: {e}"}

    summary = _assemble_summary(cfg, model_name, used_ids, results)
    if skipped:
        summary["skipped_samples"] = skipped
    report_dir = cfg.model_run_dir(model_name, "onnx-precision")
    store.write_json(report_dir / "onnx_precision.json", summary)
    store.write_text(report_dir / "onnx_precision.md", _render_md(summary, cfg))
    summary["report_json"] = str(report_dir / "onnx_precision.json")
    summary["report_md"] = str(report_dir / "onnx_precision.md")
    return summary


def _assemble_summary(cfg: Config, model_name: str, sample_ids: list[str],
                      results: dict[str, dict]) -> dict:
    oc = cfg.onnx_precision
    summary = {
        "model": model_name,
        "reference_model_path": cfg.inference.hf.model_path,
        "dtype": oc.dtype,
        "device": oc.device,
        "opset": oc.opset,
        "cosine_min": oc.cosine_min,
        "rel_l2_max": oc.rel_l2_max,
        "num_samples": len(sample_ids),
        "sample_ids": sample_ids,
        "compared_at": datetime.now(timezone.utc).isoformat(),
        "targets": results,
    }
    summary["flags"] = _build_flags(results, oc.cosine_min, oc.rel_l2_max)
    return summary


def _build_flags(results: dict[str, dict], cosine_min: float, rel_l2_max: float) -> list[str]:
    """据各 target 首发散层生成人类可读判定(报告顶部醒目提示 + 误差定位)。"""
    flags: list[str] = []
    label = {"vit": "视觉塔(ViT)", "llm": "LLM 解码器"}
    any_available = False
    any_diverged = False
    for t, r in results.items():
        name = label.get(t, t)
        if not r.get("available"):
            flags.append(f"⏭️ {name}:未对比({r.get('reason', '未知原因')})。")
            continue
        any_available = True
        fd = r.get("first_divergence_layer")
        if fd is not None:
            any_diverged = True
            worst = r.get("worst_layer") or {}
            flags.append(
                f"🔴 {name}:第 {fd} 层(`{r.get('first_divergence_name')}`)起 ONNX 与 torch 发散"
                f"(阈值 cosine≥{cosine_min} 且 rel_l2≤{rel_l2_max});"
                f"最差层 `{worst.get('name')}` min-cosine={worst.get('min_cosine')} / "
                f"max-rel_l2={worst.get('max_rel_l2')}——torch→ONNX 导出保真度在此层已破。"
            )
        else:
            worst = r.get("worst_layer") or {}
            flags.append(
                f"✅ {name}:全部 {r.get('num_layers')} 层达标"
                f"(最差层 `{worst.get('name')}` min-cosine={worst.get('min_cosine')})。"
            )
    if not any_available and not flags:
        flags.append("⚠️ 无可对比 target。")
    elif any_available and not any_diverged:
        flags.insert(0, "✅ 所有已对比子模型逐层均在容差内:torch→ONNX 导出数值保真。")
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
    out.append(f"- 层数: {r['num_layers']} · 对比图数: {r['num_images']} · "
               f"首发散层: {('第 ' + str(fd) + ' 层 `' + str(r.get('first_divergence_name')) + '`') if fd is not None else '无(全部达标)'}")
    out.append("")
    layers = r.get("layers") or []
    cap = cfg.onnx_precision.max_layers_in_report
    shown = layers[:cap]
    out += [
        "| 层 | 名称 | 均cosine | 最差cosine | 均rel_l2 | 最差rel_l2 | 均max_abs | 判定 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for ly in shown:
        mark = "🔴" if ly["diverged"] else "✅"
        out.append(
            f"| {ly['index']} | `{ly['name']}` | {ly['mean_cosine']} | {ly['min_cosine']} "
            f"| {ly['mean_rel_l2']} | {ly['max_rel_l2']} | {ly['mean_max_abs_err']} | {mark} |"
        )
    if len(layers) > cap:
        out.append(f"| … | (省略 {len(layers) - cap} 层;调大 max_layers_in_report 可展开) | | | | | | |")
    out.append("")
    return out


def _render_md(summary: dict, cfg: Config) -> str:
    lines = [
        f"# ONNX 逐层激活精度报告 — 模型 `{summary['model']}`",
        "",
        f"- 参考权重(safetensors): `{summary['reference_model_path']}`",
        f"- 精度/设备: {summary['dtype']} / {summary['device']} · opset {summary['opset']}",
        f"- 达标阈值: cosine ≥ {summary['cosine_min']} 且 rel_l2 ≤ {summary['rel_l2_max']}",
        f"- 对比图数: {summary['num_samples']}",
        f"- 对比时间: {summary['compared_at']}",
        "",
        "> 校验 torch(safetensors)→ ONNX 导出的**逐层数值保真度**(管线 safetensors→ONNX→MNN 前半段)。",
        "> 两端喂完全相同的预处理输入、统一 float32,故差异纯来自导出/onnxruntime 实现。",
        "> 与行为级 `precision`(MNN vs safetensors 文本一致性)互补:此处早发散层=导出/算子问题定位点。",
        "",
        "## 判定",
        "",
    ]
    lines += [f"- {f}" for f in summary["flags"]]
    lines.append("")
    for t in ("vit", "llm"):
        if t in summary["targets"]:
            lines += _render_target_section(t, summary["targets"][t], cfg)
    # 其它非标准 target(理论上不会有,兜底)
    for t, r in summary["targets"].items():
        if t not in ("vit", "llm"):
            lines += _render_target_section(t, r, cfg)
    return "\n".join(lines)
