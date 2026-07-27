"""precision 命令:量化 mnn(转换后)相对 hf(转换前)的**行为级精度误差**。

背景:MNN 端侧推理与 LlamaFactory 训练态推理有偏差,但 MNN 高层 API 拿不到 logits,
无法直接做张量级 cosine/KL。本模块改测**行为级**差异——两端在同一批样本上的
文本输出一致性,并辅以**输入对齐审计**(prompt token 数 / 图片 resize 像素),
把误差定位到「预处理 / prefill / 解码」某一环。

**解耦设计**:不在此进程跑模型,只读两份已生成的 predictions.jsonl
(候选=MNN 目录、参考=HF 目录,均由 `pred` 产出),按 (id, turn) 对齐后对比。
因此候选与参考可在不同机器上分别产出(MNN 边缘机 / HF GPU 机)。

产物落在候选(MNN)模型子目录:precision.json(机器可读)+ precision.md(人读)。

指标与定位:
  - 输出一致率 / 平均 char token-F1 / 平均编辑相似度 —— 行为漂移程度;
  - 首个发散字符位置 —— 早发散(靠前)= prefill/预处理/算子疑点;晚发散 = 解码噪声;
  - prompt token 数 delta / 图片像素 delta —— 预处理是否对齐(命中 VLM 最常见坑)。
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from . import compare
from .config import Config
from .data.schema import Prediction, Sample
from .results import store
from .scoring import get_scorer

# 逐字编辑距离的长度上限:描述文本一般几百字,但异常复读可能刷到 max_tokens。
# 超过则各自截断到该长度再算(O(n*m) 保护),对早/晚发散判断无实质影响。
_EDIT_LEN_CAP = 4000


# ---------------------------------------------------------------------------
# 文本级指标
# ---------------------------------------------------------------------------
def _levenshtein(a: str, b: str) -> int:
    """标准编辑距离(两行 DP)。超长文本各自截断到 _EDIT_LEN_CAP 再算。"""
    a, b = a[:_EDIT_LEN_CAP], b[:_EDIT_LEN_CAP]
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _edit_similarity(a: str, b: str) -> float:
    """1 - 归一化编辑距离(1.0=完全一致,0.0=完全不同)。"""
    if not a and not b:
        return 1.0
    dist = _levenshtein(a, b)
    denom = max(len(a[:_EDIT_LEN_CAP]), len(b[:_EDIT_LEN_CAP]), 1)
    return 1.0 - dist / denom


def _first_divergence(a: str, b: str) -> Optional[int]:
    """两串首个不同字符的下标;完全相同返回 None。

    一个是另一个的前缀时,返回较短串的长度(即前缀之后的第一个位置)。
    早发散(小下标)提示 prefill/预处理/算子问题;晚发散提示解码噪声。
    """
    if a == b:
        return None
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


# ---------------------------------------------------------------------------
# 对齐元数据(从 Prediction.raw 取,兼容不同后端的键名)
# ---------------------------------------------------------------------------
def _alignment_value(raw: Optional[dict], *keys: str) -> Optional[float]:
    """按优先顺序从 raw 取第一个存在的键值(不同后端键名不同);都没有返回 None。"""
    if not raw:
        return None
    for k in keys:
        v = raw.get(k)
        if v is not None:
            return v
    return None


def _prompt_tokens(raw: Optional[dict]) -> Optional[float]:
    return _alignment_value(raw, "prompt_token_count", "prompt_len")


def _image_pixels(raw: Optional[dict]) -> Optional[float]:
    px = _alignment_value(raw, "image_pixels", "resized_pixels")
    if px is None:
        mp = _alignment_value(raw, "pixels_mp")   # MNN 统计的百万像素兜底
        if mp is not None:
            px = float(mp) * 1_000_000
    return px


# ---------------------------------------------------------------------------
# 对比主逻辑
# ---------------------------------------------------------------------------
def _resolve_names(cfg: Config, candidate: Optional[str],
                   reference: Optional[str]) -> tuple[str, str]:
    """定候选(MNN)/参考(HF)**模型名**:显式 > precision 配置 > 后端块推断。

    产物路径由 <数据集>/<模型名>/<后端类型>/ 组成,候选固定走 mnn 子目录、
    参考固定走 hf 子目录(见 compare_precision),故这里只需定模型名。
    """
    cand = candidate or cfg.precision.candidate_dir
    if not cand:
        cp = cfg.inference.mnn.config_path
        cand = Path(cp).expanduser().parent.name if cp else "mnn-model"
    ref = reference or cfg.precision.reference_dir
    if not ref:
        mp = cfg.inference.hf.model_path
        ref = Path(mp).expanduser().name if mp else "hf-model"
    return cand, ref


def compare_precision(cfg: Config, candidate: Optional[str] = None,
                      reference: Optional[str] = None,
                      datadir_format: bool = False) -> dict:
    """对比候选(MNN)与参考(HF)两份预测,算行为级精度误差,落盘报告并返回汇总。

    datadir_format=True:两端预测由 pred --datadir 产出(LlamaFactory 对话格式,输出在
    messages 最后一个 assistant 轮),按此解析;否则按 run / pred --dataset 的
    Prediction 格式(顶层 prediction/turn)解析。
    """
    cand_name, ref_name = _resolve_names(cfg, candidate, reference)
    cand_path = cfg.predictions_path_for(cand_name, "mnn")
    ref_path = cfg.predictions_path_for(ref_name, "hf")
    if not cand_path.exists():
        raise FileNotFoundError(
            f"未找到候选预测 {cand_path};请先运行 "
            f"`eval-vlm pred --dataset <ds> --backend mnn ...` 产出它。"
        )
    if not ref_path.exists():
        raise FileNotFoundError(
            f"未找到参考预测 {ref_path};请先运行 "
            f"`eval-vlm pred --dataset <ds> --backend hf --hf-model <目录>` 产出它。"
        )

    cand_preds = store.load_predictions(cand_path, datadir_format=datadir_format)
    ref_preds = store.load_predictions(ref_path, datadir_format=datadir_format)
    cand_by_key = {(p.id, p.turn): p for p in cand_preds}
    ref_by_key = {(p.id, p.turn): p for p in ref_preds}

    common = sorted(set(cand_by_key) & set(ref_by_key))
    only_cand = len(set(cand_by_key) - set(ref_by_key))
    only_ref = len(set(ref_by_key) - set(cand_by_key))

    f1_scorer = get_scorer("token_f1")
    pairs: list[dict] = []
    errors_skipped = 0
    for key in common:
        cp, rp = cand_by_key[key], ref_by_key[key]
        if cp.error or rp.error:
            errors_skipped += 1
            continue
        pairs.append(_compare_one(key, cp, rp, f1_scorer))

    quality = _compute_quality(cfg, cand_name, ref_name)
    summary = _aggregate(cfg, cand_name, ref_name, cand_path, ref_path,
                         pairs, len(cand_preds), len(ref_preds),
                         only_cand, only_ref, errors_skipped, quality)

    report_dir = cfg.model_run_dir(cand_name, "mnn")
    store.write_json(report_dir / "precision.json", summary)
    store.write_text(report_dir / "precision.md", _render_md(summary, pairs, cfg))
    summary["report_json"] = str(report_dir / "precision.json")
    summary["report_md"] = str(report_dir / "precision.md")
    return summary


def _compute_quality(cfg: Config, cand_name: str, ref_name: str) -> dict:
    """净质量Δ:交叉候选(mnn)与参考(hf)各自 vs gold 的 scored.jsonl。

    需两端都跑过 score(scored.jsonl 存在)才计算;缺任一则优雅降级(不报错),
    返回 available=False + 原因,报告里注明「先跑 score」。
    """
    cand_scored = cfg.model_run_dir(cand_name, "mnn") / "scored.jsonl"
    ref_scored = cfg.model_run_dir(ref_name, "hf") / "scored.jsonl"
    missing = [str(p) for p in (cand_scored, ref_scored) if not p.exists()]
    if missing:
        return {"available": False,
                "reason": "未评分,跳过净质量Δ(先对两端各跑 score):缺 " + " / ".join(missing)}
    ct = compare.quality_crosstab(compare.load_scored(cand_scored),
                                  compare.load_scored(ref_scored))
    ct["available"] = True
    return ct


def _compare_one(key: tuple[str, int], cp: Prediction, rp: Prediction,
                 f1_scorer) -> dict:
    """对齐后的一对预测 -> 逐条指标(行为 + 输入对齐)。"""
    cand_text = cp.prediction or ""
    ref_text = rp.prediction or ""
    # token_f1:reference=HF(参考),prediction=MNN(候选)。
    f1 = f1_scorer.score_one(cand_text, ref_text, Sample(id=key[0])).detail.get("f1", 0.0)
    div = _first_divergence(cand_text, ref_text)

    cand_pt, ref_pt = _prompt_tokens(cp.raw), _prompt_tokens(rp.raw)
    cand_px, ref_px = _image_pixels(cp.raw), _image_pixels(rp.raw)
    return {
        "id": key[0],
        "turn": key[1],
        "exact": cand_text.strip() == ref_text.strip(),
        "token_f1": round(f1, 4),
        "edit_similarity": round(_edit_similarity(cand_text, ref_text), 4),
        "first_divergence": div,
        "len_candidate": len(cand_text),
        "len_reference": len(ref_text),
        "prompt_tokens_candidate": cand_pt,
        "prompt_tokens_reference": ref_pt,
        "prompt_tokens_delta": (cand_pt - ref_pt) if (cand_pt is not None and ref_pt is not None) else None,
        "image_pixels_candidate": cand_px,
        "image_pixels_reference": ref_px,
        "image_pixels_delta": (cand_px - ref_px) if (cand_px is not None and ref_px is not None) else None,
        "candidate": cand_text,
        "reference": ref_text,
    }


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _aggregate(cfg: Config, cand_name: str, ref_name: str,
               cand_path: Path, ref_path: Path, pairs: list[dict],
               n_cand: int, n_ref: int, only_cand: int, only_ref: int,
               errors_skipped: int, quality: dict) -> dict:
    """把逐条指标汇总成整体报告 + 生成人类可读的判定 flags。"""
    n = len(pairs)
    exacts = [p for p in pairs if p["exact"]]
    diverged = [p for p in pairs if p["first_divergence"] is not None]
    # 首发散归一化位置(相对参考长度):越小越靠前 = 越像 prefill/预处理/算子问题。
    norm_div = [p["first_divergence"] / max(p["len_reference"], 1)
                for p in diverged if p["len_reference"]]
    early = [p for p in diverged if (p["first_divergence"] or 0) < 10]

    pt_deltas = [p["prompt_tokens_delta"] for p in pairs if p["prompt_tokens_delta"] is not None]
    px_deltas = [p["image_pixels_delta"] for p in pairs if p["image_pixels_delta"] is not None]
    pt_mismatch = [d for d in pt_deltas if d != 0]
    px_mismatch = [d for d in px_deltas if d != 0]

    agreement = _mean([1.0 if p["exact"] else 0.0 for p in pairs])
    mean_f1 = _mean([p["token_f1"] for p in pairs])

    behavior = {
        "agreement_rate": round(agreement, 4),
        "mean_token_f1": round(mean_f1, 4),
        "mean_edit_similarity": round(_mean([p["edit_similarity"] for p in pairs]), 4),
        "num_identical": len(exacts),
        "num_diverged": len(diverged),
        "num_early_divergence": len(early),   # 首发散落在前 10 个字符
        "mean_normalized_first_divergence": round(_mean(norm_div), 4) if norm_div else None,
    }
    alignment = {
        "prompt_tokens": {
            "available": len(pt_deltas),
            "mismatches": len(pt_mismatch),
            "mean_abs_delta": round(_mean([abs(d) for d in pt_deltas]), 2) if pt_deltas else None,
            "max_abs_delta": max((abs(d) for d in pt_deltas), default=None),
        },
        "image_pixels": {
            "available": len(px_deltas),
            "mismatches": len(px_mismatch),
            "mean_abs_delta": round(_mean([abs(d) for d in px_deltas]), 2) if px_deltas else None,
            "max_abs_delta": max((abs(d) for d in px_deltas), default=None),
        },
    }

    flags = _build_flags(cfg, behavior, alignment, n, quality)

    return {
        "candidate": cand_name,
        "reference": ref_name,
        "candidate_predictions": str(cand_path),
        "reference_predictions": str(ref_path),
        "compared_at": datetime.now(timezone.utc).isoformat(),
        "num_candidate": n_cand,
        "num_reference": n_ref,
        "num_compared": n,
        "num_only_candidate": only_cand,
        "num_only_reference": only_ref,
        "num_errors_skipped": errors_skipped,
        "behavior": behavior,
        "alignment": alignment,
        "quality": quality,
        "flags": flags,
    }


def _build_flags(cfg: Config, behavior: dict, alignment: dict, n: int,
                 quality: dict) -> list[str]:
    """据阈值生成人类可读判定(报告顶部醒目提示 + 误差来源指向)。"""
    flags: list[str] = []
    pc = cfg.precision
    if n == 0:
        flags.append("⚠️ 无可对比样本(两份预测无共同 (id, turn),或均报错)。")
        return flags
    if behavior["agreement_rate"] < pc.agreement_min:
        flags.append(
            f"⚠️ 输出完全一致率 {behavior['agreement_rate']:.1%} < 阈值 {pc.agreement_min:.0%}:行为偏差偏大。"
        )
    if behavior["mean_token_f1"] < pc.token_f1_min:
        flags.append(
            f"⚠️ 平均 token-F1 {behavior['mean_token_f1']:.3f} < 阈值 {pc.token_f1_min}:文本相似度偏低。"
        )
    pt, px = alignment["prompt_tokens"], alignment["image_pixels"]
    if pc.alignment_strict and pt["available"] and pt["mismatches"]:
        flags.append(
            f"🔴 prompt token 数不对齐:{pt['mismatches']}/{pt['available']} 条 delta≠0"
            f"(最大 {pt['max_abs_delta']})——预处理/模板疑似不一致(VLM 最常见误差源)。"
        )
    if pc.alignment_strict and px["available"] and px["mismatches"]:
        flags.append(
            f"🔴 图片 resize 像素不一致:{px['mismatches']}/{px['available']} 条 delta≠0"
            f"(最大 {px['max_abs_delta']})——图片预处理未对齐。"
        )
    if behavior["num_diverged"] and behavior["num_early_divergence"] >= max(1, behavior["num_diverged"] // 2):
        flags.append(
            f"🟠 {behavior['num_early_divergence']}/{behavior['num_diverged']} 条发散发生在前 10 字符:"
            f"偏向 prefill/预处理/算子问题,而非解码噪声。"
        )
    # 净质量Δ(需两端都评过分):HF对了但候选错了的净回归率超阈值 -> 确有质量损失(非仅行为漂移)。
    if quality.get("available"):
        qb = quality.get("binary", {})
        if qb.get("num") and qb["net_regression_rate"] > pc.quality_regression_max:
            flags.append(
                f"🔴 净质量回归 {qb['net_regression_rate']:.1%}(HF对/候选错 "
                f"{qb['cand_wrong_ref_correct']}/{qb['num']} 条)> 阈值 {pc.quality_regression_max:.0%}:"
                f"转换/量化确实掉了质量,不只是行为漂移。"
            )
    if not flags:
        flags.append("✅ 未触发告警阈值:两端行为与输入对齐均在容差内。")
    return flags


# ---------------------------------------------------------------------------
# 人类可读报告
# ---------------------------------------------------------------------------
def _fence(text) -> str:
    s = "" if text is None else str(text)
    return "```\n" + s + "\n```"


def _render_quality_section(quality: dict) -> list[str]:
    """净质量Δ 小节:候选(MNN)相对参考(HF)各自 vs gold 的净回归/改进。"""
    out = ["## 净质量Δ(候选 vs 参考,各自 vs gold)", ""]
    if not quality.get("available"):
        out.append(f"> ⏭️ {quality.get('reason', '未评分,跳过(先对两端各跑 score)。')}")
        out.append("")
        return out
    b = quality.get("binary", {})
    c = quality.get("continuous", {})
    if b.get("num"):
        out += [
            f"二值轮(命中/未命中,{b['num']} 对):",
            "",
            "| 类别 | 条数 |",
            "| --- | --- |",
            f"| 双方都对 | {b['both_correct']} |",
            f"| 🔴 HF对 / 候选错(净回归) | {b['cand_wrong_ref_correct']}(净回归率 {b['net_regression_rate']:.1%}) |",
            f"| HF错 / 候选对(净改进) | {b['cand_correct_ref_wrong']}(净改进率 {b['net_improvement_rate']:.1%}) |",
            f"| 双方都错 | {b['both_wrong']} |",
            "",
        ]
    if c.get("num"):
        out += [
            f"连续轮(如 token_f1,{c['num']} 对):候选均分 {c['mean_score_candidate']} vs "
            f"参考 {c['mean_score_reference']}(Δ {c['mean_score_delta']:+g};负=候选更差)。",
            "",
        ]
    if not b.get("num") and not c.get("num"):
        out.append("> 两端无共同 (id, turn) 已评分记录。")
        out.append("")
    return out


def _render_md(summary: dict, pairs: list[dict], cfg: Config) -> str:
    b = summary["behavior"]
    a = summary["alignment"]
    lines = [
        f"# 精度对比报告(行为级) — 候选 `{summary['candidate']}` vs 参考 `{summary['reference']}`",
        "",
        f"- 候选(转换后·MNN)预测: `{summary['candidate_predictions']}`",
        f"- 参考(转换前·HF)预测: `{summary['reference_predictions']}`",
        f"- 对比时间: {summary['compared_at']}",
        f"- 可对比对: {summary['num_compared']}(候选 {summary['num_candidate']} / 参考 {summary['num_reference']};"
        f"仅候选有 {summary['num_only_candidate']} / 仅参考有 {summary['num_only_reference']};"
        f"跳过报错 {summary['num_errors_skipped']})",
        "",
        "## 判定",
        "",
    ]
    lines += [f"- {f}" for f in summary["flags"]]
    lines += [
        "",
        "## 行为级指标",
        "",
        "| 指标 | 值 | 定位 |",
        "| --- | --- | --- |",
        f"| 输出完全一致率 | {b['agreement_rate']:.1%} | 端到端行为(越高越好) |",
        f"| 平均 char token-F1 | {b['mean_token_f1']:.4f} | 文本相似度(→1.0 好) |",
        f"| 平均编辑相似度 | {b['mean_edit_similarity']:.4f} | 逐字漂移(→1.0 好) |",
        f"| 完全一致条数 | {b['num_identical']} / {summary['num_compared']} | —— |",
        f"| 发散条数 | {b['num_diverged']} | —— |",
        f"| 早发散条数(前10字符) | {b['num_early_divergence']} | 早=prefill/预处理/算子 |",
        f"| 平均首发散归一化位置 | {b['mean_normalized_first_divergence']} | 越小越靠前(越像结构性误差) |",
        "",
        "## 输入对齐审计(定位预处理)",
        "",
        "| 项 | 有数据条数 | 不一致条数 | 平均\\|Δ\\| | 最大\\|Δ\\| |",
        "| --- | --- | --- | --- | --- |",
        f"| prompt token 数 | {a['prompt_tokens']['available']} | {a['prompt_tokens']['mismatches']} "
        f"| {a['prompt_tokens']['mean_abs_delta']} | {a['prompt_tokens']['max_abs_delta']} |",
        f"| 图片 resize 像素 | {a['image_pixels']['available']} | {a['image_pixels']['mismatches']} "
        f"| {a['image_pixels']['mean_abs_delta']} | {a['image_pixels']['max_abs_delta']} |",
        "",
        "> prompt token 数已含视觉 token 展开,是预处理是否对齐的**主信号**:delta≠0 基本等价于"
        "「喂给模型的输入不一致」,此时应先修预处理再谈模型精度。",
        "",
    ]

    lines += _render_quality_section(summary.get("quality") or {})

    # 最差样本(按 token_f1 升序,排除完全一致)side-by-side。
    worst = sorted((p for p in pairs if not p["exact"]),
                   key=lambda p: p["token_f1"])[:cfg.precision.max_worst_samples]
    lines.append(f"## 最差样本(token-F1 最低 {len(worst)} 条)")
    lines.append("")
    if not worst:
        lines.append("✅ 所有可对比样本输出完全一致。")
        lines.append("")
    for p in worst:
        div = p["first_divergence"]
        pt_d, px_d = p["prompt_tokens_delta"], p["image_pixels_delta"]
        lines.append(f"### `{p['id']}` · turn {p['turn']} · token-F1 {p['token_f1']} · "
                     f"首发散@{div if div is not None else '—'}")
        meta = []
        if pt_d is not None:
            meta.append(f"prompt token Δ={pt_d:+g}")
        if px_d is not None:
            meta.append(f"图片像素 Δ={px_d:+g}")
        if meta:
            lines.append("- 对齐: " + ", ".join(meta))
        lines.append("- 候选(MNN)输出:")
        lines.append(_fence(p["candidate"]))
        lines.append("- 参考(HF)输出:")
        lines.append(_fence(p["reference"]))
        lines.append("")
    return "\n".join(lines)
