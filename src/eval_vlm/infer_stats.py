"""通用推理性能指标统计与报告生成模块。

统一支持 MNN 与 llama.cpp 等后端：
解析 predictions.jsonl 中的耗时与 token 数，生成统计摘要并落盘 infer_stats.txt 与 infer_stats.json。
"""
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, median
from typing import Any, Optional, Union


def percentile(sorted_vals: list[float], p: float) -> float:
    """线性插值分位数（numpy 风格）。"""
    n = len(sorted_vals)
    if n == 0:
        return float("nan")
    if n == 1:
        return sorted_vals[0]
    k = (n - 1) * p
    f = int(k)
    c = f + 1
    if c >= n:
        return sorted_vals[-1]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def safe_div(a: Optional[float], b: Optional[float]) -> Optional[float]:
    return a / b if b else None


def parse_metrics_line(line: str, lineno: int = 1) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    """解析单行 predictions.jsonl，返回 (metrics_dict, skip_reason)。
    metrics_dict 里以 '_' 开头的键只用于聚合统计，不参与逐样本指标求均值。
    """
    line = line.strip()
    if not line:
        return None, "empty line"

    try:
        obj = json.loads(line)
    except json.JSONDecodeError as e:
        return None, f"invalid json: {e}"

    if obj.get("error"):
        return None, f"error field: {obj['error']}"

    raw = obj.get("raw")
    if not isinstance(raw, dict):
        return None, "missing 'raw' dict"

    prompt_len = raw.get("prompt_len", raw.get("prompt_token_count"))
    gen_seq_len = raw.get("gen_seq_len")
    vision_us = raw.get("vision_us")
    prefill_us = raw.get("prefill_us")
    decode_us = raw.get("decode_us")
    latency = obj.get("latency")

    m: dict[str, Any] = {}

    if vision_us is not None:
        m["vision_s"] = vision_us / 1e6
    if prefill_us is not None:
        m["prefill_s"] = prefill_us / 1e6

    # TTFT = 视觉编码 + prefill（首个 token 在 prefill 结束时产出）
    if vision_us is not None and prefill_us is not None:
        m["ttft_s"] = (vision_us + prefill_us) / 1e6
    elif prefill_us is not None:
        # 兼容无显式 vision 计时（例如 llama.cpp 中 prefill 耗时已涵盖视觉多模态 tokens）
        m["ttft_s"] = prefill_us / 1e6

    # TPOT：prefill 已产出第 1 个 token，decode 产出剩余 (gen_seq_len - 1) 个
    if decode_us is not None and gen_seq_len is not None:
        if gen_seq_len > 1:
            m["tpot_ms"] = decode_us / (gen_seq_len - 1) / 1e3
        elif gen_seq_len == 1:
            m["tpot_ms"] = 0.0
        # gen_seq_len == 0 时跳过

    if latency is not None:
        m["e2e_s"] = float(latency)

    if prefill_us is not None and decode_us is not None:
        v_us = vision_us if vision_us is not None else 0
        m["e2e_calc_s"] = (v_us + prefill_us + decode_us) / 1e6

    if decode_us and gen_seq_len:
        m["decode_tps"] = gen_seq_len / (decode_us / 1e6)

    if prefill_us and prompt_len:
        m["prefill_tps"] = prompt_len / (prefill_us / 1e6)

    # 用于全样本聚合的原始计数
    if gen_seq_len is not None:
        m["_gen_seq_len"] = gen_seq_len
    if prompt_len is not None:
        m["_prompt_len"] = prompt_len

    return m, None


METRIC_META = [
    ("vision_s",     "Vision (s)",        "image / vision-tower encoding time"),
    ("prefill_s",    "Prefill (s)",       "prompt prefill time"),
    ("ttft_s",       "TTFT (s)",          "time to first token (vision + prefill)"),
    ("tpot_ms",      "TPOT (ms/token)",   "time per output token, excl. first"),
    ("e2e_s",        "E2E (s)",           "wall-clock latency as reported"),
    ("e2e_calc_s",   "E2E calc (s)",      "vision + prefill + decode summed"),
    ("decode_tps",   "Decode (tok/s)",    "decode throughput"),
    ("prefill_tps",  "Prefill (tok/s)",   "prefill throughput"),
]


def aggregate_metrics(metric_lists: dict[str, list[float]]) -> dict[str, dict[str, Any]]:
    """metric_lists: {name: [values]} -> {name: stats_dict}"""
    out: dict[str, dict[str, Any]] = {}
    for name, vals in metric_lists.items():
        if not vals:
            continue
        vals = sorted(vals)
        out[name] = {
            "count": len(vals),
            "mean": mean(vals),
            "median": median(vals),
            "min": vals[0],
            "max": vals[-1],
            "p90": percentile(vals, 0.90),
            "p99": percentile(vals, 0.99),
        }
    return out


def format_inference_report(
    stats: dict[str, dict[str, Any]],
    n_ok: int,
    n_skipped: int,
    skip_reasons: dict[str, int],
    total_prompt_tokens: int,
    total_gen_tokens: int,
    total_time_s: float,
) -> str:
    """构建人类可读的文本报告字符串。"""
    L = []
    L.append("=" * 78)
    L.append("VLM inference metrics summary")
    L.append("=" * 78)
    L.append(f"Total samples : {n_ok + n_skipped}")
    L.append(f"  analyzed    : {n_ok}")
    L.append(f"  skipped     : {n_skipped}")
    for reason, cnt in skip_reasons.items():
        L.append(f"    - {reason}: {cnt}")
    L.append("-" * 78)

    header = (f"{'metric':<18}{'count':>7}{'mean':>12}{'median':>12}"
              f"{'min':>12}{'max':>12}{'p90':>12}{'p99':>12}")
    L.append(header)
    L.append("-" * len(header))
    for name, label, _desc in METRIC_META:
        if name not in stats:
            continue
        s = stats[name]
        L.append(f"{label:<18}{s['count']:>7}{s['mean']:>12.3f}{s['median']:>12.3f}"
                 f"{s['min']:>12.3f}{s['max']:>12.3f}{s['p90']:>12.3f}{s['p99']:>12.3f}")

    L.append("-" * 78)
    L.append("Aggregate throughput (across all analyzed samples):")
    L.append(f"  total prompt tokens : {total_prompt_tokens}")
    L.append(f"  total output tokens : {total_gen_tokens}")
    L.append(f"  total wall time     : {total_time_s:.3f} s")
    if total_time_s > 0:
        L.append(f"  output tok/s        : {total_gen_tokens / total_time_s:.2f}")
        L.append(f"  total tok/s         : {(total_prompt_tokens + total_gen_tokens) / total_time_s:.2f}")
    L.append("=" * 78)
    return "\n".join(L) + "\n"


def generate_inference_report(
    predictions_path: Union[Path, str],
    out_dir: Optional[Union[Path, str]] = None,
    print_report: bool = True,
) -> tuple[dict[str, Any], str]:
    """解析 predictions.jsonl，生成推理性能报告并落盘 infer_stats.txt 与 infer_stats.json。

    Returns:
        (summary_dict, report_text)
    """
    pred_path = Path(predictions_path)
    if not pred_path.exists():
        raise FileNotFoundError(f"predictions file not found: {pred_path}")

    metric_lists: dict[str, list[float]] = {}
    total_gen_tokens = 0
    total_prompt_tokens = 0
    total_time_s = 0.0
    n_ok = 0
    n_skipped = 0
    skip_reasons: dict[str, int] = {}

    with open(pred_path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            m, reason = parse_metrics_line(line, lineno)
            if m is None:
                n_skipped += 1
                if reason:
                    skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
                continue
            n_ok += 1
            for k, v in m.items():
                if k.startswith("_"):
                    continue
                metric_lists.setdefault(k, []).append(v)
            total_gen_tokens += m.get("_gen_seq_len") or 0
            total_prompt_tokens += m.get("_prompt_len") or 0
            total_time_s += m.get("e2e_s") or 0.0

    stats = aggregate_metrics(metric_lists)
    report = format_inference_report(
        stats, n_ok, n_skipped, skip_reasons,
        total_prompt_tokens, total_gen_tokens, total_time_s,
    )

    summary = {
        "samples": {
            "total": n_ok + n_skipped,
            "analyzed": n_ok,
            "skipped": n_skipped,
            "skip_reasons": skip_reasons,
        },
        "metrics": stats,
        "aggregate": {
            "total_prompt_tokens": total_prompt_tokens,
            "total_output_tokens": total_gen_tokens,
            "total_wall_time_s": total_time_s,
            "output_tokens_per_s": safe_div(total_gen_tokens, total_time_s),
            "total_tokens_per_s": safe_div(total_prompt_tokens + total_gen_tokens, total_time_s),
        },
    }

    target_dir = Path(out_dir) if out_dir else pred_path.parent
    target_dir.mkdir(parents=True, exist_ok=True)
    txt_path = target_dir / "infer_stats.txt"
    json_path = target_dir / "infer_stats.json"

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(report)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if print_report:
        print(report)
        print(f"[infer_stats] 性能报告已写入:\n  {txt_path}\n  {json_path}")

    return summary, report
