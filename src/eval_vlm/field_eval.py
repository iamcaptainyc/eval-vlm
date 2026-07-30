"""第一轮描述的「字段抽取 -> 逐字段准确率」评测(field-eval 子命令)。

第一轮是自由文本描述,表面字符串 scorer(exact_match/token_f1…)难以衡量其质量。本模块把
ref(标准答案)与 pred(模型输出)两段描述**分别**发给 value-extract 服务(见服务端
`/vlm/value-extract`),各自解析成一组**固定枚举字段**(主辅路/道路结构/车道位置/警示标志),
再**逐字段严格相等**比对,得出每字段准确率——既可聚合成数字,又能定位模型在哪个维度弱。

与 label_extract 的关系:
  - 复用其 HTTP 调用/重试(`_post_data`/`_retry_extract`/`LabelExtractError`),但路由指向
    `label_extract.value_path`(通过 dataclasses.replace 换 path,不影响旧 label-extract)。
  - **不复用** `parse_cn_labels`:它把字段打平并丢「无」;逐字段准确率必须**保留字段结构、
    保留「无」**(ref=无 且 pred=无 判对),故用本模块的 `parse_cn_fields`。

三阶段(见 `run_field_eval`):
  A. 抽 ref 字段 -> **数据集级**缓存(ref 跨模型运行不变,各模型/后端复用);
  B. 抽 pred 字段 -> **运行级**;
  C. 逐字段比对 + 聚合 -> field_metrics.json / field_summary.md / field_mismatches.md。
抽取阶段复刻 label_extract 的并发/断点续跑/flush+fsync/失败落盘。
"""
from __future__ import annotations

import dataclasses
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from .config import Config, LabelExtractConfig
from .data.loader import load_samples
from .data.schema import Sample
from .label_extract import LabelExtractError, _endpoint, _retry_extract
from .results import store


# ---------------------------------------------------------------------------
# 解析:保留字段结构与「无」(与 label_extract.parse_cn_labels 的打平语义相反)
# ---------------------------------------------------------------------------
def parse_cn_fields(data: dict, none_label: str = "无") -> dict[str, list[str]]:
    """从服务响应的 data.labels.cn 解析出 {字段: [值,...]},**保留全部字段键**。

    每个字段的值列表:strip、去重保序过滤掉空串与 none_label(「无」)。因此空字段 -> []
    (即「无」),非空字段 -> 该字段实际取值。注意 value-extract 的「无警示牌」是合法枚举值
    (不等于 none_label「无」),会被保留 —— ref/pred 两侧一致即判对。
    """
    cn = ((data or {}).get("labels") or {}).get("cn") or {}
    out: dict[str, list[str]] = {}
    for field, values in cn.items():
        vals: list[str] = []
        seen: set[str] = set()
        if isinstance(values, list):
            for v in values:
                s = str(v).strip()
                if not s or s == none_label or s in seen:
                    continue
                seen.add(s)
                vals.append(s)
        out[str(field)] = sorted(vals)   # 排序:比较用集合,存盘顺序无关,排序便于人读/稳定
    return out


def extract_fields_one(text: str, le: LabelExtractConfig) -> dict[str, list[str]]:
    """对单条描述抽取字段字典(带重试);le 的 path 应已指向 value-extract 路由。"""
    return _retry_extract(text, le, lambda data: parse_cn_fields(data, le.none_label))


def _value_config(le: LabelExtractConfig) -> LabelExtractConfig:
    """复制一份把 path 换成 value_path 的配置(其余不变),使 _post_data 打到 value-extract。"""
    return dataclasses.replace(le, path=le.value_path)


# ---------------------------------------------------------------------------
# 通用抽取批(复刻 label_extract.run_label_extract 的并发/续跑/落盘)
# ---------------------------------------------------------------------------
def _append(fh, obj: dict) -> None:
    """写一条 -> flush -> fsync:中途崩溃/被杀,已写结果也已落盘,可断点续跑。"""
    fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
    fh.flush()
    try:
        os.fsync(fh.fileno())
    except OSError:
        pass


def _done_ids(path: Path) -> set[str]:
    """读 out 文件中已成功抽取的 id(断点续跑跳过;有 error 的不算)。"""
    done: set[str] = set()
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("error") is None and "id" in obj:
                done.add(str(obj["id"]))
    return done


def load_fields(path: Path) -> dict[str, dict[str, list[str]]]:
    """读 fields jsonl -> {id: {字段: [值,...]}}(只取成功行,后写覆盖先写)。"""
    out: dict[str, dict[str, list[str]]] = {}
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("error") is None and "id" in obj and isinstance(obj.get("fields"), dict):
                out[str(obj["id"])] = {str(k): list(v) for k, v in obj["fields"].items()}
    return out


def _extract_batch(items: list[tuple[str, str]], le: LabelExtractConfig,
                   out_path: Path, fail_path: Path, *, label: str,
                   overwrite: bool) -> dict:
    """把 [(id, text)] 逐条发给服务抽取字段,成功写 out_path、失败写 fail_path。

    并发(le.max_concurrency)、断点续跑(out_path 已成功的 id 跳过)、每条 flush+fsync、
    失败不中断整批。overwrite=True 忽略已有 out_path 整份重跑。返回统计。
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set() if overwrite else _done_ids(out_path)
    todo = [(sid, text) for sid, text in items if sid not in done]

    n_ok = 0
    n_err = 0
    interrupted = False
    if not todo:
        print(f"[field-eval] {label}: 全部 {len(items)} 条已抽取,跳过(断点续跑)。", flush=True)
        return {"total": len(items), "newly_completed": 0, "errors": 0,
                "skipped_already_done": len(items), "interrupted": False}

    print(f"[field-eval] {label}: 待抽取 {len(todo)} 条(已完成跳过 {len(items) - len(todo)} 条)"
          f" -> {_endpoint(le)}", flush=True)
    max_workers = max(1, le.max_concurrency)
    with out_path.open("w" if overwrite else "a", encoding="utf-8") as ok_fh, \
            fail_path.open("w", encoding="utf-8") as fail_fh:

        def _record(sid: str, result) -> None:
            nonlocal n_ok, n_err
            if isinstance(result, Exception):
                n_err += 1
                _append(fail_fh, {"id": sid, "error": f"{type(result).__name__}: {result}"})
            else:
                n_ok += 1
                _append(ok_fh, {"id": sid, "fields": result})

        try:
            if max_workers == 1:
                for sid, text in tqdm(todo, desc=f"field-extract:{label}", unit="item"):
                    try:
                        _record(sid, extract_fields_one(text, le))
                    except Exception as e:  # noqa: BLE001 - 记录而非中断整批
                        _record(sid, e)
            else:
                with ThreadPoolExecutor(max_workers=max_workers) as pool:
                    futures = {pool.submit(extract_fields_one, text, le): sid
                               for sid, text in todo}
                    try:
                        for fut in tqdm(as_completed(futures), total=len(futures),
                                        desc=f"field-extract:{label}", unit="item"):
                            sid = futures[fut]
                            try:
                                _record(sid, fut.result())
                            except Exception as e:  # noqa: BLE001
                                _record(sid, e)
                    except KeyboardInterrupt:
                        for fut in futures:
                            fut.cancel()
                        raise
        except KeyboardInterrupt:
            interrupted = True
            print(f"\n[field-eval] {label} 已中断:本轮成功 {n_ok} 条已落盘 -> {out_path};"
                  f"重跑同一命令即可续跑补齐。", flush=True)

    return {"total": len(items), "newly_completed": n_ok, "errors": n_err,
            "skipped_already_done": len(items) - len(todo), "interrupted": interrupted}


# ---------------------------------------------------------------------------
# 取描述文本:ref 从 test.json、pred 从 predictions.jsonl
# ---------------------------------------------------------------------------
def _first_assistant(sample: Sample) -> tuple[int, str]:
    """取样本第一个 assistant 轮的 (turn_index, content);无则 (-1, "")。

    第一轮描述 = 首个 assistant 轮,不依赖 cfg.eval.targets 模式。
    """
    for i, t in enumerate(sample.turns):
        if getattr(t, "role", "") == "assistant":
            return i, str(getattr(t, "content", "") or "")
    return -1, ""


def _ref_items(samples: list[Sample]) -> tuple[list[tuple[str, str]], dict[str, int]]:
    """从样本取 [(id, 描述文本)] 与 {id: 描述轮 index};跳过无描述/空描述的样本。"""
    items: list[tuple[str, str]] = []
    desc_turn: dict[str, int] = {}
    for s in samples:
        idx, text = _first_assistant(s)
        if idx < 0 or not text.strip():
            continue
        items.append((s.id, text))
        desc_turn[s.id] = idx
    return items, desc_turn


def _detect_datadir_format(path: Path) -> bool:
    """预测文件是 pred --datadir 的对话格式(有 messages、无顶层 prediction)则 True。"""
    if not path.exists():
        return False
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            return "messages" in obj and "prediction" not in obj
    return False


def _pred_items(pred_path: Path, desc_turn: dict[str, int]) -> list[tuple[str, str]]:
    """从 predictions.jsonl 取每 id 描述轮的 [(id, 描述文本)];缺/空/报错的 id 不产出。

    兼容 run/pred --dataset 的 Prediction 格式与 pred --datadir 的对话格式。按描述轮
    turn_index 对齐;取不到该轮时回落到该 id 唯一/首个 assistant 预测。
    """
    preds = store.load_predictions(pred_path, datadir_format=_detect_datadir_format(pred_path))
    by_key: dict[tuple[str, int], object] = {}
    by_id: dict[str, list] = {}
    for p in preds:
        by_key[(p.id, p.turn)] = p
        by_id.setdefault(p.id, []).append(p)

    items: list[tuple[str, str]] = []
    for sid, turn in desc_turn.items():
        p = by_key.get((sid, turn))
        if p is None:
            cands = by_id.get(sid) or []
            # 回落:优先无 error 的、turn 最小(最靠前的 assistant)的预测
            cands = sorted((c for c in cands if not c.error), key=lambda c: c.turn)
            p = cands[0] if cands else None
        if p is None or p.error:
            continue
        text = str(p.prediction or "")
        if not text.strip():
            continue
        items.append((sid, text))
    return items


# ---------------------------------------------------------------------------
# 比对 + 聚合
# ---------------------------------------------------------------------------
def _canonical_fields(ref_fields: dict[str, dict[str, list[str]]]) -> list[str]:
    """规范字段集 = 全部 ref 记录里出现过的字段键并集(排序)。空则回落到已知 4 字段。"""
    keys: set[str] = set()
    for fields in ref_fields.values():
        keys.update(fields.keys())
    if not keys:
        return ["主辅路", "道路结构", "车道位置", "警示标志"]
    return sorted(keys)


def _aggregate(samples: list[Sample], desc_turn: dict[str, int],
               ref_fields: dict[str, dict[str, list[str]]],
               pred_fields: dict[str, dict[str, list[str]]],
               pred_desc_ids: set[str]) -> tuple[dict, list[dict]]:
    """逐字段严格相等比对并聚合。返回 (metrics, rows) —— rows 供渲染失配清单。

    判定规则:
      - ref 抽取失败/无描述(id 不在 ref_fields)-> 跳过该 id(skipped_ref)。
      - pred **无描述文本**(模型没产出;id 不在 pred_desc_ids)-> 该 id 全字段判错(pred_missing)。
      - pred 有描述但**抽取失败**(id 不在 pred_fields 却在 pred_desc_ids)-> 跳过(skipped_pred_error)。
      - 两侧都有 -> 逐字段 set(ref)==set(pred) 即对(两侧皆空=对,某字段缺=空集参与)。
    """
    fields = _canonical_fields(ref_fields)
    per_field = {f: {"correct": 0, "total": 0} for f in fields}
    n_scored = 0
    n_exact = 0
    n_pred_missing = 0
    skipped_ref = 0
    skipped_pred_error = 0
    rows: list[dict] = []

    for s in samples:
        sid = s.id
        if sid not in desc_turn:            # 无 ref 描述文本
            skipped_ref += 1
            continue
        ref_f = ref_fields.get(sid)
        if ref_f is None:                   # ref 抽取失败
            skipped_ref += 1
            continue

        has_pred_text = sid in pred_desc_ids
        pred_f = pred_fields.get(sid)
        if not has_pred_text:
            state = "pred_missing"
            n_pred_missing += 1
        elif pred_f is None:                # 有描述但抽取失败 -> 基建问题,跳过
            skipped_pred_error += 1
            continue
        else:
            state = "compared"

        n_scored += 1
        all_correct = True
        field_rows = []
        for f in fields:
            r = sorted(set(ref_f.get(f, [])))
            if state == "pred_missing":
                p = []
                correct = False                       # 模型没产出 -> 一律判错
            else:
                p = sorted(set(pred_f.get(f, [])))
                correct = set(r) == set(p)
            per_field[f]["total"] += 1
            if correct:
                per_field[f]["correct"] += 1
            else:
                all_correct = False
            field_rows.append({"field": f, "ref": r, "pred": p, "correct": correct})
        if all_correct:
            n_exact += 1
        if not all_correct:                 # 只把有失配(含 pred_missing)的 id 列入清单
            rows.append({"id": sid, "state": state, "fields": field_rows})

    # 聚合指标
    for f in fields:
        c, t = per_field[f]["correct"], per_field[f]["total"]
        per_field[f]["accuracy"] = round(c / t, 4) if t else 0.0
    tot_correct = sum(per_field[f]["correct"] for f in fields)
    tot_total = sum(per_field[f]["total"] for f in fields)
    macro = (sum(per_field[f]["accuracy"] for f in fields) / len(fields)) if fields else 0.0

    metrics = {
        "fields": fields,
        "num_samples": len(samples),
        "num_scored": n_scored,
        "num_pred_missing": n_pred_missing,
        "skipped_ref": skipped_ref,
        "skipped_pred_error": skipped_pred_error,
        "per_field": per_field,
        "overall": {
            "micro_accuracy": round(tot_correct / tot_total, 4) if tot_total else 0.0,
            "macro_accuracy": round(macro, 4),
            "exact_match_samples": n_exact,
            "exact_match_rate": round(n_exact / n_scored, 4) if n_scored else 0.0,
        },
    }
    return metrics, rows


# ---------------------------------------------------------------------------
# 渲染
# ---------------------------------------------------------------------------
def _render_summary(metrics: dict, cfg: Config) -> str:
    ov = metrics["overall"]
    lines = [
        f"# 逐字段准确率 — {cfg.run_name}",
        "",
        f"- 模型: `{cfg.inference.result_name}`  后端: `{cfg.inference.backend}`",
        f"- 样本数: {metrics['num_samples']}  已评: {metrics['num_scored']}  "
        f"模型无输出(判错): {metrics['num_pred_missing']}",
        f"- 跳过(ref 抽取失败/无描述): {metrics['skipped_ref']}  "
        f"跳过(pred 抽取失败): {metrics['skipped_pred_error']}",
        f"- **micro 准确率**: {ov['micro_accuracy']}  **macro 准确率**: {ov['macro_accuracy']}",
        f"- 整样本全对: {ov['exact_match_samples']} / {metrics['num_scored']}"
        f"(全对率 {ov['exact_match_rate']})",
        "",
        "## 逐字段准确率",
        "",
        "| 字段 | 准确率 | 命中/总数 |",
        "| --- | --- | --- |",
    ]
    for f in metrics["fields"]:
        pf = metrics["per_field"][f]
        lines.append(f"| {f} | {pf['accuracy']} | {pf['correct']}/{pf['total']} |")
    lines.append("")
    return "\n".join(lines)


def _fmt(vals: list[str]) -> str:
    return "、".join(vals) if vals else "无"


def _render_mismatches(rows: list[dict], metrics: dict, cfg: Config) -> str:
    head = [
        f"# 字段失配清单 — {cfg.run_name}",
        "",
        f"- 模型: `{cfg.inference.result_name}`  后端: `{cfg.inference.backend}`",
        f"- 有失配的样本: {len(rows)} / 已评 {metrics['num_scored']}",
        "",
        "> 每个样本列出各字段的 ref vs pred 值与 ✓/✗;`state=pred_missing` 表示模型未产出描述(全字段判错)。",
        "",
        "---",
        "",
    ]
    if not rows:
        return "\n".join(head[:-3] + ["✅ 无字段失配。", ""])
    blocks: list[str] = []
    for row in rows:
        tag = "  ⚠️ 模型未产出描述" if row["state"] == "pred_missing" else ""
        blocks.append(f"## 样本 `{row['id']}`{tag}")
        blocks.append("")
        blocks.append("| 字段 | ref | pred | |")
        blocks.append("| --- | --- | --- | --- |")
        for fr in row["fields"]:
            mark = "✓" if fr["correct"] else "✗"
            blocks.append(f"| {fr['field']} | {_fmt(fr['ref'])} | {_fmt(fr['pred'])} | {mark} |")
        blocks.append("")
    return "\n".join(head + blocks)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def run_field_eval(cfg: Config, *, overwrite: bool = False) -> dict:
    """三阶段跑通字段抽取评测,落盘并返回聚合指标。"""
    if not cfg.test_path.exists():
        raise FileNotFoundError(
            f"未找到测试集 {cfg.test_path},请先运行: eval-vlm split --dataset <源json>"
        )
    if not cfg.predictions_path.exists():
        raise FileNotFoundError(
            f"未找到预测文件 {cfg.predictions_path},请先运行 pred/eval 生成描述。"
        )

    le = _value_config(cfg.label_extract)
    samples = load_samples(cfg, source=cfg.test_path)

    # ---- A. 抽 ref 字段(数据集级缓存,跨模型复用)----
    ref_items, desc_turn = _ref_items(samples)
    ref_stats = _extract_batch(ref_items, le, cfg.field_ref_path, cfg.field_ref_failures_path,
                               label="ref", overwrite=overwrite)

    # ---- B. 抽 pred 字段(运行级)----
    pred_items = _pred_items(cfg.predictions_path, desc_turn)
    pred_desc_ids = {sid for sid, _ in pred_items}
    pred_stats = _extract_batch(pred_items, le, cfg.field_pred_path, cfg.field_pred_failures_path,
                                label="pred", overwrite=overwrite)

    # ---- C. 比对 + 聚合 ----
    ref_fields = load_fields(cfg.field_ref_path)
    pred_fields = load_fields(cfg.field_pred_path)
    metrics, rows = _aggregate(samples, desc_turn, ref_fields, pred_fields, pred_desc_ids)

    metrics = {
        "run_name": cfg.run_name,
        "model": cfg.inference.result_name,
        "backend": cfg.inference.backend,
        "scored_at": datetime.now(timezone.utc).isoformat(),
        "endpoint": _endpoint(le),
        "ref_extract": ref_stats,
        "pred_extract": pred_stats,
        **metrics,
    }
    store.write_json(cfg.field_metrics_path, metrics)
    store.write_text(cfg.field_summary_path, _render_summary(metrics, cfg))
    store.write_text(cfg.field_mismatches_path, _render_mismatches(rows, metrics, cfg))
    return metrics
