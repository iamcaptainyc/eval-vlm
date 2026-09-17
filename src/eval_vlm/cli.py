"""命令行入口(工作目录模型)。

机器级设置放全局配置(~/.eval_vlm/config.yaml);每个数据集是工作目录下一个
自包含文件夹(含 config.yaml + 全部产物)。各命令对 --dataset / --datadir 的含义:

    config  : 管理全局配置(init / show / set)
    split   : 初始化  —— --dataset = 源数据集 JSON 路径
              在 workspace 下建 <数据集名>/,从内置模板生成 config.yaml,再分割
    pred    : 预测(不评分),二选一:
              --dataset = 数据集名(或文件夹路径);对其 test.json 推理 -> predictions.jsonl
              --datadir = 无标注图片文件夹;逐张单轮描述,产物落 workspace/<同名>/
    score   : 读取已有 —— predictions.jsonl -> metrics.json / scored.jsonl / failures.md / summary.md
    eval    : 读取已有 —— 一键连续执行 pred(--dataset)+ score(不含 split:split 后需先部署模型)
    sweep   : 批量评测 —— 同一模型对一批数据集依次跑各自的 eval/field-eval(方法由各数据集 eval.method 决定)

产物按 模型/后端 分目录:pred/score/eval 的结果落在 工作目录/<数据集>/<模型名>/<后端类型>/,
同一模型的不同后端互不覆盖(openai/vllm/fake 模型名取 inference.openai.model;
mnn 取 inference.mnn.config_path 所在目录名;hf 取 model_path 目录名;后端类型取
inference.backend);split 产物(train/test/val)是各模型共享的,落在数据集文件夹本身。

CLI 覆盖会「永久写回」该数据集 config.yaml(用户参数优先且持久化,不再是临时):
    --base-url / --model     pred/eval 写回 inference.openai.base_url / inference.openai.model
    --scorer                 score/eval 写回 scoring.scorer
    --backend                pred/eval 写回 inference.backend
    --mnn-config / --mnn-image-max-side / --mnn-quant   pred/eval 写回 inference.mnn.*
    --hf-model               pred/eval 写回 inference.hf.model_path
    --prompt / --system-prompt            pred --datadir 写回 pred.prompt / pred.system_prompt
    --label-extract-url / --label-extract-token  pred 写回 label_extract.base_url / auth_token
    --value-path              field-eval 写回 label_extract.value_path(不同数据集可用不同抽取路由)
    --targets                 field-eval 写回 eval.targets(评哪个 assistant 轮:all/last/first/数字,如 2=第2轮)
    --method                  sweep 写回各数据集的 eval.method(批量覆盖评测方法)
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Optional

from pathlib import Path

from .config import Config, DEFAULT_PROMPT, load_dataset_config
from .data.splitter import split_dataset
from .runner import run_inference
from .predict import predict_folder
from .label_extract import run_label_extract
from .field_eval import run_field_eval
from .evaluate import score_predictions
from .data.loader import load_samples, resolve_image_path
from .precision import compare_precision
from .report import build_report, render_report_md
from .results import store
from .scoring import available_scorers
from . import workspace
from .sweep import run_sweep
from .comparison import compare_dataset, filter_records, render_html, sort_records


# ---------------------------------------------------------------------------
# config:管理全局配置
# ---------------------------------------------------------------------------
def _cmd_config(args: argparse.Namespace) -> int:
    action = args.action
    if action == "init":
        path = workspace.init_global_config(force=args.force)
        print(f"[config] 全局配置 -> {path}")
    elif action == "show":
        path = workspace.global_config_path()
        cfg = workspace.load_global_config()
        print(f"[config] {path}")
        print(json.dumps(cfg, ensure_ascii=False, indent=2))
    elif action == "keys":
        print(workspace.describe_settable_keys())
    elif action == "set":
        if not args.key:
            print("用法: eval-vlm config set <key> <value>", file=sys.stderr)
            return 2
        value = args.value
        if value is not None and value.lower() in ("null", "none", ""):
            value = None
        path = workspace.set_global_value(args.key, value)
        print(f"[config] {args.key} = {value!r} -> {path}")
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    """Generate a review-first, offline comparison report without rerunning models."""
    folder = _resolve_folder(args)
    cfg = load_dataset_config(folder)
    run_specs = [item for group in args.runs for item in group]
    comparison = compare_dataset(
        cfg, run_specs, args.baseline,
        allow_mixed_dataset=args.allow_mixed_dataset,
    )
    records = filter_records(comparison["records"], category=args.filter,
                             query=args.query, include_agreements=args.include_agreements)
    records = sort_records(records, args.sort, args.descending)
    if args.offset:
        records = records[args.offset:]
    if args.limit is not None and args.limit > 0:
        records = records[:args.limit]
    summary = comparison["summary"]
    print(f"[compare] 对齐 {summary['aligned_total']} 条；当前筛选 {len(records)} 条；baseline={comparison['baseline']}")
    print("[compare] 差异分类: " + ", ".join(f"{k}={v}" for k, v in sorted(summary["categories"].items())))
    for run, value in summary["runs"].items():
        print(f"  {run}: score={value['mean_score']} correct/wrong={value['correct']}/{value['wrong']} "
              f"missing/error={value['missing_or_error']}")
    for warning in comparison["warnings"]:
        print(f"[compare] 警告: {warning}", file=sys.stderr)
    if args.output:
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        public = {key: value for key, value in comparison.items() if key != "records"}
        public["filtered_total"] = len(records)
        (output / "comparison.json").write_text(json.dumps(public, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        with (output / "comparison_samples.jsonl").open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        def report_image_url(ref: str) -> str:
            if str(ref).startswith(("http://", "https://", "data:")):
                return str(ref)
            try:
                return resolve_image_path(str(ref), cfg).resolve().as_uri()
            except Exception:
                return str(ref)
        (output / "comparison.html").write_text(render_html(comparison, records, report_image_url), encoding="utf-8")
        print(f"[compare] 报告 -> {output.resolve()}")
    return 0


# ---------------------------------------------------------------------------
# split:初始化数据集文件夹 + 分割
# ---------------------------------------------------------------------------
def _cmd_split(args: argparse.Namespace) -> int:
    global_cfg = workspace.load_global_config()
    ws = workspace.resolve_workspace(args.workspace, global_cfg)

    # 仅收集显式提供的 split 参数,写进生成的 config.yaml。
    so: dict = {}
    for key in ("train", "test", "val", "seed", "stratify_by"):
        val = getattr(args, key, None)
        if val is not None:
            so[key] = val

    folder = workspace.init_dataset(
        args.dataset, ws,
        name=args.name,
        split_overrides=so,
        split_defaults=global_cfg.get("split"),   # 不传 --train 等时用全局默认
        media_root=global_cfg.get("media_root", "."),
        image_strip_prefix=global_cfg.get("image_strip_prefix"),
        force=args.force,
    )
    cfg = load_dataset_config(folder)
    # 自定义产物位置(临时覆盖,可直接写到 LlamaFactory data/):
    # 带路径=写该路径;光杆旗标=落到全局 <份>_out_dir/<数据集名>_<份>.json。
    for kind in ("train", "val", "test"):
        out = workspace.resolve_split_out(
            getattr(args, f"{kind}_out"), kind, folder.name, global_cfg)
        if out:
            setattr(cfg.split, f"{kind}_out", out)

    meta = split_dataset(cfg)
    counts = meta["counts"]
    files = meta["files"]
    print(f"[split] 数据集 '{folder.name}' -> {folder}")
    print(f"[split] config.yaml 已生成;共 {meta['total_samples']} 条 -> "
          f"train {counts['train']} / val {counts['val']} / test {counts['test']} "
          f"(seed={meta['seed']})")
    for name in ("train", "val", "test"):
        if name in files:
            print(f"        {name}.json (LlamaFactory 格式) -> {files[name]}")
    print(f"[split] 后续: 部署模型 -> eval-vlm eval --dataset {folder.name}")
    return 0


# ---------------------------------------------------------------------------
# run / score / eval:读取已有数据集文件夹
# ---------------------------------------------------------------------------
def _resolve_folder(args: argparse.Namespace) -> Path:
    """把 --dataset 解析成已存在的数据集文件夹(workspace 模型)。"""
    global_cfg = workspace.load_global_config()
    ws = workspace.resolve_workspace(args.workspace, global_cfg)
    return workspace.resolve_dataset_dir(args.dataset, ws)


# CLI flag 名 -> 写回 config.yaml 的点号键(用户参数优先且持久化)。
_PERSIST_MAP: tuple[tuple[str, str], ...] = (
    ("base_url", "inference.openai.base_url"),
    ("model", "inference.openai.model"),
    ("backend", "inference.backend"),
    ("mnn_config", "inference.mnn.config_path"),
    ("mnn_image_max_side", "inference.mnn.image_max_side"),
    ("mnn_quant", "inference.mnn.quant"),
    ("hf_model", "inference.hf.model_path"),
    ("hf_image_max_side", "inference.hf.image_max_side"),
    ("vllm_model", "inference.vllm_offline.model_path"),
    ("vllm_gpu_util", "inference.vllm_offline.gpu_memory_utilization"),
    ("vllm_max_model_len", "inference.vllm_offline.max_model_len"),
    ("vllm_image_max_pixels", "inference.vllm_offline.image_max_pixels"),
    ("llamacpp_base_url", "inference.llamacpp.base_url"),
    ("llamacpp_model", "inference.llamacpp.model"),
    ("llamacpp_model_path", "inference.llamacpp.model_path"),
    ("llamacpp_mmproj", "inference.llamacpp.mmproj_path"),
    ("llamacpp_mode", "inference.llamacpp.mode"),
    ("scorer", "scoring.scorer"),
    ("prompt", "pred.prompt"),
    ("system_prompt", "pred.system_prompt"),
    ("label_extract_url", "label_extract.base_url"),
    ("label_extract_token", "label_extract.auth_token"),
    ("value_path", "label_extract.value_path"),
    ("match_mode", "label_extract.match_mode"),
    ("targets", "eval.targets"),
)


def _parse_targets(s: str) -> str | int:
    """把 --targets 的字符串转成合法值:all/last/first 原样保留,纯数字转成 int(第 N 轮)。

    这样 `--targets 2` 持久化为 eval.targets: 2(int),loader 据它选中第 2 个 assistant 轮。
    """
    t = str(s).strip()
    if t.isdigit():
        return int(t)
    return t


def _persist_overrides(folder: Path, args: argparse.Namespace) -> list[str]:
    """把用户显式提供的 CLI 覆盖永久写回该数据集 config.yaml,返回写回的键列表。

    只写回非 None 的 flag;某命令没有的 flag 自动跳过(getattr 兜底)。
    特别地，如果提供了 --model，调用 resolve_model_for_backend 根据当前 backend
    (CLI 指定优先，否则读取数据集已配置的 backend) 自动解析为对应后端的具体配置项。
    写回后由调用方 load_dataset_config 读回,实现「用户参数优先 + 持久化」。
    """
    persisted: list[str] = []

    # 1. 首先确定当前有效 backend 并处理 --backend
    backend_val = getattr(args, "backend", None)
    if backend_val is not None:
        workspace.set_dataset_value(folder, "inference.backend", backend_val)
        persisted.append("inference.backend")
    else:
        # 从该数据集已有的 config.yaml 窥探 backend (若有)
        try:
            cfg_existing = load_dataset_config(folder)
            backend_val = cfg_existing.inference.backend
        except Exception:
            backend_val = "openai"

    # 2. 如果提供了统一的 --model，智能分派到具体后端的配置项中
    model_val = getattr(args, "model", None)
    if model_val is not None:
        resolved = workspace.resolve_model_for_backend(backend_val, model_val)
        for d_key, d_val in resolved.items():
            workspace.set_dataset_value(folder, d_key, d_val)
            persisted.append(d_key)

    # 3. 处理其他常规或显式指定的参数 (若显式传了 --mnn-config 等，仍保持更高优先级的显式覆盖)
    for attr, dotted in _PERSIST_MAP:
        if attr in ("backend", "model"):
            continue
        val = getattr(args, attr, None)
        if val is not None:
            workspace.set_dataset_value(folder, dotted, val)
            persisted.append(dotted)
    return persisted


def _report_persist(tag: str, persisted: list[str], folder: Path) -> None:
    if persisted:
        print(f"[{tag}] 已将 {', '.join(persisted)} 写回 {folder / 'config.yaml'}(永久生效)")


def _maybe_label_extract(cfg: Config, args: argparse.Namespace) -> None:
    """若 --label-extract:描述完成后读 predictions.jsonl,调远程服务抽取标签落 label.jsonl。

    抽取失败按用户设定记 error 跳过、不中断整批(见 label_extract.run_label_extract)。
    """
    if not getattr(args, "label_extract", False):
        return
    stats = run_label_extract(cfg, overwrite=getattr(args, "overwrite", False))
    print(f"[label-extract] 完成 {stats['newly_completed']} 条,失败 {stats['errors']} 条,"
          f"跳过(已完成) {stats['skipped_already_done']} 条 -> {stats['labels_path']}")
    if stats["errors"]:
        print(f"[label-extract] 注意:有 {stats['errors']} 条抽取失败"
              f"(已记录 label_failures.jsonl,可重跑补齐)。")


def _do_run(cfg: Config, tag: str = "pred", limit: Optional[int] = None) -> dict:
    stats = run_inference(cfg, limit=limit)
    print(f"[{tag}] 完成 {stats['newly_completed']} 个目标轮,失败 {stats['errors']} 个,"
          f"跳过(已完成样本) {stats['skipped_samples_already_done']} 条 -> {cfg.predictions_path}")
    if stats["errors"]:
        print(f"[{tag}] 注意:有 {stats['errors']} 条推理失败(已记录 error,可重跑补齐)。")
    return stats


def _do_score(cfg: Config, scorer: Optional[str], limit: Optional[int] = None) -> dict:
    metrics = score_predictions(cfg, scorer_name=scorer, limit=limit)
    per_turn = metrics.get("per_turn") or {}
    print(f"[score] {len(per_turn)} 个目标轮,总体均分 {metrics.get('overall_mean_score')} "
          f"-> {cfg.metrics_path}")
    n_fail = metrics.get("num_failed_samples", 0)
    if n_fail:
        print(f"[score] exact_match 未命中 {n_fail} 个样本"
              f"(共 {metrics.get('num_failed_targets', 0)} 个错误轮),"
              f"人类可读清单 -> {cfg.failures_path}")
        print(f"[score] 错误样本可视化(HTML,含图片) -> {cfg.failures_html_path}")
    for turn_key, turn_data in per_turn.items():
        if "confusion_matrix" in turn_data:
            from .scoring.confusion_matrix import format_confusion_matrix_text
            print(format_confusion_matrix_text(
                turn_data["confusion_matrix"],
                title=f"混淆矩阵 — {turn_key} (scorer: {turn_data.get('scorer', '')})",
            ))
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return metrics


def _maybe_generate_mnn_report(cfg: Config) -> Optional[dict]:
    """若存在 predictions.jsonl 且为 mnn / llamacpp 后端(或含推理性能计时)，自动生成推理性能报告。"""
    if not cfg.predictions_path.exists():
        return None
    should_gen = cfg.inference.backend in ("mnn", "llamacpp")
    if not should_gen:
        try:
            with open(cfg.predictions_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        obj = json.loads(line)
                        raw = obj.get("raw") or {}
                        if raw.get("backend") in ("mnn", "llamacpp") or "vision_us" in raw or "prefill_us" in raw or "timings" in raw:
                            should_gen = True
                        break
        except Exception:
            pass
    if not should_gen:
        return None
    from .infer_stats import generate_inference_report
    try:
        summary, _ = generate_inference_report(cfg.predictions_path, out_dir=cfg.run_dir, print_report=True)
        return summary
    except Exception as e:
        print(f"[infer_stats] 性能分析跳过: {e}", file=sys.stderr)
        return None



def _pred_dataset(args: argparse.Namespace) -> int:
    """pred --dataset:对已有数据集的 test.json 执行推理(只预测,不评分)。

    等价于旧的 `run` 命令。--base-url/--model 等永久写回该数据集 config.yaml。
    """
    folder = _resolve_folder(args)
    persisted = _persist_overrides(folder, args)      # --base-url/--model 永久写回
    cfg = load_dataset_config(folder)
    cfg.inference.fail_fast = getattr(args, "fail_fast", False)  # 运行时,不写回 config
    _report_persist("pred", persisted, folder)
    print(f"[pred] 数据集预测,模型目录(按 模型/后端 区分)-> {cfg.run_dir}")
    limit = getattr(args, "limit", None)
    if limit is not None:
        _do_run(cfg, limit=limit)
    else:
        _do_run(cfg)
    _maybe_label_extract(cfg, args)
    _maybe_generate_mnn_report(cfg)
    return 0


def _cmd_score(args: argparse.Namespace) -> int:
    folder = _resolve_folder(args)
    persisted = _persist_overrides(folder, args)      # --scorer 等永久写回
    cfg = load_dataset_config(folder)
    _report_persist("score", persisted, folder)
    limit = getattr(args, "limit", None)
    if limit is not None:
        _do_score(cfg, args.scorer, limit=limit)
    else:
        _do_score(cfg, args.scorer)
    _maybe_generate_mnn_report(cfg)
    return 0


def run_field_eval_once(folder: Path, args: argparse.Namespace) -> dict:
    """对一个数据集跑 field-eval(自给自足:无预测先补跑 pred),返回结构化结果。

    与 _cmd_field_eval 同逻辑,只是多返回 {dataset, method, model, backend, metrics, report}
    供 sweep 汇总;单独跑 `eval-vlm field-eval` 时由 _cmd_field_eval 薄封装,行为不变。
    """
    persisted = _persist_overrides(folder, args)      # --backend/--vllm-model/--label-extract-* 等永久写回
    cfg = load_dataset_config(folder)
    cfg.inference.fail_fast = getattr(args, "fail_fast", False)  # 运行时,不写回 config
    _report_persist("field-eval", persisted, folder)

    # 自给自足:run_dir 由 cfg.inference.backend + result_name 决定。预测**完整覆盖 test 全部
    # 目标轮**才直接评;缺失/部分/无 都用当前 backend 补跑(run_inference 可续跑,已完成样本不重推)。
    need_pred = True
    if cfg.predictions_path.exists():
        try:
            samples = load_samples(cfg, source=cfg.test_path)
            expected = {(s.id, t.turn_index) for s in samples for t in s.targets}
            done = store.load_prediction_keys(cfg.predictions_path)
            need_pred = not (expected and expected.issubset(done))
        except Exception:  # noqa: BLE001 - 读不出就当作需要补跑,交给下游报确切错
            need_pred = True
    limit = getattr(args, "limit", None)
    if need_pred:
        print(f"[field-eval] 预测缺失/不完整,用 backend={cfg.inference.backend} 补跑 pred -> {cfg.run_dir}")
        if limit is not None:
            _do_run(cfg, "field-eval", limit=limit)
        else:
            _do_run(cfg, "field-eval")
    else:
        print(f"[field-eval] 复用已有完整预测(backend={cfg.inference.backend}) -> {cfg.predictions_path}")

    fe_kwargs = {"overwrite": getattr(args, "overwrite", False)}
    if limit is not None:
        fe_kwargs["limit"] = limit
    metrics = run_field_eval(cfg, **fe_kwargs)
    ov = metrics["overall"]
    print(f"[field-eval] 已评 {metrics['num_scored']} 样本(模型无输出判错 {metrics['num_pred_missing']},"
          f" 跳过 ref {metrics['skipped_ref']}/pred {metrics['skipped_pred_error']})"
          f" -> {cfg.field_metrics_path}")
    micro_ov = ov.get("micro_overall_accuracy", ov["micro_accuracy"])
    macro_ov = ov.get("macro_overall_accuracy", ov["macro_accuracy"])
    print(f"[field-eval] 非空 micro={ov['micro_accuracy']}  非空 macro={ov['macro_accuracy']}  "
          f"总体 micro={micro_ov}  总体 macro={macro_ov}  全对率={ov['exact_match_rate']}")
    for f in metrics["fields"]:
        pf = metrics["per_field"][f]
        ne_acc = f"{pf.get('non_empty_accuracy', pf['accuracy']) * 100:.2f}%"
        ov_acc = f"{pf.get('overall_accuracy', pf['accuracy']) * 100:.2f}%"
        ov_c = pf.get("overall_correct", pf["correct"])
        ov_t = pf.get("overall_total", metrics["num_scored"])
        empty_c = pf.get("empty_count", 0)
        ne_c = pf.get("non_empty_count", pf["total"])
        print(f"  - {f}: 非空 {ne_acc} ({pf['correct']}/{pf['total']}) | 总体 {ov_acc} ({ov_c}/{ov_t}) [空: {empty_c}, 非空: {ne_c}]")
        for v, d in (metrics.get("per_value", {}).get(f) or {}).items():
            print(f"      · {v}: {d['accuracy']}  ({d['correct']}/{d['support']})")
    for f, cm in (metrics.get("confusion_matrices") or {}).items():
        from .scoring.confusion_matrix import format_confusion_matrix_text
        print(format_confusion_matrix_text(cm, title=f"字段混淆矩阵 — {f}"))
    print(f"[field-eval] 失配清单 -> {cfg.field_mismatches_path}")
    print(f"[field-eval] 失配清单(HTML,含图片) -> {cfg.field_mismatches_html_path}")
    _maybe_generate_mnn_report(cfg)
    return {"dataset": folder.name, "method": "field-eval",
            "model": cfg.inference.result_name, "backend": cfg.inference.backend,
            "metrics": metrics, "report": str(cfg.field_summary_path)}


def _cmd_field_eval(args: argparse.Namespace) -> int:
    """field-eval:逐字段准确率。自给自足——用 config.inference.backend 解析 run_dir,
    没有 predictions.jsonl 就先用当前 backend 跑 pred,有则直接评(避免 pred/field-eval 后端不一致)。
    """
    folder = _resolve_folder(args)
    run_field_eval_once(folder, args)
    return 0


def run_eval_once(folder: Path, args: argparse.Namespace) -> dict:
    """对一个数据集跑 eval(预测 + 评分),返回结构化结果。

    与 _cmd_eval 同逻辑,只是多返回 {dataset, method, model, backend, metrics, report}
    供 sweep 汇总;单独跑 `eval-vlm eval` 时由 _cmd_eval 薄封装,行为不变。
    """
    persisted = _persist_overrides(folder, args)      # --base-url/--model/--scorer 永久写回
    cfg = load_dataset_config(folder)
    cfg.inference.fail_fast = getattr(args, "fail_fast", False)  # 运行时,不写回 config
    _report_persist("eval", persisted, folder)
    print(f"[eval] 模型目录(按 模型/后端 区分)-> {cfg.run_dir}")
    limit = getattr(args, "limit", None)
    if limit is not None:
        _do_run(cfg, "eval", limit=limit)
        metrics = _do_score(cfg, args.scorer, limit=limit)
    else:
        _do_run(cfg, "eval")
        metrics = _do_score(cfg, args.scorer)
    _maybe_generate_mnn_report(cfg)
    return {"dataset": folder.name, "method": "eval",
            "model": cfg.inference.result_name, "backend": cfg.inference.backend,
            "metrics": metrics, "report": str(cfg.summary_path)}


def _cmd_eval(args: argparse.Namespace) -> int:
    """一键连续执行 预测 + 评分(不含 split)。"""
    folder = _resolve_folder(args)
    run_eval_once(folder, args)
    return 0


def _cmd_sweep(args: argparse.Namespace) -> int:
    """批量评测:同一模型(后端)对一批数据集依次跑各自的 eval / field-eval。"""
    run_sweep(args)
    return 0


def _cmd_precision(args: argparse.Namespace) -> int:
    """对比 mnn(转换后)与 hf(转换前)两份预测,量化行为级精度误差并出报告。

    只读两个模型子目录下已生成的 predictions.jsonl(候选=MNN、参考=HF),
    因此候选/参考可在不同机器上分别用 `pred` 产出。默认模型名从 config.yaml 的
    inference.mnn / inference.hf 推断,可用 --candidate-dir / --reference-dir 覆盖。

    --dataset(run/dataset 格式)与 --datadir(pred --datadir 的对话格式)二选一;
    用 --datadir 时按 messages 末尾 assistant 轮解析预测。
    """
    datadir_format = args.datadir is not None
    # 两种入口都解析成"该 workspace 文件夹";_resolve_folder 读 args.dataset,
    # 故用 --datadir 时把它的值临时赋给 dataset 供解析(二者互斥,不会冲突)。
    if datadir_format:
        args.dataset = args.datadir
    folder = _resolve_folder(args)
    cfg = load_dataset_config(folder)
    summary = compare_precision(cfg, candidate=args.candidate_dir,
                                reference=args.reference_dir,
                                datadir_format=datadir_format)
    b = summary["behavior"]
    print(f"[precision] 候选 `{summary['candidate']}` vs 参考 `{summary['reference']}`:"
          f"对比 {summary['num_compared']} 条,输出一致率 {b['agreement_rate']:.1%},"
          f"平均 token-F1 {b['mean_token_f1']:.4f}")
    for f in summary["flags"]:
        print("  " + f)
    print(f"[precision] 报告 -> {summary['report_md']}")
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    """跨格式合并报告:扫描数据集下全部已跑格式,汇成一页质量门禁报告。

    纯读取已落盘产物(metrics/scored/precision/run_meta),与当前 backend 无关。
    """
    folder = _resolve_folder(args)
    cfg = load_dataset_config(folder)
    report = build_report(cfg)
    store.write_json(cfg.report_json_path, report)
    store.write_text(cfg.report_md_path, render_report_md(report))
    print(f"[report] 数据集 `{report['dataset']}`:发现 {report['num_formats']} 个格式,"
          f"HF 基准 {'有' if report['baseline'] else '无'}")
    for d in report["diagnosis"]:
        print("  " + d)
    print(f"[report] 合并报告 -> {cfg.report_md_path}")
    return 0


# ---------------------------------------------------------------------------
# pred:统一的「预测(不评分)」命令
#   --dataset = 对已有数据集的 test.json 推理(旧 run)
#   --datadir = 对无标注图片文件夹逐张描述(旧 pred)
# ---------------------------------------------------------------------------
def _cmd_pred(args: argparse.Namespace) -> int:
    """pred 分发:--dataset 走数据集预测;--datadir 走无标注图片描述。

    互斥参数组已保证二者恰好提供其一。
    """
    if args.dataset:
        return _pred_dataset(args)
    return _pred_datadir(args)


def _pred_datadir(args: argparse.Namespace) -> int:
    """遍历 --datadir 内所有图片,按 config.yaml 组织对话调 VLM 描述,落到 workspace/<同名>。

    自包含文件夹模型(同 split→run):首次运行在 workspace/<名>/ 生成 config.yaml
    (含 inference + pred 两段),再次运行直接读它。产物按模型落 workspace/<名>/<模型>/。
    CLI flag(--backend/--base-url/--model/--prompt 等)会永久写回该 config.yaml
    (用户参数优先且持久化);--overwrite 整份重跑覆盖已有结果。
    """
    global_cfg = workspace.load_global_config()
    ws = workspace.resolve_workspace(args.workspace, global_cfg)

    datadir = Path(args.datadir).expanduser().resolve()
    if not datadir.is_dir():
        raise FileNotFoundError(f"--datadir 不是文件夹: {datadir}")

    name = args.name or datadir.name
    out_dir = (ws / name).resolve()
    if out_dir == datadir:
        raise ValueError(
            f"输出文件夹与 --datadir 相同({out_dir});请用 --name 指定不同名字,"
            f"或把 workspace 设为别处(预测产物不应写回原图片文件夹)。"
        )

    # 生成(首次/--force)或沿用已有 config.yaml。
    config_path = out_dir / "config.yaml"
    existed = config_path.exists()
    workspace.init_pred_config(out_dir, datadir, global_cfg, force=args.force)
    action = "重新生成(--force)" if (existed and args.force) else ("沿用" if existed else "首次生成")
    print(f"[pred] {action}配置 -> {config_path}")

    # 用户 CLI 覆盖永久写回 config.yaml(--backend/--base-url/--model/--mnn-config/--prompt/--system-prompt),
    # 再读回成强类型 Config —— 用户参数优先且持久化。
    persisted = _persist_overrides(out_dir, args)
    cfg = load_dataset_config(out_dir)          # 读 config.yaml + 钉 dataset_dir=out_dir
    cfg.inference.fail_fast = getattr(args, "fail_fast", False)  # 运行时,不写回 config
    _report_persist("pred", persisted, out_dir)

    # 图片永远定位到 --datadir(即便 config 里 media_root 被改过)。
    cfg.data.media_root = str(datadir)
    print(f"[pred] 模型目录(按 模型/后端 区分)-> {cfg.run_dir}")

    # prompt/system_prompt 已写回 config,故 prompt=None 交给 cfg.pred 驱动。
    stats = predict_folder(cfg, datadir, prompt=None,
                           overwrite=getattr(args, "overwrite", False))
    print(f"[pred] 完成 {stats['newly_completed']} 张描述,失败 {stats['errors']} 张,"
          f"跳过(已完成) {stats['skipped_already_done']} 张 -> {stats['predictions_path']}")
    print(f"[pred] 人类可读视图 -> {cfg.run_dir / 'predictions.txt'}")
    if stats["errors"]:
        print(f"[pred] 注意:有 {stats['errors']} 张失败(已记录到 failures.jsonl,可重跑补齐)。")
    _maybe_label_extract(cfg, args)
    return 0


# ---------------------------------------------------------------------------
# infer:单图推理(mnn),只打印结果、不落盘
# ---------------------------------------------------------------------------
def _cmd_infer(args: argparse.Namespace) -> int:
    """单图推理:用 mnn 后端对一张图推理,把结果打印到控制台,**不保存任何文件**。

    复用现有 MNN 后端(与 pred/eval 同一套预处理:image_max/min_pixels 对齐
    LlamaFactory、默认贪心+重复惩罚)。只构造内存 Config,不读写 workspace/产物目录。
    状态信息走 stderr,推理结果走 stdout(便于管道取用)。
    """
    from .config import InferenceConfig, MNNBackendConfig
    from .data.schema import Turn
    from .inference import build_backend

    img_path = Path(args.img_path).expanduser()
    if not img_path.is_file():
        print(f"[infer] --img-path 不是文件: {img_path}", file=sys.stderr)
        return 2

    mnn = MNNBackendConfig(config_path=args.mnn_config)
    if args.max_pixels is not None:
        mnn.image_max_pixels = args.max_pixels
    if args.min_pixels is not None:
        mnn.image_min_pixels = args.min_pixels
    if args.max_tokens is not None:
        mnn.max_tokens = args.max_tokens
    if args.system_prompt is not None:
        mnn.system_prompt = args.system_prompt
    if args.temperature is not None:
        mnn.temperature = args.temperature
    if args.top_k is not None:
        mnn.top_k = args.top_k
    if args.top_p is not None:
        mnn.top_p = args.top_p
    cfg = Config(inference=InferenceConfig(backend="mnn", mnn=mnn))

    prompt = args.prompt
    content = prompt if "<image>" in prompt else f"<image>{prompt}"
    context = [Turn(role="user", content=content)]

    print(f"[infer] 加载 mnn 模型({args.mnn_config})...", file=sys.stderr, flush=True)
    backend = build_backend(cfg)
    try:
        # 绝对路径直接透传:resolve_image_path 对绝对路径原样返回,无需 media_root。
        pred = backend.complete(context, [str(img_path.resolve())], sample_id=img_path.name)
    finally:
        backend.close()

    if pred.error:
        print(f"[infer] 推理失败: {pred.error}", file=sys.stderr)
        return 1
    # print(pred.prediction or "")
    return pred


def _cmd_convert_gguf(args: argparse.Namespace) -> int:
    """HF 转 GGUF 命令:调用 llama.cpp 执行多模态/单模态转换并可选量化。"""
    from .gguf_convert import run_gguf_conversion

    try:
        res = run_gguf_conversion(
            hf_path=args.hf_path,
            name=args.name,
            out_dir=args.out_dir,
            outtype=args.outtype,
            is_multimodal=True if args.mmproj else (False if args.no_mmproj else None),
            mmproj_outtype=args.mmproj_outtype,
            mmproj_type=args.mmproj_type,
            quantize=args.quantize,
            clean_intermediate=args.clean_intermediate,
            llama_cpp_dir=args.llama_cpp_dir,
        )
        print(f"\n[convert-gguf] 转换成功: 模型 {res['model_name']} 落在 {res['out_dir']}")
        return 0
    except Exception as e:
        if getattr(args, "fail_fast", False):
            raise
        print(f"[convert-gguf] 转换失败: {e}", file=sys.stderr)
        return 1



# ---------------------------------------------------------------------------
# 参数
# ---------------------------------------------------------------------------
def _add_workspace_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument("--workspace", default=None,
                   help="覆盖全局配置中的 workspace(数据集文件夹的父目录)")


def _add_inference_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--base-url", default=None,
                   help="临时覆盖 inference.openai.base_url(部署地址,如 http://localhost:8000/v1)")
    p.add_argument("--model", default=None,
                   help="统一模型参数(权重文件夹/模型文件/模型名，底层自动根据当前后端解析为对应设置)")
    p.add_argument("--limit", type=int, default=None,
                   help="评测样本数量上限(截取前 N 条;<=0 表示不限制,缺省评全部)")
    # 调试用:任一条推理出错立即抛出原始异常(带完整 traceback)并中断整批,不再默默记
    # error 跑完全程。仅运行时生效,不写回 config.yaml。默认关闭(保持容错、可断点续跑)。
    p.add_argument("--fail-fast", dest="fail_fast", action="store_true",
                   help="任一条推理出错就立即抛出完整 traceback 并中断(调试用;默认记 error 继续跑)")


def _add_vllm_offline_args(p: argparse.ArgumentParser) -> None:
    """backend=vllm_offline(离线 vLLM 进程内加载合并权重)的参数。"""
    p.add_argument("--vllm-model", dest="vllm_model", default=None,
                   help="backend=vllm_offline 时:合并后全精度权重目录"
                        "(临时覆盖 inference.vllm_offline.model_path;也用作产物子目录名)")
    p.add_argument("--vllm-gpu-util", dest="vllm_gpu_util", type=float, default=None,
                   help="backend=vllm_offline 时:GPU 显存利用率 0~1"
                        "(临时覆盖 inference.vllm_offline.gpu_memory_utilization)")
    p.add_argument("--vllm-max-model-len", dest="vllm_max_model_len", type=int, default=None,
                   help="backend=vllm_offline 时:最大序列长度"
                        "(临时覆盖 inference.vllm_offline.max_model_len)")
    p.add_argument("--vllm-image-max-pixels", dest="vllm_image_max_pixels", type=int, default=None,
                   help="backend=vllm_offline 时:图片像素上限,建议与训练 image_max_pixels 对齐"
                        "(临时覆盖 inference.vllm_offline.image_max_pixels)")


def _add_llamacpp_args(p: argparse.ArgumentParser) -> None:
    """backend=llamacpp(llama.cpp HTTP 服务或本地 mtmd-cli)的参数。"""
    p.add_argument("--llamacpp-base-url", dest="llamacpp_base_url", default=None,
                   help="backend=llamacpp 时:llama-server 服务地址(如 http://127.0.0.1:8080/v1;"
                        "临时覆盖 inference.llamacpp.base_url)")
    p.add_argument("--llamacpp-model", dest="llamacpp_model", default=None,
                   help="backend=llamacpp 时:模型逻辑名称或别名(临时覆盖 inference.llamacpp.model)")
    p.add_argument("--llamacpp-model-path", dest="llamacpp_model_path", default=None,
                   help="backend=llamacpp 时:本地 GGUF 语言模型文件路径(临时覆盖 inference.llamacpp.model_path)")
    p.add_argument("--llamacpp-mmproj", dest="llamacpp_mmproj", default=None,
                   help="backend=llamacpp 时:本地 GGUF 多模态投影器路径(临时覆盖 inference.llamacpp.mmproj_path)")
    p.add_argument("--llamacpp-mode", dest="llamacpp_mode", default=None, choices=["server", "cli"],
                   help="backend=llamacpp 时:运行模式 server(连 HTTP 服务)或 cli(调 mtmd-cli 可执行文件)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eval_vlm",
        description="解耦的 VLM 测试集评测工具(工作目录模型:config / split / pred / score / eval)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # config
    p_config = sub.add_parser("config", help="管理全局配置(workspace/media_root 等)")
    p_config.add_argument("action", choices=["init", "show", "set", "keys"],
                          help="init=生成默认 / show=查看当前 / set=改一个键 / keys=列出所有可设置键")
    p_config.add_argument("key", nargs="?", default=None,
                          help="set 时的键:workspace/media_root/image_strip_prefix,"
                               "或 split 默认 split.train/split.test/split.val/split.seed/split.stratify_by")
    p_config.add_argument("value", nargs="?", default=None, help="set 时的值(null 表示清空)")
    p_config.add_argument("--force", action="store_true", help="init 时覆盖已有全局配置")
    p_config.set_defaults(func=_cmd_config)

    # convert-gguf(HF 转 GGUF)
    p_conv = sub.add_parser("convert-gguf", help="将 HuggingFace 格式模型转换为 GGUF 格式(支持多模态两阶段导出与量化)")
    p_conv.add_argument("--hf-path", required=True, help="源 HuggingFace 模型文件夹路径")
    p_conv.add_argument("--name", "-n", default=None,
                        help="目标模型名 A(产物落地 <llamacpp_models_dir>/<A>/;缺省从 hf-path 提取)")
    p_conv.add_argument("--out-dir", default=None,
                        help="明确指定输出目录(缺省自动取 <llamacpp_models_dir>/<A>/)")
    p_conv.add_argument("--outtype", default="bf16", choices=["bf16", "f16", "f32"],
                        help="主模型权重导出精度(默认 bf16)")
    p_conv.add_argument("--mmproj", action="store_true", default=False,
                        help="强制导出多模态投影器(默认对含视觉配置的 HF 目录自动开启)")
    p_conv.add_argument("--no-mmproj", action="store_true", default=False,
                        help="强制禁用多模态投影器导出(纯文本语言模型)")
    p_conv.add_argument("--mmproj-outtype", default="f16", choices=["f16", "bf16", "f32"],
                        help="多模态投影器导出精度(默认 f16)")
    p_conv.add_argument("--mmproj-type", default=None,
                        help="多模态投影器架构类型(如 auto, qwen2vl, minicpmv, llava, clip 等;缺省自动检测)")
    p_conv.add_argument("--quantize", "-q", default=None,
                        help="llama-quantize 量化格式(如 Q4_K_M, Q8_0, Q5_K_M 等;投影器保持不量化)")
    p_conv.add_argument("--clean-intermediate", action="store_true", default=False,
                        help="量化后删除未量化的基座 gguf 临时文件")
    p_conv.add_argument("--llama-cpp-dir", default=None,
                        help="llama.cpp 源码/编译根目录(缺省自动查找系统 PATH 或常见位置)")
    _add_workspace_arg(p_conv)
    p_conv.set_defaults(func=_cmd_convert_gguf)

    # split(初始化)
    p_split = sub.add_parser("split", help="初始化数据集文件夹并分割(--dataset=源JSON)")
    p_split.add_argument("--dataset", "-d", required=True, help="源数据集 JSON 文件路径")
    p_split.add_argument("--name", default=None, help="数据集文件夹名(默认取源文件名,不含扩展名)")
    p_split.add_argument("--train", type=float, default=None,
                         help="训练集比例(如 0.8);不传则用全局配置 split.train")
    p_split.add_argument("--test", type=float, default=None,
                         help="测试集比例(如 0.2);不传则用全局配置 split.test")
    p_split.add_argument("--val", type=float, default=None,
                         help="验证集比例(>0 才产出 val.json);不传则用全局配置 split.val")
    p_split.add_argument("--seed", type=int, default=None,
                         help="随机种子(可复现);不传则用全局配置 split.seed")
    p_split.add_argument("--stratify-by", dest="stratify_by", default=None,
                         help="分层抽样字段名(如标签字段);不传则用全局配置 split.stratify_by")
    p_split.add_argument("--train-out", nargs="?", const="", default=None,
                         help="覆盖 train 输出:带路径=写该路径;不带路径(光杆旗标)="
                              "写到全局 train_out_dir/<数据集名>_train.json")
    p_split.add_argument("--val-out", nargs="?", const="", default=None,
                         help="覆盖 val 输出:带路径=写该路径;不带路径=写到全局 val_out_dir/<数据集名>_val.json")
    p_split.add_argument("--test-out", nargs="?", const="", default=None,
                         help="覆盖 test 输出:带路径=写该路径;不带路径=写到全局 test_out_dir/<数据集名>_test.json")
    p_split.add_argument("--force", action="store_true", help="数据集文件夹已存在时重建(覆盖 config.yaml)")
    _add_workspace_arg(p_split)
    p_split.set_defaults(func=_cmd_split)

    # score
    p_score = sub.add_parser("score", help="对已有数据集的预测评分(--dataset=名|路径)")
    p_score.add_argument("--dataset", "-d", required=True, help="数据集名(或文件夹路径)")
    p_score.add_argument("--model", default=None,
                         help="统一模型参数(权重文件夹/模型文件/模型名，定位对应后端的 predictions.jsonl)")
    p_score.add_argument("--scorer", default=None,
                         help=f"临时覆盖评分器。可用: {', '.join(available_scorers())}")
    p_score.add_argument("--limit", type=int, default=None,
                         help="评分样本数量上限(截取前 N 条;<=0 表示不限制)")
    p_score.add_argument("--targets", dest="targets", default=None, type=_parse_targets,
                         help="评哪个 assistant 轮:all/last/first,或数字(如 2=第2轮,1 起始)")
    p_score.add_argument("--backend", default=None,
                         choices=["openai", "vllm", "mnn", "hf", "vllm_offline", "llamacpp", "fake"],
                         help="临时覆盖推理后端(写回 inference.backend);定位对应后端的 predictions.jsonl")
    p_score.add_argument("--mnn-config", dest="mnn_config", default=None,
                         help="backend=mnn 时:转换产物目录里 config.json 路径(写回 inference.mnn.config_path)")
    _add_workspace_arg(p_score)
    p_score.set_defaults(func=_cmd_score)

    # field-eval:指定轮描述(默认第一轮,eval.targets 决定) -> value-extract 服务抽固定字段 -> 逐字段准确率
    p_fe = sub.add_parser(
        "field-eval",
        help="描述轮逐字段准确率(轮由 eval.targets 决定,默认第一轮):无预测则自动用当前 backend 跑 pred,再发 value-extract 抽字段严格比对(--dataset=名|路径)")
    p_fe.add_argument("--dataset", "-d", required=True, help="数据集名(或文件夹路径)")
    p_fe.add_argument("--targets", dest="targets", default=None, type=_parse_targets,
                      help="评哪个 assistant 轮:all/last/first,或数字(如 2=第2轮,1 起始)"
                           "(永久写回 eval.targets;field-eval 与自动补 pred 都按此轮)")
    p_fe.add_argument("--overwrite", action="store_true",
                      help="无视已有 fields_ref/fields_pred 整份重抽;默认断点续跑只补未完成")
    p_fe.add_argument("--label-extract-url", dest="label_extract_url", default=None,
                      help="临时覆盖抽取服务地址 base_url(永久写回 label_extract.base_url)")
    p_fe.add_argument("--label-extract-token", dest="label_extract_token", default=None,
                      help="临时覆盖 Authorization 头(需含 bearer 前缀;永久写回 label_extract.auth_token)")
    p_fe.add_argument("--value-path", dest="value_path", default=None,
                      help="临时覆盖 value-extract 路由路径(永久写回 label_extract.value_path;"
                           "不同数据集的固定枚举字段结构不同时可指定各自的抽取路由)")
    p_fe.add_argument("--match-mode", dest="match_mode", default=None,
                      choices=["exact", "contain", "subset"],
                      help="field-eval 匹配判定模式:exact(严格双向相等)|contain/subset(包含真值即对,模型抽取词包含真值)"
                           "(永久写回 label_extract.match_mode)")
    # field-eval 自给自足:没预测就用当前 backend 先跑 pred。下面这些 flag 指定/覆盖该 backend
    # 及其模型(与 pred/eval 一致,持久化到 config.yaml),既决定 run_dir 也决定自动 pred 用哪个后端。
    p_fe.add_argument("--backend", default=None,
                      choices=["openai", "vllm", "mnn", "hf", "vllm_offline", "llamacpp", "fake"],
                      help="临时覆盖推理后端(写回 inference.backend);决定 run_dir 与自动 pred 用哪个后端")
    p_fe.add_argument("--hf-model", dest="hf_model", default=None,
                      help="backend=hf 时:本地 HF 权重目录(写回 inference.hf.model_path)")
    p_fe.add_argument("--mnn-config", dest="mnn_config", default=None,
                      help="backend=mnn 时:转换产物 config.json 路径(写回 inference.mnn.config_path)")
    _add_inference_args(p_fe)          # --base-url / --model / --fail-fast
    _add_vllm_offline_args(p_fe)       # --vllm-model / --vllm-gpu-util / --vllm-max-model-len / --vllm-image-max-pixels
    _add_llamacpp_args(p_fe)           # --llamacpp-base-url / --llamacpp-model / --llamacpp-model-path / --llamacpp-mmproj / --llamacpp-mode
    _add_workspace_arg(p_fe)
    p_fe.set_defaults(func=_cmd_field_eval)

    # eval = run + score
    p_eval = sub.add_parser("eval", help="一键连续执行 run + score(不含 split)")
    p_eval.add_argument("--dataset", "-d", required=True, help="数据集名(或文件夹路径)")
    p_eval.add_argument("--targets", dest="targets", default=None, type=_parse_targets,
                         help="评哪个 assistant 轮:all/last/first,或数字(如 2=第2轮,1 起始)")
    _add_inference_args(p_eval)
    p_eval.add_argument("--scorer", default=None, help="临时覆盖评分器")
    # 后端 / 权重 flag(与 pred 一致,永久写回 config.yaml):让每个格式一条 eval 即可跑完 pred+score
    p_eval.add_argument("--backend", default=None,
                        choices=["openai", "vllm", "mnn", "hf", "vllm_offline", "llamacpp", "fake"],
                        help="临时覆盖推理后端(写回 inference.backend);openai/vllm/mnn/hf/vllm_offline/llamacpp/fake")
    p_eval.add_argument("--hf-model", dest="hf_model", default=None,
                        help="backend=hf 时:本地 HF 权重目录(写回 inference.hf.model_path;也用作产物子目录名)")
    p_eval.add_argument("--hf-image-max-side", dest="hf_image_max_side", type=int, default=None,
                        help="backend=hf 时:图片最长边像素上限,纯等比缩放不 patch 对齐"
                             "(适合 MiniCPM-V 等非 Qwen 切片模型;设 0 关闭)"
                             "(写回 inference.hf.image_max_side)")
    p_eval.add_argument("--mnn-config", dest="mnn_config", default=None,
                        help="backend=mnn 时:转换产物目录里 config.json 路径(写回 inference.mnn.config_path)")
    p_eval.add_argument("--mnn-image-max-side", dest="mnn_image_max_side", type=int, default=None,
                        help="backend=mnn 时:图片最长边像素上限(写回 inference.mnn.image_max_side)")
    p_eval.add_argument("--mnn-quant", dest="mnn_quant", default=None,
                        help="backend=mnn 时:量化配方标签(如 hqq-4bit/hqq-8bit),供 report 标注(写回 inference.mnn.quant)")
    _add_vllm_offline_args(p_eval)
    _add_llamacpp_args(p_eval)
    _add_workspace_arg(p_eval)
    p_eval.set_defaults(func=_cmd_eval)

    # sweep = 同一模型对一批数据集依次跑各自的 eval / field-eval(方法由各数据集 eval.method 决定)
    p_sweep = sub.add_parser(
        "sweep",
        help="批量评测:同一模型(后端)对一批数据集依次跑各自的 eval/field-eval(方法由各数据集 eval.method 决定)")
    p_sweep.add_argument("--dataset", "-d", default=None,
                         help="要跑的数据集名(逗号分隔,可重复);可与 --dataset-list 并用")
    p_sweep.add_argument("--dataset-list", dest="dataset_list", default=None,
                         help="数据集名清单文件(每行一个,# 开头为注释);可与 --dataset 并用")
    p_sweep.add_argument("--method", default=None, choices=["eval", "field-eval"],
                         help="批量覆盖每个数据集的 eval.method 并永久写回(不传则各数据集用自己配置里的值)")
    p_sweep.add_argument("--overwrite", action="store_true",
                         help="转给 field-eval 类型数据集:无视已有 fields_ref/fields_pred 整份重抽")
    p_sweep.add_argument("--stop-on-error", dest="stop_on_error", action="store_true",
                         help="任一数据集失败即整批中止(默认记 error 继续下一个)")
    p_sweep.add_argument("--dry-run", dest="dry_run", action="store_true",
                         help="只打印「数据集 -> 方法」计划表,不执行任何评测")
    p_sweep.add_argument("--scorer", default=None, help="转给 eval 类型数据集:临时覆盖评分器")
    p_sweep.add_argument("--label-extract-url", dest="label_extract_url", default=None,
                         help="转给 field-eval 类型数据集:覆盖抽取服务地址 base_url")
    p_sweep.add_argument("--label-extract-token", dest="label_extract_token", default=None,
                         help="转给 field-eval 类型数据集:覆盖 Authorization 头(需含 bearer 前缀)")
    p_sweep.add_argument("--value-path", dest="value_path", default=None,
                         help="转给 field-eval 类型数据集:覆盖 value-extract 路由路径(永久写回 label_extract.value_path)")
    p_sweep.add_argument("--match-mode", dest="match_mode", default=None,
                         choices=["exact", "contain", "subset"],
                         help="转给 field-eval 类型数据集:覆盖匹配判定模式 exact|contain(永久写回 label_extract.match_mode)")
    p_sweep.add_argument("--targets", dest="targets", default=None, type=_parse_targets,
                         help="批量覆盖每个数据集的 eval.targets(all/last/first/数字如 2=第2轮)并永久写回")
    p_sweep.add_argument("--backend", default=None,
                         choices=["openai", "vllm", "mnn", "hf", "vllm_offline", "llamacpp", "fake"],
                         help="临时覆盖推理后端(写回 inference.backend);决定 run_dir 与自动 pred 用哪个后端")
    p_sweep.add_argument("--hf-model", dest="hf_model", default=None,
                         help="backend=hf 时:本地 HF 权重目录(写回 inference.hf.model_path)")
    p_sweep.add_argument("--mnn-config", dest="mnn_config", default=None,
                         help="backend=mnn 时:转换产物 config.json 路径(写回 inference.mnn.config_path)")
    _add_inference_args(p_sweep)       # --base-url / --model / --fail-fast
    _add_vllm_offline_args(p_sweep)    # --vllm-model / --vllm-gpu-util / --vllm-max-model-len / --vllm-image-max-pixels
    _add_llamacpp_args(p_sweep)
    _add_workspace_arg(p_sweep)
    p_sweep.set_defaults(func=_cmd_sweep)

    # precision(对比 mnn 转换后 vs hf 转换前 的行为级精度误差)
    p_prec = sub.add_parser(
        "precision",
        help="对比 mnn(转换后) vs hf(转换前) 的行为级精度误差:读两份 predictions.jsonl 出报告")
    prec_src = p_prec.add_mutually_exclusive_group(required=True)
    prec_src.add_argument("--dataset", "-d", default=None,
                          help="数据集名(或文件夹路径):预测为 run / pred --dataset 格式")
    prec_src.add_argument("--datadir", default=None,
                          help="pred --datadir 产出的文件夹名(或路径):预测为 LlamaFactory 对话格式,"
                               "按 messages 末尾 assistant 轮解析(否则会读成空串、误报 100%% 一致)")
    p_prec.add_argument("--candidate-dir", dest="candidate_dir", default=None,
                        help="候选(转换后·MNN)模型子目录名;默认取 inference.mnn 的产物目录名")
    p_prec.add_argument("--reference-dir", dest="reference_dir", default=None,
                        help="参考(转换前·HF)模型子目录名;默认取 inference.hf.model_path 的目录名")
    _add_workspace_arg(p_prec)
    p_prec.set_defaults(func=_cmd_precision)

    # report(跨格式合并质量报告:HF vs 各 MNN 变体,读全部已跑产物)
    p_report = sub.add_parser(
        "report",
        help="跨格式合并质量报告:扫描数据集下全部已跑格式,出质量并排 + 净质量Δ + 诊断(纯读取)")
    p_report.add_argument("--dataset", "-d", required=True, help="数据集名(或文件夹路径)")
    _add_workspace_arg(p_report)
    p_report.set_defaults(func=_cmd_report)

    # compare: review source images/gold/model outputs side-by-side.  Unlike
    # report it never writes into a run directory or invokes inference/scoring.
    p_compare = sub.add_parser("compare", help="多模型样本级对比：默认只导出预测有差异的样本")
    p_compare.add_argument("--dataset", "-d", required=True, help="数据集名(或文件夹路径)")
    p_compare.add_argument("--runs", required=True, nargs="+", action="append", help="两个或更多 Run，格式 model/backend；可重复传入")
    p_compare.add_argument("--baseline", default=None, help="基准 Run，默认 --runs 的第一个")
    p_compare.add_argument("--filter", choices=["regression", "improvement", "correctness_disagreement", "text_disagreement", "field_disagreement", "all_wrong", "missing_or_error"], default=None)
    p_compare.add_argument("--sort", choices=["priority", "score_delta", "id"], default="priority")
    p_compare.add_argument("--descending", action="store_true", help="反向排序")
    p_compare.add_argument("--include-agreements", action="store_true", help="包含所有模型预测一致的样本")
    p_compare.add_argument("--query", default=None, help="按 sample id、真值或预测全文搜索")
    p_compare.add_argument("--limit", type=int, default=None)
    p_compare.add_argument("--offset", type=int, default=0)
    p_compare.add_argument("--output", default=None, help="输出 comparison.json/jsonl/html 的目录")
    p_compare.add_argument("--allow-mixed-dataset", action="store_true", help="允许 Run 的 test SHA 与当前数据集不一致")
    _add_workspace_arg(p_compare)
    p_compare.set_defaults(func=_cmd_compare)

    # infer(单图推理:mnn 后端,只打印结果不落盘)
    p_infer = sub.add_parser(
        "infer",
        help="单图推理(mnn 后端):对一张图推理并打印结果,不保存任何文件")
    p_infer.add_argument("--mnn-config", dest="mnn_config", required=True,
                         help="转换后 mnn 模型目录里的 config.json 路径(传给 MNN.llm.create)")
    p_infer.add_argument("--img-path", dest="img_path", required=True, help="单张图片路径")
    p_infer.add_argument("--max-pixels", dest="max_pixels", type=int, default=None,
                         help="图片总像素上限(超过按 sqrt 因子缩小,对齐 LlamaFactory;缺省用 mnn 默认 589824;<=0 关闭)")
    p_infer.add_argument("--min-pixels", dest="min_pixels", type=int, default=None,
                         help="图片总像素下限(不足按 sqrt 因子放大;缺省用 mnn 默认 1024;<=0 关闭)")
    p_infer.add_argument("--prompt", default=DEFAULT_PROMPT,
                         help=f"提问文本(缺省 {DEFAULT_PROMPT!r});不含 <image> 时自动前置")
    p_infer.add_argument("--system-prompt", dest="system_prompt", default=None,
                         help="系统提示(应与训练一致;缺省沿用模型 config.json)")
    p_infer.add_argument("--max-tokens", dest="max_tokens", type=int, default=1024,
                         help="生成上限(缺省 1024)")
    p_infer.add_argument("--temperature", dest="temperature", type=float, default=0.8,
                             help="0 -> 确定性; 大于0 -> 温度随机采样")
    p_infer.add_argument("--top_k", dest="top_k", type=float, default=0.8,
                             help="top-k 截断(只在前 K 个候选里选);仅随机采样(temperature>0)有意义")
    p_infer.add_argument("--top_p", dest="top_p", type=float, default=0.8,
                             help="nucleus 截断(累积概率到 p);仅随机采样(temperature>0)有意义")
    p_infer.set_defaults(func=_cmd_infer)

    # pred(统一预测命令:--dataset=数据集 test.json | --datadir=无标注图片文件夹)
    p_pred = sub.add_parser(
        "pred",
        help="预测(不评分):--dataset=对数据集 test.json 推理 | --datadir=对无标注图片文件夹逐张描述")
    src = p_pred.add_mutually_exclusive_group(required=True)
    src.add_argument("--dataset", default=None,
                     help="数据集名(或文件夹路径):对其 test.json 推理(等价旧 run,只预测不评分)")
    src.add_argument("--datadir", default=None,
                     help="无标注图片文件夹路径:逐张单轮描述(等价旧 pred),产物落 workspace/<同名>")
    p_pred.add_argument("--name", default=None,
                        help="[--datadir] 输出文件夹名(默认取图片文件夹名);产物落在 workspace/<名>/")
    p_pred.add_argument("--prompt", default=None,
                        help=f"[--datadir] 临时覆盖单轮提示词(不传则用文件夹 config.yaml 的 pred.prompt,"
                             f"默认 {DEFAULT_PROMPT!r};config 里设了多轮 template 时此项无效)")
    p_pred.add_argument("--system-prompt", dest="system_prompt", default=None,
                        help="[--datadir] 临时覆盖系统提示(不传则用 config.yaml 的 pred.system_prompt)")
    p_pred.add_argument("--backend", default=None,
                        choices=["openai", "vllm", "mnn", "hf", "vllm_offline", "llamacpp", "fake"],
                        help="临时覆盖推理后端:openai/vllm(调 OpenAI 兼容 API,vllm 为别名)| "
                             "mnn(本地 pymnn 推理转换后的 mnn 模型,需 --mnn-config)| "
                             "hf(本地 transformers 推理,作转换前参考基准,需 --hf-model)| "
                             "vllm_offline(本地离线 vLLM 进程内加载合并权重,需 --vllm-model)| "
                             "llamacpp(本地或远端 llama.cpp/llama-server 多模态推理)| "
                             "fake(回显,不联网,自检用)")
    p_pred.add_argument("--mnn-config", dest="mnn_config", default=None,
                        help="backend=mnn 时:转换产物目录里 config.json 的路径"
                             "(临时覆盖 inference.mnn.config_path)")
    p_pred.add_argument("--hf-model", dest="hf_model", default=None,
                        help="backend=hf 时:本地 HF 权重目录路径"
                             "(临时覆盖 inference.hf.model_path;也用作产物子目录名)")
    p_pred.add_argument("--hf-image-max-side", dest="hf_image_max_side", type=int,
                        default=None,
                        help="backend=hf 时:图片最长边像素上限,纯等比缩放不 patch 对齐"
                             "(适合 MiniCPM-V 等非 Qwen 切片模型;需把 hf.image_max/min_pixels "
                             "设 0);设 0 关闭。临时覆盖 inference.hf.image_max_side")
    p_pred.add_argument("--mnn-image-max-side", dest="mnn_image_max_side", type=int,
                        default=None,
                        help="backend=mnn 时:图片最长边像素上限(超大图等比缩放;默认 2048;"
                             "设 0 关闭缩放)。临时覆盖 inference.mnn.image_max_side")
    p_pred.add_argument("--mnn-quant", dest="mnn_quant", default=None,
                        help="backend=mnn 时:量化配方标签(如 hqq-4bit / hqq-8bit),纯记录用,"
                             "落进 run_meta/pred_meta 供 report 标注(永久写回 inference.mnn.quant)")
    p_pred.add_argument("--force", action="store_true",
                        help="[--datadir] 重新生成文件夹内 config.yaml(覆盖你的手改)")
    p_pred.add_argument("--overwrite", action="store_true",
                        help="[--datadir] 无视已有结果整份重跑(覆盖 predictions.jsonl);默认断点续跑只补未完成")
    p_pred.add_argument("--label-extract", dest="label_extract", action="store_true",
                        help="描述完成后,把每张图的描述发给远程标签服务抽取结构化标签,"
                             "结果存 label.jsonl(每行 {image, labels});失败记 label_failures.jsonl 可重跑。"
                             "--overwrite 同时整份重抽标签")
    p_pred.add_argument("--label-extract-url", dest="label_extract_url", default=None,
                        help="临时覆盖标签服务地址 base_url(永久写回 label_extract.base_url)")
    p_pred.add_argument("--label-extract-token", dest="label_extract_token", default=None,
                         help="临时覆盖 Authorization 头(需含 bearer 前缀;永久写回 label_extract.auth_token)")
    p_pred.add_argument("--targets", dest="targets", default=None, type=_parse_targets,
                         help="评哪个 assistant 轮:all/last/first,或数字(如 2=第2轮,1 起始)")
    _add_inference_args(p_pred)
    _add_vllm_offline_args(p_pred)
    _add_llamacpp_args(p_pred)
    _add_workspace_arg(p_pred)
    p_pred.set_defaults(func=_cmd_pred)

    return parser


def _force_utf8_stdout() -> None:
    """Windows 控制台默认 GBK,会把中文输出变乱码;强制 UTF-8。"""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8")
            except Exception:  # noqa: BLE001
                pass


def main(argv: Optional[list[str]] = None) -> int:
    _force_utf8_stdout()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (FileExistsError, FileNotFoundError, ValueError, KeyError) as e:
        # --fail-fast:用户明确要看完整 traceback,连这些「已知用户侧错误」也照抛不拦。
        if getattr(args, "fail_fast", False):
            raise
        # 已知的用户侧错误:打印简洁信息,不抛完整堆栈。
        print(f"[error] {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
