"""sweep 命令:同一个模型(后端)对一批数据集依次跑各自的评测方法(eval / field-eval)。

每个数据集该用哪种方法,由该数据集 config.yaml 的 `eval.method` 字段决定(eval 默认 /
field-eval)。数据集列表**显式给出**(--dataset 逗号分隔 和/或 --dataset-list 每行一个),
顺序跑、不并发(mnn/hf/vllm_offline 是进程内加载模型,并发会重复占显存/内存;openai/vllm
的请求级并发已在各自后端 max_concurrency 里)。单个数据集失败记录后继续下一个,
--stop-on-error 才整批中止。

一次 sweep = 一个模型在多个数据集上的结果。汇总报告落
`<workspace>/_sweep/<模型>/<后端>/summary.json` + `summary.md`,其中 <模型>/<后端> 取自
第一个成功跑完的数据集(同一模型下对所有数据集应一致)。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

from . import workspace
from .config import safe_model_dirname, load_dataset_config
from .results import store as results_store


def _read_dataset_names(args) -> list[str]:
    """合并 --dataset(逗号分隔)与 --dataset-list(每行一个,支持空行/# 注释),去重保序。"""
    names: list[str] = []
    seen: set[str] = set()

    def add(name: str) -> None:
        name = name.strip()
        if name and name not in seen:
            seen.add(name)
            names.append(name)

    raw = getattr(args, "dataset", None)
    if raw:
        for part in str(raw).split(","):
            add(part)

    list_path = getattr(args, "dataset_list", None)
    if list_path:
        p = Path(list_path).expanduser()
        if not p.is_file():
            raise FileNotFoundError(f"--dataset-list 不是文件: {p}")
        for raw_line in p.read_text(encoding="utf-8").splitlines():
            line = raw_line.split("#", 1)[0].strip()   # 去掉行内 # 注释
            add(line)

    if not names:
        raise ValueError(
            "未指定数据集:请用 --dataset a,b,c 或 --dataset-list <文件> 提供至少一个数据集名"
        )
    return names


def _key_metric(r: dict) -> str:
    """从一个结果记录提炼关键指标,供汇总表展示。"""
    m = r.get("metrics") or {}
    if r.get("method") == "field-eval":
        ov = m.get("overall") or {}
        return (f"micro={ov.get('micro_accuracy')}  macro={ov.get('macro_accuracy')}  "
                f"全对={ov.get('exact_match_rate')}")
    return f"overall_mean={m.get('overall_mean_score')}"


def _rel(path_str: str, base: Path) -> str:
    """把绝对路径渲染成相对 workspace 的路径(跨目录搬移不破坏展示)。"""
    if not path_str:
        return "-"
    try:
        return os.path.relpath(path_str, base)
    except ValueError:          # 跨盘符(Windows)无法算相对 -> 退原样
        return path_str


def _render_sweep_summary(summary: dict, ws: Path) -> str:
    results = summary["results"]
    lines = [
        "# sweep 汇总",
        "",
        f"- 数据集: {len(results)} 个,成功 {summary['num_ok']},失败 {summary['num_error']}",
        "",
        "| 数据集 | 方法 | 状态 | 关键指标 | 详情报告 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in results:
        name, method, status = r["dataset"], r["method"], r["status"]
        if status == "error":
            metric = r.get("error", "")
            report = "-"
        else:
            metric = _key_metric(r)
            report = _rel(r.get("report", ""), ws)
        lines.append(f"| {name} | {method} | {status} | {metric} | {report} |")
    lines.append("")
    return "\n".join(lines)


def run_sweep(args) -> dict:
    """批量跑一批数据集,汇总并落盘,返回 {datasets, results, num_ok, num_error}。

    延迟 import cli.run_eval_once/run_field_eval_once,避免 cli ↔ sweep 循环导入。
    """
    from .cli import run_eval_once, run_field_eval_once

    global_cfg = workspace.load_global_config()
    ws = workspace.resolve_workspace(getattr(args, "workspace", None), global_cfg)
    names = _read_dataset_names(args)
    method_override = getattr(args, "method", None)

    # 先解析全部数据集 + 其方法,打印计划表(dry-run 只到这一步)。
    plan: list[tuple[str, Optional[Path], str]] = []
    for name in names:
        try:
            folder = workspace.resolve_dataset_dir(name, ws)
        except FileNotFoundError:
            plan.append((name, None, "?"))
            continue
        if method_override:
            workspace.set_dataset_value(folder, "eval.method", method_override)
        cfg = load_dataset_config(folder)
        plan.append((name, folder, cfg.eval.method))

    print(f"[sweep] 共 {len(plan)} 个数据集:")
    for name, folder, method in plan:
        print(f"  - {name}: {method}"
              + ("  ⚠️ 数据集不存在" if folder is None else ""))
    if getattr(args, "dry_run", False):
        return {"datasets": [n for n, _, _ in plan], "dry_run": True,
                "results": [], "num_ok": 0, "num_error": 0}

    results: list[dict] = []
    for name, folder, method in plan:
        if folder is None:
            results.append({"dataset": name, "method": "?", "status": "error",
                            "error": "数据集不存在(workspace 下未找到该文件夹)"})
            print(f"[sweep] {name}: 数据集不存在,跳过", file=sys.stderr)
            if getattr(args, "stop_on_error", False):
                raise FileNotFoundError(f"未找到数据集 {name}")
            continue
        try:
            if method == "field-eval":
                r = run_field_eval_once(folder, args)
            elif method == "eval":
                r = run_eval_once(folder, args)
            else:
                raise ValueError(f"未知 eval.method={method!r}(应为 eval 或 field-eval)")
            results.append({**r, "status": "ok"})
        except Exception as e:  # noqa: BLE001 - 记录该数据集失败,继续下一个
            results.append({"dataset": name, "method": method, "status": "error",
                            "error": f"{type(e).__name__}: {e}"})
            print(f"[sweep] {name} 失败,已跳过继续下一个: {e}", file=sys.stderr)
            if getattr(args, "stop_on_error", False):
                raise

    summary = {
        "datasets": names,
        "results": results,
        "num_ok": sum(1 for r in results if r["status"] == "ok"),
        "num_error": sum(1 for r in results if r["status"] == "error"),
    }

    # 汇总目录:用第一个成功跑完的数据集的 模型/后端(同一模型下应一致)。
    summary_dir: Optional[Path] = None
    for r in results:
        if r["status"] == "ok":
            summary_dir = ws / "_sweep" / safe_model_dirname(str(r["model"])) / str(r["backend"])
            break
    if summary_dir is not None:
        results_store.write_json(summary_dir / "summary.json", summary)
        results_store.write_text(summary_dir / "summary.md", _render_sweep_summary(summary, ws))
        print(f"[sweep] 汇总 -> {summary_dir / 'summary.md'}")

    # 紧凑表格回显
    print(f"[sweep] 完成:成功 {summary['num_ok']},失败 {summary['num_error']}")
    for r in results:
        mark = "✓" if r["status"] == "ok" else "✗"
        if r["status"] == "error":
            print(f"  {mark} {r['dataset']} ({r['method']}): {r.get('error', '')}")
        else:
            print(f"  {mark} {r['dataset']} ({r['method']}): {_key_metric(r)}")
    return summary
