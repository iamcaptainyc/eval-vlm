"""运行结果、指标与报告查看服务。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from fastapi import HTTPException, status
from fastapi.responses import FileResponse, HTMLResponse

from ..config import Config
from ..results.store import discover_run_dirs
from .locks import calc_file_sha256
from .models import RunSummary


def _find_run_dir(cfg: Config, model: str, backend: str) -> Path:
    target = cfg.dataset_dir / model / backend
    if not target.is_dir():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"未找到运行结果目录: {model}/{backend}",
        )
    return target


def list_runs(cfg: Config) -> list[RunSummary]:
    """枚举当前数据集下的所有模型/后端运行结果及过期状态。"""
    current_test_sha = calc_file_sha256(cfg.test_path)
    runs = discover_run_dirs(cfg.dataset_dir)
    summaries: list[RunSummary] = []

    for model, backend, rdir in runs:
        metrics_file = rdir / "metrics.json"
        scored_file = rdir / "scored.jsonl"
        failures_file = rdir / "failures.html"
        dirty_file = rdir / "dataset_dirty.json"
        meta_file = rdir / "run_meta.json"

        # field-eval 产物
        field_metrics_file = rdir / "field_metrics.json"
        field_mismatches_file = rdir / "field_mismatches.json"
        field_mismatches_html = rdir / "field_mismatches.html"

        is_stale = False
        stale_reason: Optional[str] = None

        # 检查是否显式标记了脏数据
        if dirty_file.exists():
            is_stale = True
            try:
                d_info = json.loads(dirty_file.read_text(encoding="utf-8"))
                stale_reason = d_info.get("reason", "数据集已被修改，评测结果已过期")
            except Exception:
                stale_reason = "数据集已被修改，评测结果已过期"
        elif meta_file.exists() and current_test_sha:
            try:
                m_info = json.loads(meta_file.read_text(encoding="utf-8"))
                run_sha = m_info.get("test_sha256")
                if run_sha and run_sha != current_test_sha:
                    is_stale = True
                    stale_reason = "当前 test.json 的 SHA 与运行记录不一致"
            except Exception:
                pass

        # 提取 eval 指标摘要
        metrics_summary: Optional[dict[str, Any]] = None
        if metrics_file.exists():
            try:
                m_data = json.loads(metrics_file.read_text(encoding="utf-8"))
                metrics_summary = {
                    "overall_mean_score": m_data.get("overall_mean_score"),
                    "num_samples": m_data.get("num_samples"),
                    "num_failed_targets": m_data.get("num_failed_targets"),
                    "model": m_data.get("model"),
                    "backend": m_data.get("backend"),
                    "scorer": m_data.get("scorer"),
                }
            except Exception:
                pass

        # 提取 field-eval 指标摘要
        field_metrics_summary: Optional[dict[str, Any]] = None
        if field_metrics_file.exists():
            try:
                fm_data = json.loads(field_metrics_file.read_text(encoding="utf-8"))
                ov = fm_data.get("overall", {}) if isinstance(fm_data.get("overall"), dict) else {}
                field_metrics_summary = {
                    "micro_accuracy": ov.get("micro_accuracy", fm_data.get("micro_accuracy")),
                    "macro_accuracy": ov.get("macro_accuracy", fm_data.get("macro_accuracy")),
                    "exact_match_rate": ov.get("exact_match_rate", fm_data.get("exact_match_rate", fm_data.get("exact_match_ratio"))),
                    "exact_match_samples": ov.get("exact_match_samples", fm_data.get("exact_match_samples")),
                    "num_samples": fm_data.get("num_samples", fm_data.get("total_samples")),
                    "num_scored": fm_data.get("num_scored", fm_data.get("evaluated_samples")),
                    "num_pred_missing": fm_data.get("num_pred_missing", 0),
                    "fields": fm_data.get("fields", []),
                }
            except Exception:
                pass

        has_eval = metrics_file.exists() or scored_file.exists()
        has_field_eval = field_metrics_file.exists() or field_mismatches_file.exists()

        summaries.append(
            RunSummary(
                model=model,
                backend=backend,
                path=str(rdir),
                has_eval=has_eval,
                has_field_eval=has_field_eval,
                has_metrics=metrics_file.exists(),
                has_scored=scored_file.exists(),
                has_failures_html=failures_file.exists(),
                has_field_metrics=field_metrics_file.exists(),
                has_field_mismatches_html=field_mismatches_html.exists(),
                has_field_mismatches_json=field_mismatches_file.exists(),
                is_stale=is_stale,
                stale_reason=stale_reason,
                metrics_summary=metrics_summary,
                field_metrics_summary=field_metrics_summary,
            )
        )

    return summaries


def get_metrics_detail(cfg: Config, model: str, backend: str) -> dict[str, Any]:
    """读取指定运行的完整 metrics.json。"""
    rdir = _find_run_dir(cfg, model, backend)
    metrics_file = rdir / "metrics.json"
    if not metrics_file.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"{model}/{backend} 缺少 metrics.json",
        )
    return json.loads(metrics_file.read_text(encoding="utf-8"))


def get_field_metrics_detail(cfg: Config, model: str, backend: str) -> dict[str, Any]:
    """读取指定运行的完整 field_metrics.json。"""
    rdir = _find_run_dir(cfg, model, backend)
    fm_file = rdir / "field_metrics.json"
    if not fm_file.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"{model}/{backend} 缺少 field_metrics.json",
        )
    return json.loads(fm_file.read_text(encoding="utf-8"))


def get_scored_records(
    cfg: Config,
    model: str,
    backend: str,
    offset: int = 0,
    limit: int = 50,
    min_score: Optional[float] = None,
    max_score: Optional[float] = None,
    order_by: str = "default",  # "default" | "lowest" | "highest"
) -> dict[str, Any]:
    """分页读取与筛选逐样本评分记录。"""
    rdir = _find_run_dir(cfg, model, backend)
    scored_file = rdir / "scored.jsonl"
    if not scored_file.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"{model}/{backend} 缺少 scored.jsonl",
        )

    rows: list[dict[str, Any]] = []
    with scored_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                score = obj.get("score")
                if min_score is not None and score is not None and score < min_score:
                    continue
                if max_score is not None and score is not None and score > max_score:
                    continue
                rows.append(obj)
            except Exception:
                continue

    if order_by == "lowest":
        rows.sort(key=lambda x: (x.get("score") is None, x.get("score", 0.0)))
    elif order_by == "highest":
        rows.sort(key=lambda x: x.get("score", -1.0), reverse=True)

    total = len(rows)
    paged = rows[offset : offset + limit]

    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "records": paged,
    }


def get_field_mismatches_records(
    cfg: Config,
    model: str,
    backend: str,
    offset: int = 0,
    limit: int = 50,
    filter_state: Optional[str] = None,
) -> dict[str, Any]:
    """分页读取与筛选 field-eval 的逐字段失配记录 (field_mismatches.json)。"""
    rdir = _find_run_dir(cfg, model, backend)
    fm_file = rdir / "field_mismatches.json"
    if not fm_file.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"{model}/{backend} 缺少 field_mismatches.json",
        )

    data = json.loads(fm_file.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = data if isinstance(data, list) else data.get("rows", [])
    if filter_state:
        rows = [r for r in rows if r.get("state") == filter_state]

    total = len(rows)
    paged = rows[offset : offset + limit]

    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "records": paged,
    }


def serve_failures_html(cfg: Config, model: str, backend: str) -> FileResponse:
    """返回生成的 failures.html 文件。"""
    rdir = _find_run_dir(cfg, model, backend)
    failures_file = rdir / "failures.html"
    if not failures_file.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="该运行未生成 failures.html (可能全部命中或未运行 evaluate)",
        )
    return FileResponse(path=str(failures_file), media_type="text/html")


def serve_field_mismatches_html(cfg: Config, model: str, backend: str) -> FileResponse:
    """返回生成的 field_mismatches.html 文件。"""
    rdir = _find_run_dir(cfg, model, backend)
    html_file = rdir / "field_mismatches.html"
    if not html_file.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="未找到 field_mismatches.html",
        )
    return FileResponse(path=str(html_file), media_type="text/html")
