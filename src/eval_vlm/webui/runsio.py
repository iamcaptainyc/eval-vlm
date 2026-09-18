"""运行结果、指标与报告查看服务。"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Optional
import urllib.parse

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
        if not failures_file.exists() and (rdir / "failure.html").exists():
            failures_file = rdir / "failure.html"
        dirty_file = rdir / "dataset_dirty.json"
        meta_file = rdir / "run_meta.json"

        # field-eval 产物
        field_metrics_file = rdir / "field_metrics.json"
        field_mismatches_file = rdir / "field_mismatches.json"
        field_mismatches_html = rdir / "field_mismatches.html"
        if not field_mismatches_html.exists() and (rdir / "field_mismatch.html").exists():
            field_mismatches_html = rdir / "field_mismatch.html"

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


_SAMPLES_CACHE: dict[str, tuple[float, int, dict[str, Any]]] = {}


def _get_samples_map(cfg: Config) -> dict[str, Any]:
    """快速获取带 mtime 与 size 缓存的样本映射字典，避免每次请求重复解析 test.json。"""
    test_path = cfg.test_path
    if not test_path.exists():
        return {}
    try:
        stat = test_path.stat()
        cache_key = str(test_path.resolve())
        cached = _SAMPLES_CACHE.get(cache_key)
        if cached and cached[0] == stat.st_mtime and cached[1] == stat.st_size:
            return cached[2]

        from ..data.loader import load_samples

        samples = load_samples(cfg, source=test_path)
        samples_map = {s.id: s for s in samples}
        _SAMPLES_CACHE[cache_key] = (stat.st_mtime, stat.st_size, samples_map)
        return samples_map
    except Exception:
        return {}


def _resolve_sample_turn(obj: dict[str, Any], sample: Any = None) -> tuple[int, int]:
    """解析记录对应的 (实际助手消息轮次 actual_turn, 目标序号 ordinal)。

    在多模态对话规范中：
    - 偶数下标 (0, 2, 4...) 为用户提问 (User 问答)
    - 奇数下标 (1, 3, 5...) 为助手回复 (Assistant 模型预测目标轮次)
    """
    raw_turn = obj.get("turn")
    raw_ord = obj.get("ordinal")

    # 1. 若有关联样本且包含 targets，以 sample.targets (turn_index 均为奇数轮) 为权威基准
    if sample and hasattr(sample, "targets") and sample.targets:
        # A. 若记录包含显式 ordinal
        if raw_ord is not None:
            try:
                ord_i = int(raw_ord)
                if 0 <= ord_i < len(sample.targets):
                    return sample.targets[ord_i].turn_index, ord_i
            except (TypeError, ValueError):
                pass
        # B. 检查 raw_turn 是否直接命中某个 target.turn_index (1, 3, 5...)
        if raw_turn is not None:
            try:
                t_val = int(raw_turn)
                for i, tgt in enumerate(sample.targets):
                    if tgt.turn_index == t_val:
                        return t_val, i
                # 若 raw_turn 是偶数 (如 0, 2, 4... 用户误当成 0 起始的目标序号或用户轮次)
                # 例如 turn: 0 -> 目标 0 -> targets[0].turn_index (1)
                # turn: 2 -> 对应第 2 个用户问答后的助手回复 -> targets[1].turn_index (3)
                if t_val % 2 == 0:
                    guess_ord = t_val // 2
                    if 0 <= guess_ord < len(sample.targets):
                        return sample.targets[guess_ord].turn_index, guess_ord
                elif 0 <= t_val < len(sample.targets):
                    return sample.targets[t_val].turn_index, t_val
            except (TypeError, ValueError):
                pass
        # 默认回退到第 0 个目标 (通常为 turn 1)
        return sample.targets[0].turn_index, 0

    # 2. 若无 sample 对象的纯数据推算
    if raw_ord is not None:
        try:
            ord_i = max(0, int(raw_ord))
            return 2 * ord_i + 1, ord_i
        except (TypeError, ValueError):
            pass

    if raw_turn is not None:
        try:
            t_val = int(raw_turn)
            if t_val % 2 == 1:
                # 已经是奇数助手轮 (1, 3, 5...)
                return t_val, (t_val - 1) // 2
            else:
                # 偶数轮 (0, 2, 4...) 为用户提问，模型预测必然为对应的后续助手轮 (1, 3, 5...)
                return t_val + 1, t_val // 2
        except (TypeError, ValueError):
            pass

    return 1, 0


def _match_turn(filter_turn: int, actual_turn: int, ord_i: int, raw_turn: Any) -> bool:
    # 1. 优先匹配实际助手轮次 (奇数轮 1, 3, 5...)
    if filter_turn == actual_turn:
        return True
    # 2. 若 filter_turn 是偶数 (0, 2, 4...)，偶数是用户提问轮，对应其后续助手模型预测 (0->1, 2->3, 4->5)
    #    或者作为 0 起始的 ordinal 序号 (0 对应第 1 个目标轮 actual_turn 1)
    if filter_turn % 2 == 0:
        if filter_turn + 1 == actual_turn or filter_turn == ord_i:
            return True
    return False


def get_scored_records(
    cfg: Config,
    model: str,
    backend: str,
    offset: int = 0,
    limit: int = 50,
    min_score: Optional[float] = None,
    max_score: Optional[float] = None,
    order_by: str = "default",  # "default" | "lowest" | "highest"
    turn: Optional[int] = None,
    query: Optional[str] = None,
    only_miss: bool = False,
) -> dict[str, Any]:
    """分页读取与筛选逐样本评分记录，附加原图与上下文信息。"""
    rdir = _find_run_dir(cfg, model, backend)
    scored_file = rdir / "scored.jsonl"
    if not scored_file.exists():
        return {
            "total": 0,
            "offset": offset,
            "limit": limit,
            "records": [],
            "warning": f"{model}/{backend} 尚未生成 scored.jsonl 评测明细文件",
        }

    # 读取/复用 test.json 样本缓存以附带原图和前置多轮对话上下文
    samples_map = _get_samples_map(cfg)

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
                s_item = samples_map.get(obj.get("id"))
                actual_turn, ord_i = _resolve_sample_turn(obj, s_item)
                if turn is not None and not _match_turn(turn, actual_turn, ord_i, obj.get("turn")):
                    continue
                is_miss = (score is None or score < 1.0 or bool(obj.get("error")))
                if only_miss and not is_miss:
                    continue
                if query:
                    q = query.strip().lower()
                    if (q not in str(obj.get("id", "")).lower()
                        and q not in str(obj.get("prediction", "")).lower()
                        and q not in str(obj.get("reference", "")).lower()):
                        continue
                obj["is_miss"] = is_miss
                obj["turn"] = actual_turn
                obj["ordinal"] = ord_i
                rows.append(obj)
            except Exception:
                continue

    if order_by == "lowest":
        rows.sort(key=lambda x: (x.get("score") is None, x.get("score", 0.0)))
    elif order_by == "highest":
        rows.sort(key=lambda x: x.get("score", -1.0), reverse=True)

    total = len(rows)
    paged = rows[offset : offset + limit]

    ds_name = cfg.dataset_dir.name
    for row in paged:
        s = samples_map.get(row.get("id"))
        imgs = list(s.images) if (s and hasattr(s, "images")) else (row.get("images") or [])
        row["images"] = imgs
        row["image_urls"] = [
            f"/api/datasets/{ds_name}/image?ref={urllib.parse.quote(ref)}"
            for ref in imgs
        ]

        target_turn = row.get("turn", 1)

        if s and hasattr(s, "turns") and s.turns:
            turns_ctx = []
            # target_turn 是模型预测所在的 Assistant 轮次 (奇数轮 1, 3, 5...)
            # 前置对话是且仅是位于该预测之前的轮次 (即 idx < target_turn)
            for idx, t in enumerate(s.turns):
                if idx >= target_turn:
                    break
                role = getattr(t, "role", "user") if hasattr(t, "role") else (t.get("role") if isinstance(t, dict) else "user")
                content = getattr(t, "content", "") if hasattr(t, "content") else (t.get("content") if isinstance(t, dict) else str(t))
                turns_ctx.append({"role": role, "content": content, "turn_index": idx})
            row["turns"] = turns_ctx
        else:
            row["turns"] = []

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
    filter_field: Optional[str] = None,
    query: Optional[str] = None,
) -> dict[str, Any]:
    """分页读取与筛选 field-eval 的逐字段失配记录 (field_mismatches.json)，附加图片与字段过滤。"""
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

    if filter_field:
        ff = filter_field.strip()
        rows = [
            r for r in rows
            if (
                str(r.get("field", "")).strip() == ff
                or any(
                    str(f.get("field", "")).strip() == ff
                    and (not f.get("correct") or (f.get("is_empty_ref") is False and not f.get("correct")))
                    for f in r.get("fields", [])
                )
            )
        ]

    if query:
        q = query.strip().lower()
        rows = [
            r for r in rows
            if (
                q in str(r.get("id", "")).lower()
                or q in str(r.get("pred_desc", "")).lower()
                or q in str(r.get("field", "")).lower()
                or q in str(r.get("actual", "")).lower()
                or q in str(r.get("expected", "")).lower()
            )
        ]

    total = len(rows)
    paged = rows[offset : offset + limit]

    ds_name = cfg.dataset_dir.name
    for row in paged:
        if "fields" not in row and "field" in row:
            row["fields"] = [{
                "field": row.get("field"),
                "ref": row.get("expected"),
                "pred": row.get("actual"),
                "correct": False,
                "is_empty_ref": False,
            }]
        if "state" not in row:
            row["state"] = "mismatch"
        row["image_urls"] = [
            f"/api/datasets/{ds_name}/image?ref={urllib.parse.quote(ref)}"
            for ref in row.get("images", [])
        ]

    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "records": paged,
    }


def serve_failures_html(cfg: Config, model: str, backend: str) -> FileResponse:
    """返回生成的 failures.html 文件 (兼容 failure.html 命名)。"""
    rdir = _find_run_dir(cfg, model, backend)
    failures_file = rdir / "failures.html"
    if not failures_file.exists():
        if (rdir / "failure.html").exists():
            failures_file = rdir / "failure.html"
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"{model}/{backend} 未生成 failures.html (或 failure.html)，可能全部命中或尚未运行 eval",
            )
    return FileResponse(path=str(failures_file), media_type="text/html")


def serve_field_mismatches_html(cfg: Config, model: str, backend: str) -> FileResponse:
    """返回生成的 field_mismatches.html 文件 (兼容 field_mismatch.html 命名)。"""
    rdir = _find_run_dir(cfg, model, backend)
    html_file = rdir / "field_mismatches.html"
    if not html_file.exists():
        if (rdir / "field_mismatch.html").exists():
            html_file = rdir / "field_mismatch.html"
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"{model}/{backend} 缺少 field_mismatches.html",
            )
    return FileResponse(path=str(html_file), media_type="text/html")


def list_dataset_html_files(cfg: Config) -> list[dict[str, Any]]:
    """递归检索当前数据集目录下的全部 HTML 报告/可视化文件 (含 failures.html, field_mismatches.html 等)。"""
    if not cfg.dataset_dir.exists():
        return []
    items: list[dict[str, Any]] = []
    for p in sorted(cfg.dataset_dir.rglob("*.html")):
        if not p.is_file():
            continue
        rel = p.relative_to(cfg.dataset_dir).as_posix()
        # 排除隐藏或状态目录
        if any(part.startswith(".") or part.startswith("_") for part in p.parts):
            continue
        st = p.stat()
        items.append({
            "name": p.name,
            "path": rel,
            "size": st.st_size,
            "modified_at": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat(),
            "url": f"/api/datasets/{urllib.parse.quote(cfg.dataset_dir.name)}/html-view?path={urllib.parse.quote(rel)}",
        })
    return items


def serve_dataset_html(cfg: Config, rel_path: str) -> FileResponse:
    """安全读取并返回数据集目录内的 HTML 文件，严防路径穿越并支持友好回退查找。"""
    clean_path = rel_path.strip().replace("\\", "/").lstrip("/")
    target = (cfg.dataset_dir / clean_path).resolve()
    ds_root = cfg.dataset_dir.resolve()
    try:
        target.relative_to(ds_root)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="非法路径访问：禁止访问数据集目录之外的文件",
        )

    if not target.exists() or not target.is_file():
        # 容错查找：如传入 failure.html，自动在数据集内寻找 failures.html 等
        fname = Path(clean_path).name.lower()
        candidates: list[Path] = []
        if fname in ("failure.html", "failures.html"):
            candidates = list(cfg.dataset_dir.rglob("*failure*.html"))
        elif "mismatch" in fname:
            candidates = list(cfg.dataset_dir.rglob("*mismatch*.html"))
        else:
            candidates = list(cfg.dataset_dir.rglob(Path(clean_path).name))

        valid_candidates = []
        for candidate in candidates:
            resolved_candidate = candidate.resolve()
            try:
                resolved_candidate.relative_to(ds_root)
            except ValueError:
                continue
            if resolved_candidate.is_file() and not resolved_candidate.name.startswith("."):
                valid_candidates.append(resolved_candidate)
        if valid_candidates:
            target = valid_candidates[0].resolve()
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"在数据集 {cfg.dataset_dir.name} 中未找到 HTML 文件: {clean_path}",
            )

    return FileResponse(path=str(target), media_type="text/html")
