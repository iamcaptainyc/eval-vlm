"""自动化审核、数据集健康检查与 Run 对比。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from ..config import Config
from ..data.loader import _stable_id, load_raw_records, resolve_image_path
from ..results.store import discover_run_dirs, load_predictions
from .runsio import _find_run_dir


def check_dataset_health(cfg: Config) -> dict[str, Any]:
    """对数据集进行 P0 健康检查：缺图、<image> 占位符失配、重复 ID、数据格式。"""
    test_path = cfg.test_path
    if not test_path.exists():
        return {
            "dataset": cfg.dataset_dir.name,
            "total_samples": 0,
            "is_healthy": False,
            "error": "test.json 不存在",
        }

    records = load_raw_records(test_path)
    conv_key = cfg.data.mapping.messages
    img_key = cfg.data.mapping.images
    content_key = cfg.data.mapping.tags.content

    missing_images: list[dict[str, Any]] = []
    placeholder_mismatches: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    duplicate_ids: list[str] = []

    referenced_images: set[Path] = set()

    for i, rec in enumerate(records):
        sid = _stable_id(i, rec)
        if sid in seen_ids:
            duplicate_ids.append(sid)
        seen_ids.add(sid)

        images = list(rec.get(img_key, []))
        turns = rec.get(conv_key, [])

        # 检查占位符
        placeholder_count = sum(str(turn.get(content_key, "")).count("<image>") for turn in turns)
        if images and placeholder_count != len(images):
            placeholder_mismatches.append({
                "sample_id": sid,
                "position": i,
                "placeholder_count": placeholder_count,
                "images_count": len(images),
            })

        # 检查图片引用
        for img_ref in images:
            if img_ref.startswith(("http://", "https://", "data:")):
                continue
            try:
                local_p = resolve_image_path(img_ref, cfg).resolve()
                referenced_images.add(local_p)
                if not local_p.exists():
                    missing_images.append({
                        "sample_id": sid,
                        "position": i,
                        "ref": img_ref,
                        "resolved_path": str(local_p),
                    })
            except Exception as e:
                missing_images.append({
                    "sample_id": sid,
                    "position": i,
                    "ref": img_ref,
                    "error": str(e),
                })

    is_healthy = not (missing_images or placeholder_mismatches or duplicate_ids)

    return {
        "dataset": cfg.dataset_dir.name,
        "total_samples": len(records),
        "is_healthy": is_healthy,
        "missing_images_count": len(missing_images),
        "missing_images": missing_images[:100],  # 最多返回 100 条详细信息
        "placeholder_mismatches_count": len(placeholder_mismatches),
        "placeholder_mismatches": placeholder_mismatches,
        "duplicate_ids": duplicate_ids,
    }


def compare_runs(
    cfg: Config,
    run_a_spec: str,  # "model/backend"
    run_b_spec: str,
) -> dict[str, Any]:
    """对比两个 Run 的结果指标与预测一致性。"""
    try:
        m_a, b_a = run_a_spec.split("/", 1)
        m_b, b_b = run_b_spec.split("/", 1)
    except ValueError:
        return {"error": "Run 格式应为 'model/backend'"}

    dir_a = _find_run_dir(cfg, m_a, b_a)
    dir_b = _find_run_dir(cfg, m_b, b_b)

    preds_a = { (p.id, p.turn): p for p in load_predictions(dir_a / "predictions.jsonl") }
    preds_b = { (p.id, p.turn): p for p in load_predictions(dir_b / "predictions.jsonl") }

    common_keys = set(preds_a.keys()) & set(preds_b.keys())
    if not common_keys:
        return {
            "run_a": run_a_spec,
            "run_b": run_b_spec,
            "common_samples": 0,
            "agreement_rate": 0.0,
            "mismatches": [],
        }

    agreements = 0
    mismatches: list[dict[str, Any]] = []

    for k in sorted(common_keys):
        p_a = preds_a[k].prediction.strip()
        p_b = preds_b[k].prediction.strip()
        if p_a == p_b:
            agreements += 1
        else:
            if len(mismatches) < 50:
                mismatches.append({
                    "id": k[0],
                    "turn": k[1],
                    "pred_a": p_a,
                    "pred_b": p_b,
                })

    return {
        "run_a": run_a_spec,
        "run_b": run_b_spec,
        "common_samples": len(common_keys),
        "agreements": agreements,
        "agreement_rate": round(agreements / len(common_keys), 4) if common_keys else 0.0,
        "mismatches_count": len(common_keys) - agreements,
        "sample_mismatches": mismatches,
    }
