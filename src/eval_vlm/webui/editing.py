"""样本删除、恢复与下游同步机制。"""
from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import HTTPException, status

from ..config import Config, load_dataset_config
from ..data.loader import _stable_id, load_raw_records
from ..data.splitter import load_split_meta
from ..results.store import discover_run_dirs, write_json
from .auth import write_audit
from .locks import calc_file_sha256, dataset_lock, verify_test_sha
from .models import DeleteSampleResponse, RestoreResponse
from .settings import Settings


def _locate_sample(records: list[dict[str, Any]], sample_id: str) -> int:
    """在 records 列表中精确定位 sample_id 对应的位置下标。"""
    parts = sample_id.split("-")
    if len(parts) >= 2 and parts[0].isdigit():
        pos = int(parts[0])
        if 0 <= pos < len(records):
            if _stable_id(pos, records[pos]) == sample_id:
                return pos

    # 若根据下标未直接命中，全量扫描以防此前被移动
    for i, rec in enumerate(records):
        if _stable_id(i, rec) == sample_id:
            return i
    return -1


def delete_sample(
    cfg: Config,
    settings: Settings,
    sample_id: str,
    expected_sha256: str,
    user: str = "anonymous",
    reason: Optional[str] = None,
    mode: str = "record",
    image_index: Optional[int] = None,
) -> DeleteSampleResponse:
    """删除样本（整条或单图），更新 test.json、split_meta.json 并标记受影响的 run。"""
    test_path = cfg.test_path
    if not test_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"测试集文件不存在: {test_path}",
        )

    # 1. 校验乐观锁
    test_sha_before = verify_test_sha(test_path, expected_sha256)

    # 2. 读取 raw records 并定位
    records = load_raw_records(test_path)
    pos = _locate_sample(records, sample_id)
    if pos < 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"在 test.json 中未找到样本: {sample_id}",
        )

    target_record = records[pos]
    original_record_snapshot = json.loads(json.dumps(target_record))
    now = datetime.now(timezone.utc)
    ts_str = now.strftime("%Y%m%d_%H%M%S")
    trash_id = f"{ts_str}-{sample_id}"
    trash_dir = settings.trash_dir / cfg.dataset_dir.name / trash_id
    trash_dir.mkdir(parents=True, exist_ok=True)

    # 读取 split_meta 以获取原始源索引
    split_meta = load_split_meta(cfg)
    orig_idx: Optional[int] = None
    if split_meta and "indices" in split_meta and "test" in split_meta["indices"]:
        test_indices: list[int] = split_meta["indices"]["test"]
        if 0 <= pos < len(test_indices):
            orig_idx = test_indices[pos]

    # 3. 根据模式修改 records
    actual_mode = mode
    if mode == "image":
        img_key = cfg.data.mapping.images
        conv_key = cfg.data.mapping.messages
        content_key = cfg.data.mapping.tags.content
        images: list[str] = list(target_record.get(img_key, []))

        if len(images) <= 1 or image_index is None or image_index < 0 or image_index >= len(images):
            # 只有一张图或下标非法，退化为整条删除
            actual_mode = "record"
            records.pop(pos)
        else:
            # 移除指定图
            del images[image_index]
            target_record[img_key] = images

            # 移除对话中对应的第 image_index 个 <image> 占位符
            turns = target_record.get(conv_key, [])
            placeholder_seen = 0
            removed = False
            for turn in turns:
                content = turn.get(content_key, "")
                count = content.count("<image>")
                if not removed and placeholder_seen + count > image_index:
                    target_nth_in_turn = image_index - placeholder_seen
                    # 替换第 target_nth_in_turn 个 <image>
                    parts = content.split("<image>")
                    new_content = (
                        "<image>".join(parts[: target_nth_in_turn + 1])
                        + "".join(parts[target_nth_in_turn + 1 :])
                        if target_nth_in_turn + 1 < len(parts)
                        else content
                    )
                    # 简化替换：按序只删一个
                    # 通过找到目标下标处的 <image>
                    find_pos = 0
                    for _ in range(target_nth_in_turn):
                        find_pos = content.find("<image>", find_pos) + len("<image>")
                    del_pos = content.find("<image>", find_pos)
                    if del_pos != -1:
                        new_content = content[:del_pos] + content[del_pos + len("<image>") :]
                        turn[content_key] = new_content
                        removed = True
                placeholder_seen += count

            records[pos] = target_record
    else:
        actual_mode = "record"
        records.pop(pos)

    # 4. 备份原 test.json 与记录
    shutil.copy2(test_path, trash_dir / "test.json.bak")
    (trash_dir / "record.json").write_text(
        json.dumps(original_record_snapshot, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest = {
        "trash_id": trash_id,
        "sample_id": sample_id,
        "position": pos,
        "original_source_index": orig_idx,
        "deleted_by": user,
        "reason": reason,
        "ts": now.isoformat(),
        "test_sha_before": test_sha_before,
        "mode": actual_mode,
        "image_index": image_index if actual_mode == "image" else None,
        "restored": False,
    }
    (trash_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 5. 原子写回 test.json
    tmp_file = test_path.with_suffix(".json.tmp")
    with tmp_file.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    tmp_file.replace(test_path)
    new_sha256 = calc_file_sha256(test_path)

    # 6. 同步 split_meta.json
    if split_meta and "indices" in split_meta and "test" in split_meta["indices"]:
        if actual_mode == "record" and 0 <= pos < len(split_meta["indices"]["test"]):
            del split_meta["indices"]["test"][pos]
            split_meta["counts"]["test"] = len(split_meta["indices"]["test"])

        edits = split_meta.setdefault("webui_edits", [])
        edits.append({
            "ts": now.isoformat(),
            "deleted_ids": [sample_id],
            "deleted_original_indices": [orig_idx] if orig_idx is not None else [],
            "backup_path": str(trash_dir),
            "user": user,
            "mode": actual_mode,
        })
        split_meta["test_modified_after_split"] = True
        write_json(cfg.split_meta_path, split_meta)

    # 7. 标记下游 run 目录为 stale/dirty
    invalidated: list[str] = []
    run_dirs = discover_run_dirs(cfg.dataset_dir)
    for model_name, backend_name, rdir in run_dirs:
        dirty_file = rdir / "dataset_dirty.json"
        dirty_info = {
            "stale_since": now.isoformat(),
            "reason": f"样本 {sample_id} 被删除 (模式: {actual_mode})，原因: {reason or '无'}",
            "test_sha_before": test_sha_before,
            "test_sha_after": new_sha256,
            "deleted_ids": [sample_id],
            "by": user,
        }
        write_json(dirty_file, dirty_info)
        invalidated.append(f"{model_name}/{backend_name}")

    # 8. 审计记录
    write_audit(
        settings,
        user,
        f"delete_sample_{actual_mode}",
        {
            "dataset": cfg.dataset_dir.name,
            "sample_id": sample_id,
            "position": pos,
            "new_sha256": new_sha256,
            "invalidated_runs": invalidated,
        },
    )

    shifted_count = (len(records) - pos) if actual_mode == "record" else 0
    return DeleteSampleResponse(
        success=True,
        new_sha256=new_sha256,
        deleted_id=sample_id,
        mode=actual_mode,
        shifted_ids_count=shifted_count,
        invalidated_runs=invalidated,
    )


def list_trash(cfg: Config, settings: Settings) -> list[dict[str, Any]]:
    """列出当前数据集回收站中的样本。"""
    ds_trash = settings.trash_dir / cfg.dataset_dir.name
    if not ds_trash.exists():
        return []
    items: list[dict[str, Any]] = []
    for item_dir in sorted(ds_trash.iterdir(), reverse=True):
        if not item_dir.is_dir():
            continue
        manifest_file = item_dir / "manifest.json"
        if manifest_file.exists():
            try:
                manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
                if not manifest.get("restored", False):
                    record_file = item_dir / "record.json"
                    if record_file.exists():
                        manifest["record"] = json.loads(record_file.read_text(encoding="utf-8"))
                    items.append(manifest)
            except Exception:
                continue
    return items


def restore_sample(
    cfg: Config,
    settings: Settings,
    trash_id: str,
    expected_sha256: str,
    user: str = "anonymous",
) -> RestoreResponse:
    """从回收站恢复已删除的样本。"""
    trash_dir = settings.trash_dir / cfg.dataset_dir.name / trash_id
    manifest_file = trash_dir / "manifest.json"
    record_file = trash_dir / "record.json"

    if not manifest_file.exists() or not record_file.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"回收站记录不存在: {trash_id}",
        )

    test_path = cfg.test_path
    test_sha_before = verify_test_sha(test_path, expected_sha256)

    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    record = json.loads(record_file.read_text(encoding="utf-8"))
    if manifest.get("restored", False):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="该记录已被恢复过",
        )

    records = load_raw_records(test_path)
    target_pos = manifest.get("position", len(records))
    mode = manifest.get("mode", "record")

    if mode == "image":
        # 单图恢复：替换目标位置记录
        if 0 <= target_pos < len(records):
            records[target_pos] = record
        else:
            records.append(record)
            target_pos = len(records) - 1
    else:
        # 整条恢复：插回原位置或末尾
        if target_pos < 0 or target_pos > len(records):
            target_pos = len(records)
        records.insert(target_pos, record)

    # 原子写回 test.json
    tmp_file = test_path.with_suffix(".json.tmp")
    with tmp_file.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    tmp_file.replace(test_path)
    new_sha256 = calc_file_sha256(test_path)

    # 同步 split_meta
    split_meta = load_split_meta(cfg)
    orig_idx = manifest.get("original_source_index")
    now = datetime.now(timezone.utc)
    if split_meta and "indices" in split_meta and "test" in split_meta["indices"] and mode == "record":
        if orig_idx is not None:
            split_meta["indices"]["test"].insert(target_pos, orig_idx)
            split_meta["counts"]["test"] = len(split_meta["indices"]["test"])
            edits = split_meta.setdefault("webui_edits", [])
            edits.append({
                "ts": now.isoformat(),
                "action": "restore",
                "trash_id": trash_id,
                "restored_id": manifest["sample_id"],
                "user": user,
            })
            write_json(cfg.split_meta_path, split_meta)

    # 标记 downstream runs 为 stale
    invalidated: list[str] = []
    for model_name, backend_name, rdir in discover_run_dirs(cfg.dataset_dir):
        dirty_file = rdir / "dataset_dirty.json"
        dirty_info = {
            "stale_since": now.isoformat(),
            "reason": f"样本 {manifest['sample_id']} 已恢复，旧结果与新结构可能存在位移",
            "test_sha_before": test_sha_before,
            "test_sha_after": new_sha256,
            "restored_id": manifest["sample_id"],
            "by": user,
        }
        write_json(dirty_file, dirty_info)
        invalidated.append(f"{model_name}/{backend_name}")

    # 更新 manifest 为已恢复
    manifest["restored"] = True
    manifest["restored_at"] = now.isoformat()
    manifest["restored_by"] = user
    manifest_file.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    # 审计
    write_audit(
        settings,
        user,
        "restore_sample",
        {
            "dataset": cfg.dataset_dir.name,
            "trash_id": trash_id,
            "sample_id": manifest["sample_id"],
            "new_sha256": new_sha256,
        },
    )

    return RestoreResponse(
        success=True,
        new_sha256=new_sha256,
        restored_id=manifest["sample_id"],
        restored_position=target_pos,
        invalidated_runs=invalidated,
    )
