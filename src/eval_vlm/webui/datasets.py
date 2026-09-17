"""数据集浏览、样本检索与图片流服务。"""
from __future__ import annotations

import io
import json
import urllib.parse
from collections import OrderedDict
from pathlib import Path
from threading import RLock
from typing import Any, Optional

from fastapi import HTTPException, Response, status
from fastapi.responses import FileResponse, RedirectResponse
from PIL import Image

from ..config import Config, load_dataset_config
from ..data.loader import _parse_record, _stable_id, resolve_image_path
from ..data.splitter import load_split_meta
from ..results.store import discover_run_dirs
from .models import DatasetSummary, ImageRefInfo, SampleItem, SamplesResponse
from .settings import Settings


_CACHE_LOCK = RLock()
_RAW_RECORDS_CACHE: OrderedDict[tuple[str, int, int], tuple[list[dict[str, Any]], str]] = OrderedDict()
_THUMB_CACHE: OrderedDict[tuple[str, int, int, int], bytes] = OrderedDict()
_DATASET_LIST_CACHE: OrderedDict[tuple[str, tuple[tuple[str, int, int], ...]], list[DatasetSummary]] = OrderedDict()
_RAW_RECORDS_CACHE_MAX = 4
_THUMB_CACHE_MAX = 256
_DATASET_LIST_CACHE_MAX = 4


def _stat_key(path: Path) -> tuple[str, int, int]:
    stat = path.stat()
    return (str(path.resolve()), stat.st_mtime_ns, stat.st_size)


def _bounded_put(cache: OrderedDict, key: Any, value: Any, max_entries: int) -> None:
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > max_entries:
        cache.popitem(last=False)


def _records_and_sha(test_path: Path) -> tuple[list[dict[str, Any]], str]:
    """Read a stable test file once per stat signature, including its SHA."""
    key = _stat_key(test_path)
    with _CACHE_LOCK:
        cached = _RAW_RECORDS_CACHE.get(key)
        if cached is not None:
            _RAW_RECORDS_CACHE.move_to_end(key)
            return cached
    # Keep this read isolated from editing: edit paths use load_raw_records.
    raw = test_path.read_bytes()
    parsed = json.loads(raw.decode("utf-8"))
    records = parsed if isinstance(parsed, list) else []
    import hashlib
    value = (records, hashlib.sha256(raw).hexdigest())
    with _CACHE_LOCK:
        _bounded_put(_RAW_RECORDS_CACHE, key, value, _RAW_RECORDS_CACHE_MAX)
    return value


def _dataset_signature(workspace: Path) -> tuple[tuple[str, int, int], ...]:
    entries: list[tuple[str, int, int]] = []
    for p in workspace.iterdir():
        if not p.is_dir() or p.name.startswith(("_", ".")):
            continue
        for name in ("config.yaml", "test.json", "split_meta.json"):
            candidate = p / name
            if candidate.exists():
                entries.append(_stat_key(candidate))
        # Run dirtiness/count changes must invalidate summaries too.
        for dirty in p.glob("*/*/dataset_dirty.json"):
            entries.append(_stat_key(dirty))
        # A completed run can appear without a dirty marker. Track the same
        # marker files used by discover_run_dirs so run_count never stays stale.
        for marker in ("metrics.json", "field_metrics.json", "precision.json", "run_meta.json", "pred_meta.json"):
            for run_marker in p.glob(f"*/*/{marker}"):
                entries.append(_stat_key(run_marker))
    return tuple(sorted(entries))


def list_datasets(settings: Settings) -> list[DatasetSummary]:
    """列出当前工作区下的所有已配置数据集。"""
    ws = settings.workspace
    if not ws.exists() or not ws.is_dir():
        return []

    signature = _dataset_signature(ws)
    cache_key = (str(ws.resolve()), signature)
    with _CACHE_LOCK:
        cached = _DATASET_LIST_CACHE.get(cache_key)
        if cached is not None:
            _DATASET_LIST_CACHE.move_to_end(cache_key)
            return [item.model_copy(deep=True) for item in cached]

    summaries: list[DatasetSummary] = []
    for p in sorted(ws.iterdir()):
        if not p.is_dir() or p.name.startswith(("_", ".")):
            continue
        cfg_file = p / "config.yaml"
        if not cfg_file.exists():
            continue

        try:
            cfg = load_dataset_config(p)
            test_path = cfg.test_path
            split_meta = load_split_meta(cfg)
            counts = split_meta.get("counts", {}) if split_meta else {}
            # Split metadata is the normal fast path. Avoid hashing/loading a
            # potentially large test.json just to draw a dataset card.
            test_count = int(counts.get("test", 0) or 0)
            if not test_count and test_path.exists():
                try:
                    test_count = len(_records_and_sha(test_path)[0])
                except Exception:
                    pass
            runs = discover_run_dirs(p)
            has_dirty = any((rdir / "dataset_dirty.json").exists() for _, _, rdir in runs)

            summaries.append(
                DatasetSummary(
                    name=p.name,
                    path=str(p),
                    test_count=test_count,
                    train_count=counts.get("train", 0),
                    val_count=counts.get("val", 0),
                    run_count=len(runs),
                    test_sha256=None,
                    media_root=str(cfg.media_root_path) if cfg.media_root_path else None,
                    source=str(cfg.source_path) if cfg.source_path else None,
                    has_dirty_runs=has_dirty,
                )
            )
        except Exception:
            continue

    with _CACHE_LOCK:
        _bounded_put(_DATASET_LIST_CACHE, cache_key, summaries, _DATASET_LIST_CACHE_MAX)
    return [item.model_copy(deep=True) for item in summaries]


def get_samples_page(
    cfg: Config,
    offset: int = 0,
    limit: int = 50,
    filter_text: Optional[str] = None,
) -> SamplesResponse:
    """分页获取测试样本详情。"""
    test_path = cfg.test_path
    if not test_path.exists():
        return SamplesResponse(
            total=0,
            offset=offset,
            limit=limit,
            test_sha256="",
            samples=[],
        )

    try:
        records, test_sha = _records_and_sha(test_path)
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"无法读取测试集: {exc}")
    m = cfg.data.mapping

    ds_name = cfg.dataset_dir.name
    if filter_text:
        needle = filter_text.lower()
        matching_positions = [
            i for i, rec in enumerate(records)
            if needle in json.dumps(rec, ensure_ascii=False).lower()
            or needle in _stable_id(i, rec).lower()
        ]
        filtered_total = len(matching_positions)
        page_positions = matching_positions[offset : offset + limit]
    else:
        filtered_total = len(records)
        page_positions = range(offset, min(offset + limit, filtered_total))
    sample_items: list[SampleItem] = []

    # Parse records only after pagination; large data sets no longer build a
    # full page model and inspect all image paths for every request.
    for i in page_positions:
        rec = records[i]
        sid = _stable_id(i, rec)

        parsed_sample = _parse_record(i, rec, m, cfg.eval.targets)

        # 解析图片信息
        image_infos: list[ImageRefInfo] = []
        for img_ref in parsed_sample.images:
            is_http = img_ref.startswith(("http://", "https://", "data:"))
            exists = True
            if not is_http:
                try:
                    local_p = resolve_image_path(img_ref, cfg)
                    exists = local_p.exists()
                except Exception:
                    exists = False

            q_ref = urllib.parse.quote(img_ref)
            url = f"/api/datasets/{ds_name}/image?ref={q_ref}"
            image_infos.append(
                ImageRefInfo(
                    ref=img_ref,
                    exists=exists,
                    is_http=is_http,
                    url=url,
                )
            )

        turns_data = [{"role": t.role, "content": t.content} for t in parsed_sample.turns]
        targets_data = [{"turn_index": t.turn_index, "reference": t.reference} for t in parsed_sample.targets]

        sample_items.append(
            SampleItem(
                id=sid,
                position=i,
                turns=turns_data,
                images=image_infos,
                targets=targets_data,
                meta=parsed_sample.meta,
            )
        )

    return SamplesResponse(
        total=filtered_total,
        offset=offset,
        limit=limit,
        test_sha256=test_sha,
        samples=sample_items,
    )


def serve_image(
    cfg: Config,
    ref: str,
    thumb: bool = False,
    max_dim: Optional[int] = None,
) -> Response:
    """提供图片服务，带目录穿越校验与 Pillow 动态缩略/大图适配支持。"""
    # 1. 外部链接直接 302
    if ref.startswith(("http://", "https://")):
        return RedirectResponse(url=ref, status_code=status.HTTP_302_FOUND)

    # 2. 本地路径解析
    try:
        resolved = resolve_image_path(ref, cfg).resolve()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"无效的图片路径: {e}",
        )

    # 3. 目录穿越安全校验
    allowed_roots: list[Path] = []
    if cfg.media_root_path:
        try:
            allowed_roots.append(cfg.media_root_path.resolve())
        except Exception:
            pass
    try:
        allowed_roots.append(cfg.dataset_dir.resolve())
    except Exception:
        pass

    is_safe = False
    for root in allowed_roots:
        try:
            resolved.relative_to(root)
            is_safe = True
            break
        except ValueError:
            continue

    # 宽容处理：允许处于 workspace 根目录下的文件
    if not is_safe and cfg.dataset_dir.parent:
        try:
            resolved.relative_to(cfg.dataset_dir.parent.resolve())
            is_safe = True
        except ValueError:
            pass

    if not is_safe:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="非法路径：禁止访问数据集与 media_root 范围之外的文件",
        )

    if not resolved.exists() or not resolved.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"图片文件不存在: {resolved}",
        )

    # 4. 缩略图模式或自适应尺寸模式（如 1K/1280px 大图适配）
    effective_dim = 768 if thumb else max_dim
    if effective_dim:
        try:
            cache_key = (*_stat_key(resolved), effective_dim)
            with _CACHE_LOCK:
                cached_thumb = _THUMB_CACHE.get(cache_key)
                if cached_thumb is not None:
                    _THUMB_CACHE.move_to_end(cache_key)
                    return Response(
                        content=cached_thumb,
                        media_type="image/jpeg",
                        headers={"Cache-Control": "public, max-age=86400", "ETag": f'"{hash(cache_key)}"'},
                    )
        except OSError:
            cache_key = None
        try:
            with Image.open(resolved) as img:
                w, h = img.size
                if max(w, h) > effective_dim:
                    img = img.copy()
                    img.thumbnail((effective_dim, effective_dim), Image.Resampling.LANCZOS)
                else:
                    img = img.copy()
                if img.mode in ("RGBA", "P", "LA"):
                    bg = Image.new("RGB", img.size, (255, 255, 255))
                    if img.mode == "RGBA":
                        bg.paste(img, mask=img.split()[3])
                    else:
                        bg.paste(img.convert("RGBA"))
                    img = bg
                elif img.mode != "RGB":
                    img = img.convert("RGB")

                buf = io.BytesIO()
                quality = 82 if thumb else 86
                img.save(buf, format="JPEG", quality=quality, optimize=True)
                payload = buf.getvalue()
                if cache_key is not None:
                    with _CACHE_LOCK:
                        _bounded_put(_THUMB_CACHE, cache_key, payload, _THUMB_CACHE_MAX)
                return Response(
                    content=payload,
                    media_type="image/jpeg",
                    headers={"Cache-Control": "public, max-age=86400", "ETag": f'"{hash(cache_key)}"'},
                )
        except Exception:
            # 缩略失败回退到原图输出
            pass

    return FileResponse(
        path=str(resolved),
        headers={"Cache-Control": "public, max-age=86400"},
    )
