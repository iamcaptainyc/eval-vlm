"""数据集浏览、样本检索与图片流服务。"""
from __future__ import annotations

import io
import json
import urllib.parse
from pathlib import Path
from typing import Any, Optional

from fastapi import HTTPException, Response, status
from fastapi.responses import FileResponse, RedirectResponse
from PIL import Image

from ..config import Config, load_dataset_config
from ..data.loader import _parse_record, _stable_id, load_raw_records, resolve_image_path
from ..data.splitter import load_split_meta
from ..results.store import discover_run_dirs
from .locks import calc_file_sha256
from .models import DatasetSummary, ImageRefInfo, SampleItem, SamplesResponse
from .settings import Settings


def list_datasets(settings: Settings) -> list[DatasetSummary]:
    """列出当前工作区下的所有已配置数据集。"""
    ws = settings.workspace
    if not ws.exists() or not ws.is_dir():
        return []

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
            test_count = 0
            test_sha = ""
            if test_path.exists():
                test_sha = calc_file_sha256(test_path)
                try:
                    with test_path.open("r", encoding="utf-8") as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            test_count = len(data)
                except Exception:
                    pass

            runs = discover_run_dirs(p)
            has_dirty = any((rdir / "dataset_dirty.json").exists() for _, _, rdir in runs)

            split_meta = load_split_meta(cfg)
            counts = split_meta.get("counts", {}) if split_meta else {}

            summaries.append(
                DatasetSummary(
                    name=p.name,
                    path=str(p),
                    test_count=test_count or counts.get("test", 0),
                    train_count=counts.get("train", 0),
                    val_count=counts.get("val", 0),
                    run_count=len(runs),
                    test_sha256=test_sha,
                    media_root=str(cfg.media_root_path) if cfg.media_root_path else None,
                    source=str(cfg.source_path) if cfg.source_path else None,
                    has_dirty_runs=has_dirty,
                )
            )
        except Exception:
            continue

    return summaries


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

    test_sha = calc_file_sha256(test_path)
    records = load_raw_records(test_path)
    total_records = len(records)
    m = cfg.data.mapping

    ds_name = cfg.dataset_dir.name
    sample_items: list[SampleItem] = []

    # 遍历筛选
    for i, rec in enumerate(records):
        sid = _stable_id(i, rec)

        # 检查文本过滤
        if filter_text:
            text_haystack = json.dumps(rec, ensure_ascii=False)
            if filter_text.lower() not in text_haystack.lower() and filter_text.lower() not in sid.lower():
                continue

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

    filtered_total = len(sample_items)
    paged_samples = sample_items[offset : offset + limit]

    return SamplesResponse(
        total=filtered_total,
        offset=offset,
        limit=limit,
        test_sha256=test_sha,
        samples=paged_samples,
    )


def serve_image(
    cfg: Config,
    ref: str,
    thumb: bool = False,
) -> Response:
    """提供图片服务，带目录穿越校验与 Pillow 动态缩略图支持。"""
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

    # 4. 缩略图模式
    if thumb:
        try:
            with Image.open(resolved) as img:
                img = img.copy()
                img.thumbnail((768, 768), Image.Resampling.LANCZOS)
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
                img.save(buf, format="JPEG", quality=82, optimize=True)
                return Response(content=buf.getvalue(), media_type="image/jpeg")
        except Exception:
            # 缩略失败回退到原图输出
            pass

    return FileResponse(path=str(resolved))
