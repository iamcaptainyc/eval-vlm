"""HTML 报告的图片资源处理(缩放 + base64 内嵌),供 field-eval 等报告复用。

与 inference/openai_backend._image_to_data_url 不同:这里把本地图片先缩放(长边<=MAX_SIDE)
再转 JPEG 内嵌,避免把未压缩原图塞进 HTML 导致单文件过大;且任何一张图片解析/打开/编码失败
都返回 (None, 错误说明) 而不是抛异常,由调用方渲染占位符,单张坏图不中断整份报告。
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from .config import Config
from .data.loader import resolve_image_path

MAX_SIDE = 960
JPEG_QUALITY = 85


def image_ref_to_html_src(img: str, cfg: Config) -> tuple[Optional[str], Optional[str]]:
    """把一个图片引用变成可直接放进 <img src="..."> 的值,返回 (src, error)。

    - http(s):// 或 data: URL 原样透传(不下载、不重新编码),error=None;
    - 本地路径:先 resolve_image_path 定位(剥前缀/相对 media_root),存在则用 PIL
      缩放(长边<=MAX_SIDE)+ 转 JPEG(quality=JPEG_QUALITY)+ base64 -> data URL;
    - 路径不存在 / PIL 打开或编码失败 -> (None, 错误说明),不抛异常。
    """
    if img.startswith(("http://", "https://", "data:")):
        return img, None
    try:
        p = resolve_image_path(img, cfg)
    except Exception as e:  # noqa: BLE001 - 路径解析失败也走占位符,不打断报告
        return None, f"路径解析失败: {e}"
    if not p.exists():
        return None, f"图片不存在: {p}"
    try:
        from PIL import Image
        import base64
        import io

        with Image.open(p) as im:
            im = im.convert("RGB")
            w, h = im.size
            if max(w, h) > MAX_SIDE:
                scale = MAX_SIDE / max(w, h)
                im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BICUBIC)
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=JPEG_QUALITY)
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{b64}", None
    except Exception as e:  # noqa: BLE001 - 单张图片失败不应打断整份报告
        return None, f"图片读取/编码失败: {type(e).__name__}: {e}"
