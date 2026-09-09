"""FastAPI 依赖注入。"""
from __future__ import annotations

from pathlib import Path
from fastapi import Depends, HTTPException, status

from ..config import Config, load_dataset_config
from ..workspace import resolve_dataset_dir
from .settings import Settings, get_settings


def get_dataset_cfg(name: str, settings: Settings = Depends(get_settings)) -> Config:
    """根据数据集名称解析并加载 Config 对象。"""
    try:
        ds_dir = resolve_dataset_dir(name, settings.workspace)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"未找到数据集 '{name}': {e}",
        )
    try:
        return load_dataset_config(ds_dir)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"加载数据集 '{name}' 配置失败: {e}",
        )
