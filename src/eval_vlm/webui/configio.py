"""数据集配置读写服务。"""
from __future__ import annotations

from typing import Any
import yaml

from ..config import Config, load_dataset_config
from ..workspace import describe_settable_keys, set_dataset_value
from .auth import write_audit
from .locks import dataset_lock
from .models import ConfigUpdateItem
from .settings import Settings


def read_config_info(cfg: Config) -> dict[str, Any]:
    """读取数据集配置的结构化信息、原文与可配置模板说明。"""
    config_path = cfg.dataset_dir / "config.yaml"
    raw_text = config_path.read_text(encoding="utf-8") if config_path.exists() else ""
    parsed_yaml = yaml.safe_load(raw_text) if raw_text else {}

    return {
        "dataset": cfg.dataset_dir.name,
        "config": parsed_yaml,
        "raw": raw_text,
        "settable_doc": describe_settable_keys(),
    }


async def update_dataset_config(
    cfg: Config,
    settings: Settings,
    updates: list[ConfigUpdateItem],
    user: str = "anonymous",
) -> dict[str, Any]:
    """批量更新配置项并保留注释。"""
    ds_name = cfg.dataset_dir.name
    async with dataset_lock(ds_name, settings):
        applied: list[dict[str, Any]] = []
        for item in updates:
            set_dataset_value(cfg.dataset_dir, item.key, item.value)
            applied.append({"key": item.key, "value": item.value})

        write_audit(
            settings,
            user,
            "update_config",
            {"dataset": ds_name, "updates": applied},
        )

        new_cfg = load_dataset_config(cfg.dataset_dir)
        return {
            "success": True,
            "dataset": ds_name,
            "applied": applied,
            "config": read_config_info(new_cfg),
        }
