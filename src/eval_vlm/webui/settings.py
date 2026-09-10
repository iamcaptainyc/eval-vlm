"""WebUI 配置与路径管理。"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from ..workspace import load_global_config, resolve_workspace


class Settings:
    def __init__(
        self,
        workspace_dir: Optional[Path | str] = None,
        host: str = "127.0.0.1",
        port: int = 8080,
    ) -> None:
        self.host = host
        self.port = port
        global_cfg = load_global_config()
        cli_override = str(workspace_dir) if workspace_dir else os.environ.get("EVAL_VLM_WORKSPACE")
        self.workspace: Path = resolve_workspace(cli_override, global_cfg)
        self.media_root: Optional[str] = global_cfg.get("media_root")
        self.image_strip_prefix: Optional[str] = global_cfg.get("image_strip_prefix")
        self.hf_models_dir: Optional[str] = global_cfg.get("hf_models_dir")
        self.mnn_models_dir: Optional[str] = global_cfg.get("mnn_models_dir")
        self.state_dir: Path = self.workspace / "_webui"
        self.locks_dir: Path = self.state_dir / "locks"
        self.jobs_dir: Path = self.state_dir / "jobs"
        self.trash_dir: Path = self.state_dir / "trash"
        self.users_file: Path = self.state_dir / "users.yaml"
        self.audit_file: Path = self.state_dir / "audit.log.jsonl"
        self.token: Optional[str] = os.environ.get("EVAL_VLM_WEBUI_TOKEN")

        self.ensure_dirs()

    def reload_global_config(self) -> None:
        global_cfg = load_global_config()
        self.media_root = global_cfg.get("media_root")
        self.image_strip_prefix = global_cfg.get("image_strip_prefix")
        self.hf_models_dir = global_cfg.get("hf_models_dir")
        self.mnn_models_dir = global_cfg.get("mnn_models_dir")

    def ensure_dirs(self) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.locks_dir.mkdir(parents=True, exist_ok=True)
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.trash_dir.mkdir(parents=True, exist_ok=True)


_SETTINGS: Optional[Settings] = None


def get_settings(workspace_dir: Optional[Path | str] = None) -> Settings:
    global _SETTINGS
    if _SETTINGS is None or (workspace_dir and Path(workspace_dir).resolve() != _SETTINGS.workspace):
        _SETTINGS = Settings(workspace_dir=workspace_dir)
    return _SETTINGS


def set_settings(settings: Settings) -> None:
    global _SETTINGS
    _SETTINGS = settings
