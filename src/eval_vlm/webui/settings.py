"""WebUI 配置与路径管理。"""
from __future__ import annotations

import os
import ipaddress
from pathlib import Path
from typing import Any, Optional, Union

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
        self.hf_models_dir: Optional[Union[str, list[str]]] = global_cfg.get("hf_models_dir")
        self.mnn_models_dir: Optional[Union[str, list[str]]] = global_cfg.get("mnn_models_dir")
        self.llamacpp_models_dir: Optional[Union[str, list[str]]] = global_cfg.get("llamacpp_models_dir")
        self.train_out_dir: Optional[str] = global_cfg.get("train_out_dir")
        self.val_out_dir: Optional[str] = global_cfg.get("val_out_dir")
        self.test_out_dir: Optional[str] = global_cfg.get("test_out_dir")
        self.split: dict[str, Any] = global_cfg.get("split") or {}
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
        self.llamacpp_models_dir = global_cfg.get("llamacpp_models_dir")
        self.train_out_dir = global_cfg.get("train_out_dir")
        self.val_out_dir = global_cfg.get("val_out_dir")
        self.test_out_dir = global_cfg.get("test_out_dir")
        self.split = global_cfg.get("split") or {}

    def ensure_dirs(self) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.locks_dir.mkdir(parents=True, exist_ok=True)
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.trash_dir.mkdir(parents=True, exist_ok=True)

    @property
    def is_loopback_host(self) -> bool:
        """Whether binding this host keeps the WebUI local to this machine."""
        host = self.host.strip().strip("[]")
        if host.lower() == "localhost":
            return True
        # IPv6 zone identifiers (for example ``fe80::1%eth0``) are not
        # accepted as loopback unless their address portion says so.
        try:
            return ipaddress.ip_address(host.split("%", 1)[0]).is_loopback
        except ValueError:
            # An unrecognised hostname could resolve to a remote address.
            return False


_SETTINGS: Optional[Settings] = None


def get_settings(workspace_dir: Optional[Path | str] = None) -> Settings:
    global _SETTINGS
    if _SETTINGS is None or (workspace_dir and Path(workspace_dir).resolve() != _SETTINGS.workspace):
        _SETTINGS = Settings(workspace_dir=workspace_dir)
    return _SETTINGS


def set_settings(settings: Settings) -> None:
    global _SETTINGS
    _SETTINGS = settings
