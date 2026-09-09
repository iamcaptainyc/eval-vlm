"""鉴权与审计模块。"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Optional

import yaml
from fastapi import Depends, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBasic, HTTPBasicCredentials, HTTPBearer
from pydantic import BaseModel

from .settings import Settings, get_settings


class User(BaseModel):
    username: str
    role: str  # "viewer" | "editor"


basic_security = HTTPBasic(auto_error=False)
bearer_security = HTTPBearer(auto_error=False)


def _load_users(settings: Settings) -> dict[str, dict[str, str]]:
    if not settings.users_file.exists():
        return {}
    try:
        data = yaml.safe_load(settings.users_file.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _verify_password(input_pw: str, stored: str) -> bool:
    if input_pw == stored:
        return True
    hashed = hashlib.sha256(input_pw.encode("utf-8")).hexdigest()
    return hashed == stored


def get_current_user(
    basic_creds: Optional[HTTPBasicCredentials] = Security(basic_security),
    bearer_creds: Optional[HTTPAuthorizationCredentials] = Security(bearer_security),
    settings: Settings = Depends(get_settings),
) -> User:
    users = _load_users(settings)
    configured_token = settings.token

    # 如果既无 users.yaml 也无 EVAL_VLM_WEBUI_TOKEN，本地免密开发模式：默认 editor
    if not users and not configured_token:
        return User(username="anonymous", role="editor")

    # 1. Bearer Token 校验
    if bearer_creds and bearer_creds.credentials:
        token = bearer_creds.credentials
        if configured_token and token == configured_token:
            return User(username="token_user", role="editor")
        # 也可以在 users.yaml 里查找 token
        if token in users:
            role = users[token].get("role", "editor")
            return User(username=token, role=role)

    # 2. Basic Auth 校验
    if basic_creds and basic_creds.username:
        uname = basic_creds.username
        pwd = basic_creds.password or ""
        if configured_token and (uname == "admin" or uname == "token") and pwd == configured_token:
            return User(username=uname, role="editor")
        if uname in users:
            info = users[uname]
            expected_pwd = str(info.get("password", info.get("password_hash", "")))
            if _verify_password(pwd, expected_pwd):
                return User(username=uname, role=info.get("role", "viewer"))

    # 鉴权失败
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="未提供有效凭据或凭据不正确",
        headers={"WWW-Authenticate": "Basic"},
    )


def require_viewer(user: User = Depends(get_current_user)) -> User:
    if user.role in ("viewer", "editor"):
        return user
    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="权限不足: 需要查看权限")


def require_editor(user: User = Depends(get_current_user)) -> User:
    if user.role == "editor":
        return user
    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="权限不足: 需要编辑权限")


def write_audit(settings: Settings, user: str, action: str, details: dict[str, Any]) -> None:
    """记录操作审计日志。"""
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "user": user,
        "action": action,
        "details": details,
    }
    with settings.audit_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
