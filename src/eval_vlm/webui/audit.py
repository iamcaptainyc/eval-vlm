"""Local WebUI operation audit logging."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from .settings import Settings


def write_audit(settings: Settings, user: str, action: str, details: dict[str, Any]) -> None:
    """Record a local WebUI operation without an authentication dependency."""
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "user": user,
        "action": action,
        "details": details,
    }
    with settings.audit_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
