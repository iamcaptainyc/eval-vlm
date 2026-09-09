"""并发控制与锁机制。"""
from __future__ import annotations

import asyncio
import hashlib
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator, Optional

from fastapi import HTTPException, status

try:
    from filelock import FileLock, Timeout as LockTimeout
except ImportError:
    class LockTimeout(Exception):
        pass

    class FileLock:
        def __init__(self, lock_file: str | Path, timeout: float = -1):
            self.lock_file = Path(lock_file)
            self.timeout = timeout

        def acquire(self, timeout: Optional[float] = None):
            return self

        def release(self):
            pass


from .settings import Settings

_MEMORY_LOCKS: dict[str, asyncio.Lock] = {}


def _get_dataset_mem_lock(name: str) -> asyncio.Lock:
    if name not in _MEMORY_LOCKS:
        _MEMORY_LOCKS[name] = asyncio.Lock()
    return _MEMORY_LOCKS[name]


def calc_file_sha256(path: Path) -> str:
    """计算文件 SHA-256 哈希。"""
    if not path.exists():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_test_sha(test_path: Path, expected_sha256: str) -> str:
    """校验当前 test.json 的哈希与客户端提交的是否一致。"""
    current_sha = calc_file_sha256(test_path)
    if expected_sha256 and current_sha != expected_sha256:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "message": "test.json 已被他人或其他操作修改，请刷新后重试",
                "current_sha256": current_sha,
                "expected_sha256": expected_sha256,
            },
        )
    return current_sha


@asynccontextmanager
async def dataset_lock(
    name: str, settings: Settings, timeout: float = 10.0
) -> AsyncGenerator[None, None]:
    """数据集操作锁：进程内 asyncio.Lock + 跨进程文件锁。"""
    mem_lock = _get_dataset_mem_lock(name)
    try:
        await asyncio.wait_for(mem_lock.acquire(), timeout=timeout)
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"获取数据集 '{name}' 内存锁超时，有其他操作正在进行",
        )

    disk_lock_file = settings.locks_dir / f"{name}.lock"
    file_lock = FileLock(disk_lock_file, timeout=timeout)

    try:
        file_lock.acquire(timeout=timeout)
    except LockTimeout:
        mem_lock.release()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"获取数据集 '{name}' 文件锁超时，可能有其他进程正在编辑",
        )
    except Exception as e:
        mem_lock.release()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取数据集锁异常: {e}",
        )

    try:
        yield
    finally:
        try:
            file_lock.release()
        except Exception:
            pass
        mem_lock.release()
