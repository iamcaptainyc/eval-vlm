"""WebUI 并发锁与文件锁测试。"""
from __future__ import annotations

import asyncio
from pathlib import Path
import pytest
from fastapi import HTTPException

from eval_vlm.webui.locks import dataset_lock, calc_file_sha256, verify_test_sha
from eval_vlm.webui.settings import Settings


@pytest.mark.anyio
async def test_dataset_lock_mutual_exclusion(tmp_path):
    settings = Settings(workspace_dir=tmp_path)
    ds_name = "test_ds"

    locked_count = 0
    max_concurrent = 0

    async def worker():
        nonlocal locked_count, max_concurrent
        async with dataset_lock(ds_name, settings, timeout=5.0):
            locked_count += 1
            max_concurrent = max(max_concurrent, locked_count)
            await asyncio.sleep(0.05)
            locked_count -= 1

    # 启动 5 个并发任务抢锁
    await asyncio.gather(*(worker() for _ in range(5)))
    assert max_concurrent == 1
    assert locked_count == 0


def test_calc_and_verify_sha256(tmp_path):
    f = tmp_path / "test.json"
    f.write_text('{"a": 1}', encoding="utf-8")
    sha = calc_file_sha256(f)
    assert len(sha) == 64

    # 校验一致
    assert verify_test_sha(f, sha) == sha

    # 校验不一致抛出 409
    with pytest.raises(HTTPException) as exc_info:
        verify_test_sha(f, "bad_sha")
    assert exc_info.value.status_code == 409
