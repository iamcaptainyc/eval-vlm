"""WebUI 任务管理器与调度状态机测试。"""
from __future__ import annotations

import asyncio
from pathlib import Path
import pytest

from eval_vlm.webui.jobs import JobManager
from eval_vlm.webui.settings import Settings


@pytest.fixture
def job_mgr(tmp_path):
    settings = Settings(workspace_dir=tmp_path)
    return JobManager(settings)


def test_submit_and_list_jobs(job_mgr):
    summary = job_mgr.submit_job(
        job_type="eval",
        dataset="demo_ds",
        params={"overwrite": True},
        user="test_user",
    )
    assert summary.id.startswith("eval_demo_ds_")
    assert summary.status == "queued"
    assert summary.dataset == "demo_ds"

    jobs = job_mgr.list_jobs()
    assert len(jobs) == 1
    assert jobs[0].id == summary.id

    assert job_mgr.is_dataset_busy("demo_ds") is True
    assert job_mgr.is_dataset_busy("other_ds") is False


def test_cancel_queued_job(job_mgr):
    summary = job_mgr.submit_job(
        job_type="pred",
        dataset="ds2",
        params={},
        user="test_user",
    )
    ok = job_mgr.cancel_job(summary.id)
    assert ok is True

    job = job_mgr.get_job(summary.id)
    assert job.status == "canceled"
    assert job_mgr.is_dataset_busy("ds2") is False


@pytest.mark.anyio
async def test_job_log_streaming_sse(job_mgr):
    from eval_vlm.webui.jobs import Job
    job = Job(
        job_id="score_ds3_test",
        job_type="score",
        dataset="ds3",
        params={},
        user="test_user",
        settings=job_mgr.settings,
    )
    job.status = "succeeded"
    job.log_file.write_text("line 1\nline 2\n", encoding="utf-8")
    job_mgr.jobs[job.id] = job

    # 获取 SSE 流并取前几个事件
    stream = job_mgr.stream_job_logs(job.id, offset=0)
    events = []
    async for item in stream:
        events.append(item)

    assert any("line 1" in ev for ev in events)
    assert any("status" in ev for ev in events)
