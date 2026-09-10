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


def test_job_command_canonical_and_targets(job_mgr):
    """测试命令构建使用规范前缀 python -m eval_vlm，且正确过滤空参数与支持数字 targets。"""
    summary = job_mgr.submit_job(
        job_type="field-eval",
        dataset="test_ds",
        params={
            "match_mode": "",       # 保持默认，应被过滤不出现在命令中
            "targets": 2,           # 数字 targets（第2轮）
            "limit": 5,             # 样本数量限制
            "overwrite": True,
        },
        user="test_user",
    )
    cmd = summary.command
    assert cmd[0] == "python"
    assert cmd[1] == "-m"
    assert cmd[2] == "eval_vlm"
    assert cmd[3] == "field-eval"
    assert "-d" in cmd and cmd[cmd.index("-d") + 1] == "test_ds"
    assert "--match-mode" not in cmd  # 空值被正确过滤
    assert "--targets" in cmd and cmd[cmd.index("--targets") + 1] == "2"
    assert "--limit" in cmd and cmd[cmd.index("--limit") + 1] == "5"
    assert "--overwrite" in cmd


def test_cli_targets_and_limit_parsing():
    """测试 CLI parser 对各子命令的 --targets 与 --limit 参数解析。"""
    from eval_vlm.cli import build_parser

    parser = build_parser()

    # field-eval 带数字 targets 与 limit
    args = parser.parse_args(["field-eval", "-d", "ds1", "--targets", "2", "--limit", "10"])
    assert args.targets == 2
    assert isinstance(args.targets, int)
    assert args.limit == 10

    # eval 带 targets first 与 limit
    args_eval = parser.parse_args(["eval", "-d", "ds1", "--targets", "first", "--limit", "20"])
    assert args_eval.targets == "first"
    assert args_eval.limit == 20

    # score 带 targets all 与 limit
    args_score = parser.parse_args(["score", "-d", "ds1", "--targets", "all", "--limit", "5"])
    assert args_score.targets == "all"
    assert args_score.limit == 5

    # pred 带 targets 3 与 limit
    args_pred = parser.parse_args(["pred", "--dataset", "ds1", "--targets", "3", "--limit", "15"])
    assert args_pred.targets == 3
    assert isinstance(args_pred.targets, int)
    assert args_pred.limit == 15


@pytest.mark.anyio
async def test_job_worker_execution_and_log_emission(tmp_path):
    """测试任务调度器在活跃事件循环中正确执行子进程并输出日志。"""
    settings = Settings(workspace_dir=tmp_path)
    mgr = JobManager(settings)
    loop = asyncio.get_running_loop()
    mgr.start_worker(loop=loop)

    summary = mgr.submit_job(
        job_type="sweep",
        dataset="demo_ds",
        params={"backend": "fake", "dry_run": True},
        user="test_worker",
    )
    assert summary.id in mgr.jobs

    # 等待任务进入运行或完成
    for _ in range(30):
        await asyncio.sleep(0.1)
        job = mgr.get_job(summary.id)
        if job and job.status in ("succeeded", "failed"):
            break

    job = mgr.get_job(summary.id)
    assert job is not None
    assert job.status in ("succeeded", "failed")
    assert job.log_file.exists()
    log_content = job.log_file.read_text(encoding="utf-8")
    assert "=== 正在启动任务" in log_content
    assert "=== 任务进程已就绪" in log_content


