"""WebUI 任务管理器与调度状态机测试。"""
from __future__ import annotations

import asyncio
import json
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


def test_job_ids_keep_sweep_short_and_sanitize_regular_dataset(job_mgr):
    datasets = ",".join(f"very-long-dataset-{n}" for n in range(20))
    first = job_mgr.submit_job("sweep", datasets, {}, "tester")
    second = job_mgr.submit_job("sweep", datasets, {}, "tester")
    assert first.id.startswith("sweep_")
    assert len(first.id) < 40
    assert "," not in first.id and "very-long-dataset" not in first.id
    assert first.id != second.id

    normal = job_mgr.submit_job("eval", "../unsafe name/" + "x" * 100, {}, "tester")
    assert len(normal.id) < 100
    assert "/" not in normal.id and "\\" not in normal.id and " " not in normal.id


def test_startup_restore_preserves_metadata_and_interrupts_active(tmp_path):
    settings = Settings(workspace_dir=tmp_path)
    job_dir = settings.jobs_dir / "sweep_20260101T000000Z_abcdef"
    job_dir.mkdir(parents=True)
    meta = {
        "id": job_dir.name, "type": "sweep", "dataset": "a,b,c",
        "params": {"model": "keep-me"}, "user": "saved-user", "status": "running",
        "created_at": "2026-01-01T00:00:00+00:00", "started_at": "2026-01-01T00:01:00+00:00",
        "finished_at": None, "exit_code": None, "pid": 123, "progress": 0.5,
        "progress_msg": "halfway", "command": ["saved-command"],
    }
    (job_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    manager = JobManager(settings)
    restored = manager.get_job(meta["id"])
    assert restored is not None
    assert restored.status == "interrupted"
    assert restored.params == {"model": "keep-me"}
    assert restored.user == "saved-user"
    assert restored.command == ["saved-command"]
    persisted = json.loads((job_dir / "meta.json").read_text(encoding="utf-8"))
    assert persisted["dataset"] == "a,b,c"
    assert persisted["status"] == "interrupted"

    completed_dir = settings.jobs_dir / "eval_completed"
    completed_dir.mkdir()
    completed_meta = {
        **meta,
        "id": completed_dir.name,
        "type": "eval",
        "dataset": "done-dataset",
        "status": "succeeded",
        "finished_at": "2026-01-01T00:02:00+00:00",
        "exit_code": 0,
    }
    (completed_dir / "meta.json").write_text(json.dumps(completed_meta), encoding="utf-8")
    reloaded = JobManager(settings)
    completed = reloaded.get_job(completed_dir.name)
    assert completed is not None
    assert completed.status == "succeeded"
    assert completed.exit_code == 0
    completed_persisted = json.loads((completed_dir / "meta.json").read_text(encoding="utf-8"))
    assert completed_persisted["status"] == "succeeded"


def test_job_frontend_uses_single_flight_polling_and_session_scoped_logs():
    source = (Path(__file__).parents[1] / "src" / "eval_vlm" / "webui" / "static" / "app.js").read_text(encoding="utf-8")
    open_terminal = source.split("async function openTerminal(jobId)", 1)[1].split("function isTerminalJob", 1)[0]
    assert "jobLoadPromise" in source
    assert "updateJobPolling" in source
    assert "terminalSession" in source
    assert "const isCurrent" in source
    assert "AbortController" in source
    assert "controller.abort(), 3000" in source
    assert "connectTerminalStream(jobId, session, 0);" in open_terminal
    assert "void refreshTerminalSnapshot(jobId, session);" in open_terminal
    assert open_terminal.index("connectTerminalStream(jobId, session, 0);") < open_terminal.index("void refreshTerminalSnapshot(jobId, session);")
    assert "日志流暂时断开" in source


@pytest.mark.anyio
async def test_cancel_queued_job(job_mgr):
    summary = job_mgr.submit_job(
        job_type="pred",
        dataset="ds2",
        params={},
        user="test_user",
    )
    ok = await job_mgr.cancel_job(summary.id)
    assert ok is True

    job = job_mgr.get_job(summary.id)
    assert job.status == "canceled"
    assert job_mgr.is_dataset_busy("ds2") is False


@pytest.mark.anyio
async def test_cancel_running_job_waits_for_exit_before_releasing_dataset(job_mgr):
    """A running job remains busy until its child has accepted termination."""
    from eval_vlm.webui.jobs import Job

    class FakeProcess:
        def __init__(self):
            self.returncode = None
            self.terminated = False
            self.exit_gate = asyncio.Event()

        def terminate(self):
            self.terminated = True

        def send_signal(self, _signal):
            # JobManager uses CTRL_BREAK_EVENT on Windows.
            self.terminate()

        async def wait(self):
            await self.exit_gate.wait()
            self.returncode = -15
            return self.returncode

    job = Job("eval_ds_running", "eval", "ds-running", {}, "tester", job_mgr.settings)
    job.status = "running"
    job.proc = FakeProcess()
    job_mgr.jobs[job.id] = job

    cancel = asyncio.create_task(job_mgr.cancel_job(job.id))
    await asyncio.sleep(0)
    assert job_mgr.is_dataset_busy("ds-running") is True
    job.proc.exit_gate.set()
    assert await cancel is True
    assert job.proc.terminated is True
    assert job.status == "canceled"
    assert job_mgr.is_dataset_busy("ds-running") is False


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


@pytest.mark.anyio
async def test_sse_subscribes_before_terminal_transition_and_cleans_up(job_mgr):
    from eval_vlm.webui.jobs import Job
    job = Job("eval_sse_race", "eval", "ds", {}, "tester", job_mgr.settings)
    job.status = "running"
    job.log_file.write_text("history\n", encoding="utf-8")
    job_mgr.jobs[job.id] = job
    stream = job_mgr.stream_job_logs(job.id)
    first = await anext(stream)
    assert "history" in first
    assert len(job.subscribers) == 1
    job.status = "succeeded"
    job.broadcast("status", {"status": "succeeded", "exit_code": 0})
    events = [first]
    async for event in stream:
        events.append(event)
    assert any("succeeded" in event for event in events)
    assert job.subscribers == []


@pytest.mark.anyio
async def test_process_exit_finalizes_even_when_stdout_never_reaches_eof(job_mgr, monkeypatch):
    from eval_vlm.webui.jobs import Job

    class NeverEofStream:
        async def readline(self):
            await asyncio.Event().wait()

    class ExitedProcess:
        pid = 4321
        stdout = NeverEofStream()
        returncode = None

        async def wait(self):
            self.returncode = 0
            return 0

    async def fake_create(*_args, **_kwargs):
        return ExitedProcess()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create)
    job = Job("eval_stuck_pipe", "eval", "ds", {}, "tester", job_mgr.settings)
    job.status = "running"
    await asyncio.wait_for(job_mgr._execute_job(job), timeout=2)
    assert job.status == "succeeded"
    assert job.exit_code == 0
    assert "任务执行结束" in job.log_file.read_text(encoding="utf-8")


def test_list_jobs_reconciles_a_process_that_already_exited(job_mgr):
    from eval_vlm.webui.jobs import Job

    class ExitedProcess:
        returncode = 0

    job = Job("eval_finished_before_poll", "eval", "ds", {}, "tester", job_mgr.settings)
    job.status = "running"
    job.proc = ExitedProcess()
    job_mgr.jobs[job.id] = job

    summary = next(item for item in job_mgr.list_jobs() if item.id == job.id)
    assert summary.status == "succeeded"
    assert summary.exit_code == 0
    persisted = json.loads(job.meta_file.read_text(encoding="utf-8"))
    assert persisted["status"] == "succeeded"


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
    await mgr.shutdown()


@pytest.mark.anyio
async def test_parallel_execution_eval_and_convert(tmp_path, monkeypatch):
    """测试评测任务 (eval) 与格式转换任务 (convert-gguf) 能够在多通道中并行运行。"""
    settings = Settings(workspace_dir=tmp_path)
    mgr = JobManager(settings)
    loop = asyncio.get_running_loop()
    mgr.start_worker(loop=loop)

    eval_started = asyncio.Event()
    eval_finish = asyncio.Event()
    convert_started = asyncio.Event()
    convert_finish = asyncio.Event()

    async def fake_execute_job(job):
        if job.type == "eval":
            eval_started.set()
            await eval_finish.wait()
        elif job.type == "convert-gguf":
            convert_started.set()
            await convert_finish.wait()
        job.status = "succeeded"
        job.exit_code = 0

    monkeypatch.setattr(mgr, "_execute_job", fake_execute_job)

    # 1. 提交 eval 任务
    job1 = mgr.submit_job("eval", "ds_eval", {}, "user1")
    # 2. 提交 convert-gguf 任务
    job2 = mgr.submit_job("convert-gguf", None, {"hf_path": "/path/hf"}, "user2")

    # 等待两个任务均开始执行
    await asyncio.wait_for(asyncio.gather(eval_started.wait(), convert_started.wait()), timeout=2.0)

    # 验证两个任务同时处于 running 状态且在 running_job_ids 集合中
    assert mgr.get_job(job1.id).status == "running"
    assert mgr.get_job(job2.id).status == "running"
    assert job1.id in mgr.running_job_ids
    assert job2.id in mgr.running_job_ids

    # 释放两个任务
    eval_finish.set()
    convert_finish.set()
    await asyncio.sleep(0.05)
    await mgr.shutdown()


@pytest.mark.anyio
async def test_cancel_one_lane_does_not_affect_other(tmp_path, monkeypatch):
    """测试取消一个通道的运行任务不会影响另一个通道的运行任务。"""
    settings = Settings(workspace_dir=tmp_path)
    mgr = JobManager(settings)
    loop = asyncio.get_running_loop()
    mgr.start_worker(loop=loop)

    eval_started = asyncio.Event()
    eval_wait = asyncio.Event()
    convert_started = asyncio.Event()
    convert_canceled = asyncio.Event()

    async def fake_execute_job(job):
        if job.type == "eval":
            eval_started.set()
            await eval_wait.wait()
        elif job.type == "convert-gguf":
            convert_started.set()
            while not job.cancel_requested:
                await asyncio.sleep(0.01)
            convert_canceled.set()
        job.status = "succeeded"
        job.exit_code = 0

    monkeypatch.setattr(mgr, "_execute_job", fake_execute_job)

    job_eval = mgr.submit_job("eval", "ds_eval", {}, "user1")
    job_convert = mgr.submit_job("convert-gguf", None, {"hf_path": "/path/hf"}, "user2")

    await asyncio.wait_for(asyncio.gather(eval_started.wait(), convert_started.wait()), timeout=2.0)

    # 取消 convert-gguf
    ok = await mgr.cancel_job(job_convert.id)
    assert ok is True
    await asyncio.wait_for(convert_canceled.wait(), timeout=1.0)

    # 验证 eval 依然在运行中
    assert mgr.get_job(job_eval.id).status == "running"
    assert job_eval.id in mgr.running_job_ids

    eval_wait.set()
    await asyncio.sleep(0.05)
    await mgr.shutdown()


def test_multi_queue_positions(job_mgr):
    """测试不同通道的排队位置独立统计。"""
    eval1 = job_mgr.submit_job("eval", "ds1", {}, "user1")
    eval2 = job_mgr.submit_job("eval", "ds1", {}, "user2")
    conv1 = job_mgr.submit_job("convert-gguf", None, {}, "user3")
    conv2 = job_mgr.submit_job("convert-gguf", None, {}, "user4")

    # 未启动 worker 时，按通道独立统计 queue_position
    assert eval1.queue_position == 1
    assert eval2.queue_position == 2
    assert conv1.queue_position == 1
    assert conv2.queue_position == 2

