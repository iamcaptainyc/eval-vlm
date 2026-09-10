"""任务调度管理器 (JobManager) 与实时日志流 (SSE)。"""
from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncGenerator, Optional

from ..config import load_dataset_config
from ..data.loader import load_samples
from .models import JobSummary
from .settings import Settings


class Job:
    def __init__(
        self,
        job_id: str,
        job_type: str,
        dataset: Optional[str],
        params: dict[str, Any],
        user: str,
        settings: Settings,
    ) -> None:
        self.id = job_id
        self.type = job_type
        self.dataset = dataset
        self.params = params
        self.user = user
        self.settings = settings
        self.dir = settings.jobs_dir / job_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.meta_file = self.dir / "meta.json"
        self.log_file = self.dir / "log.txt"

        self.status = "queued"
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.started_at: Optional[str] = None
        self.finished_at: Optional[str] = None
        self.exit_code: Optional[int] = None
        self.pid: Optional[int] = None
        self.progress: Optional[float] = None
        self.progress_msg: Optional[str] = None

        self.proc: Optional[asyncio.subprocess.Process] = None
        self.subscribers: list[asyncio.Queue[dict[str, Any]]] = []
        self.command: list[str] = []

        self.save_meta()

    def save_meta(self) -> None:
        data = {
            "id": self.id,
            "type": self.type,
            "dataset": self.dataset,
            "params": self.params,
            "user": self.user,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "exit_code": self.exit_code,
            "pid": self.pid,
            "progress": self.progress,
            "progress_msg": self.progress_msg,
            "command": self.command,
            "log_file": str(self.log_file.resolve()),
        }
        self.meta_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def to_summary(self, queue_pos: Optional[int] = None) -> JobSummary:
        return JobSummary(
            id=self.id,
            type=self.type,
            dataset=self.dataset,
            user=self.user,
            status=self.status,
            created_at=self.created_at,
            started_at=self.started_at,
            finished_at=self.finished_at,
            exit_code=self.exit_code,
            progress=self.progress,
            progress_msg=self.progress_msg,
            queue_position=queue_pos,
            command=self.command,
            log_file=str(self.log_file.resolve()),
            params=self.params,
        )

    def broadcast(self, event_type: str, data: Any) -> None:
        payload = {"event": event_type, "data": data}
        for q in list(self.subscribers):
            try:
                q.put_nowait(payload)
            except Exception:
                pass


class JobManager:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.jobs: dict[str, Job] = {}
        self.queue: asyncio.Queue[str] = asyncio.Queue()
        self.worker_task: Optional[asyncio.Task] = None
        self.current_job_id: Optional[str] = None
        self._reconcile_on_startup()

    def _reconcile_on_startup(self) -> None:
        """启动时对账：扫描未完成任务。"""
        if not self.settings.jobs_dir.exists():
            return
        for jdir in sorted(self.settings.jobs_dir.iterdir()):
            if not jdir.is_dir():
                continue
            mfile = jdir / "meta.json"
            if mfile.exists():
                try:
                    data = json.loads(mfile.read_text(encoding="utf-8"))
                    job = Job(
                        job_id=data["id"],
                        job_type=data["type"],
                        dataset=data.get("dataset"),
                        params=data.get("params", {}),
                        user=data.get("user", "anonymous"),
                        settings=self.settings,
                    )
                    job.created_at = data.get("created_at", job.created_at)
                    job.started_at = data.get("started_at")
                    job.finished_at = data.get("finished_at")
                    job.exit_code = data.get("exit_code")
                    job.pid = data.get("pid")
                    job.progress = data.get("progress")
                    job.progress_msg = data.get("progress_msg")
                    job.command = data.get("command") or self._build_cmd(job)

                    # 若原本标为 running 或 queued，服务重启后标为 interrupted
                    if data.get("status") in ("running", "queued"):
                        job.status = "interrupted"
                        job.finished_at = datetime.now(timezone.utc).isoformat()
                        job.save_meta()
                    else:
                        job.status = data.get("status", "failed")

                    self.jobs[job.id] = job
                except Exception:
                    continue

    def start_worker(self) -> None:
        try:
            loop = asyncio.get_running_loop()
            if self.worker_task is None or self.worker_task.done():
                self.worker_task = loop.create_task(self._queue_worker())
        except RuntimeError:
            pass

    def is_dataset_busy(self, dataset_name: str) -> bool:
        """检查该数据集是否有正在运行或排队的任务。"""
        for job in self.jobs.values():
            if job.dataset == dataset_name and job.status in ("queued", "running"):
                return True
        return False

    def submit_job(
        self,
        job_type: str,
        dataset: Optional[str],
        params: dict[str, Any],
        user: str = "anonymous",
    ) -> JobSummary:
        now_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        suffix = os.urandom(2).hex()
        ds_part = f"_{dataset}" if dataset else ""
        job_id = f"{job_type}{ds_part}_{now_str}_{suffix}"

        job = Job(
            job_id=job_id,
            job_type=job_type,
            dataset=dataset,
            params=params,
            user=user,
            settings=self.settings,
        )
        self._build_cmd(job)
        job.save_meta()
        self.jobs[job_id] = job
        self.queue.put_nowait(job_id)
        self.start_worker()

        queue_pos = self.queue.qsize()
        return job.to_summary(queue_pos=queue_pos)

    async def _queue_worker(self) -> None:
        while True:
            job_id = await self.queue.get()
            job = self.jobs.get(job_id)
            if not job or job.status == "canceled":
                self.queue.task_done()
                continue

            self.current_job_id = job_id
            job.status = "running"
            job.started_at = datetime.now(timezone.utc).isoformat()
            job.save_meta()
            job.broadcast("status", {"status": "running", "started_at": job.started_at})

            try:
                await self._execute_job(job)
            except Exception as e:
                job.status = "failed"
                job.progress_msg = f"执行异常: {e}"
                job.finished_at = datetime.now(timezone.utc).isoformat()
                job.save_meta()
                job.broadcast("status", {"status": "failed", "error": str(e)})
            finally:
                self.current_job_id = None
                self.queue.task_done()

    def _build_cmd(self, job: Job) -> list[str]:
        cmd = [sys.executable, "-m", "eval_vlm", job.type]
        if job.dataset:
            cmd.extend(["-d", job.dataset])
        cmd.extend(["--workspace", str(self.settings.workspace)])

        # 附加额外参数 (自动转换下划线为连字符: match_mode -> --match-mode)
        for k, v in (job.params or {}).items():
            arg_name = f"--{k.replace('_', '-')}"
            if v is True:
                cmd.append(arg_name)
            elif v is not False and v is not None and str(v).strip() != "":
                cmd.extend([arg_name, str(v).strip()])

        job.command = cmd
        return cmd

    async def _execute_job(self, job: Job) -> None:
        cmd = self._build_cmd(job)
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUNBUFFERED"] = "1"

        kwargs: dict[str, Any] = {
            "stdout": asyncio.subprocess.PIPE,
            "stderr": asyncio.subprocess.STDOUT,
            "cwd": str(self.settings.workspace),
            "env": env,
        }
        if sys.platform == "win32":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

        with job.log_file.open("a", encoding="utf-8") as f_log:
            f_log.write(f"=== 命令启动: {' '.join(cmd)} ===\n")
            f_log.write(f"=== 日志路径: {job.log_file.resolve()} ===\n")
            f_log.flush()

        proc = await asyncio.create_subprocess_exec(*cmd, **kwargs)
        job.proc = proc
        job.pid = proc.pid
        job.save_meta()
        job.broadcast("started", {
            "status": "running",
            "command": cmd,
            "log_file": str(job.log_file.resolve()),
            "pid": proc.pid,
        })

        # 读取输出并推流
        assert proc.stdout is not None
        with job.log_file.open("a", encoding="utf-8") as f_log:
            while True:
                line_bytes = await proc.stdout.readline()
                if not line_bytes:
                    break
                line_str = line_bytes.decode("utf-8", errors="replace")
                f_log.write(line_str)
                f_log.flush()
                job.broadcast("log", line_str)

        exit_code = await proc.wait()
        job.exit_code = exit_code
        job.finished_at = datetime.now(timezone.utc).isoformat()
        if job.status != "canceled":
            job.status = "succeeded" if exit_code == 0 else "failed"

        job.save_meta()
        job.broadcast(
            "status",
            {
                "status": job.status,
                "exit_code": exit_code,
                "finished_at": job.finished_at,
            },
        )

    def cancel_job(self, job_id: str) -> bool:
        job = self.jobs.get(job_id)
        if not job or job.status not in ("queued", "running"):
            return False

        if job.status == "queued":
            job.status = "canceled"
            job.finished_at = datetime.now(timezone.utc).isoformat()
            job.save_meta()
            job.broadcast("status", {"status": "canceled"})
            return True

        # running 状态优雅取消
        if job.proc:
            try:
                if sys.platform == "win32":
                    job.proc.send_signal(signal.CTRL_BREAK_EVENT)
                else:
                    job.proc.terminate()
            except Exception:
                try:
                    job.proc.kill()
                except Exception:
                    pass

        job.status = "canceled"
        job.finished_at = datetime.now(timezone.utc).isoformat()
        job.save_meta()
        job.broadcast("status", {"status": "canceled"})
        return True

    def list_jobs(self) -> list[JobSummary]:
        summaries: list[JobSummary] = []
        for jid, job in sorted(self.jobs.items(), key=lambda x: x[1].created_at, reverse=True):
            summaries.append(job.to_summary())
        return summaries

    def get_job(self, job_id: str) -> Optional[Job]:
        return self.jobs.get(job_id)

    async def stream_job_logs(
        self, job_id: str, offset: int = 0
    ) -> AsyncGenerator[str, None]:
        """SSE 事件流生成器。"""
        job = self.jobs.get(job_id)
        if not job:
            yield f"event: error\ndata: {json.dumps({'message': 'Job not found'})}\n\n"
            return

        # 1. 首次连线：回放历史日志
        if job.log_file.exists():
            with job.log_file.open("r", encoding="utf-8", errors="replace") as f:
                if offset > 0:
                    f.seek(offset)
                content = f.read()
                if content:
                    yield f"event: log\ndata: {json.dumps(content)}\n\n"

        # 推送当前状态
        yield f"event: status\ndata: {json.dumps({'status': job.status, 'exit_code': job.exit_code})}\n\n"

        if job.status not in ("queued", "running"):
            return

        # 2. 挂载到广播订阅
        q: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        job.subscribers.append(q)

        try:
            while True:
                msg = await q.get()
                ev = msg.get("event", "log")
                data = msg.get("data")
                yield f"event: {ev}\ndata: {json.dumps(data)}\n\n"
                if ev == "status" and data.get("status") in ("succeeded", "failed", "canceled", "interrupted"):
                    break
        finally:
            if q in job.subscribers:
                job.subscribers.remove(q)


_JOB_MANAGER: Optional[JobManager] = None


def get_job_manager(settings: Settings) -> JobManager:
    global _JOB_MANAGER
    if _JOB_MANAGER is None:
        _JOB_MANAGER = JobManager(settings)
    return _JOB_MANAGER
