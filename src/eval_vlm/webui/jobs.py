"""任务调度管理器 (JobManager) 与实时日志流 (SSE)。"""
from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
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
        *,
        persist: bool = True,
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
        self.cancel_requested = False
        self.subscribers: list[asyncio.Queue[dict[str, Any]]] = []
        self.command: list[str] = []
        self._finalized = False

        if persist:
            self.save_meta()

    @classmethod
    def from_meta(cls, data: dict[str, Any], settings: Settings) -> "Job":
        """Restore a job without first writing default values over its metadata."""
        job = cls(
            job_id=data["id"], job_type=data["type"], dataset=data.get("dataset"),
            params=data.get("params") or {}, user=data.get("user", "anonymous"),
            settings=settings, persist=False,
        )
        for field in (
            "status", "created_at", "started_at", "finished_at", "exit_code",
            "pid", "progress", "progress_msg",
        ):
            if field in data:
                setattr(job, field, data[field])
        job.command = data.get("command") or []
        return job

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


LANE_EVAL = "eval"
LANE_CONVERT = "convert"
TOOL_JOB_TYPES = {"convert-gguf"}


def get_job_lane(job_type: str) -> str:
    """按任务类型划分调度通道：格式转换/工具任务独立通道并行执行，评测推理任务主通道串行执行。"""
    if job_type in TOOL_JOB_TYPES:
        return LANE_CONVERT
    return LANE_EVAL


class JobManager:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.jobs: dict[str, Job] = {}
        self.queues: dict[str, asyncio.Queue[str]] = {
            LANE_EVAL: asyncio.Queue(),
            LANE_CONVERT: asyncio.Queue(),
        }
        self.worker_tasks: dict[str, Optional[asyncio.Task]] = {
            LANE_EVAL: None,
            LANE_CONVERT: None,
        }
        self.running_job_ids: set[str] = set()
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self._shutdown = False
        self._reconcile_on_startup()

    @property
    def queue(self) -> asyncio.Queue[str]:
        """兼容性属性：主评测通道队列。"""
        return self.queues[LANE_EVAL]

    @property
    def worker_task(self) -> Optional[asyncio.Task]:
        """兼容性属性：主评测通道 Worker 协程任务。"""
        return self.worker_tasks.get(LANE_EVAL)

    @property
    def current_job_id(self) -> Optional[str]:
        """兼容性属性：当前正在运行的任务 ID（若有多个并发运行，返回任意一个）。"""
        return next(iter(self.running_job_ids), None)

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
                    job = Job.from_meta(data, self.settings)
                    if not job.command:
                        self._build_cmd(job)

                    # 若原本标为 running 或 queued，服务重启后标为 interrupted
                    if job.status in ("running", "queued"):
                        job.status = "interrupted"
                        job.finished_at = datetime.now(timezone.utc).isoformat()
                        job.progress_msg = "WebUI 重启时任务尚未结束，已标记为中断"
                        job.save_meta()

                    self.jobs[job.id] = job
                except Exception:
                    continue

    def start_worker(self, loop: Optional[asyncio.AbstractEventLoop] = None) -> None:
        if loop is not None:
            self.loop = loop
        elif self.loop is None or self.loop.is_closed():
            try:
                self.loop = asyncio.get_running_loop()
            except RuntimeError:
                pass

        if self.loop is not None and self.loop.is_running():
            for lane in (LANE_EVAL, LANE_CONVERT):
                task = self.worker_tasks.get(lane)
                if task is None or task.done():
                    self.worker_tasks[lane] = self.loop.create_task(self._queue_worker(lane))

    def is_dataset_busy(self, dataset_name: str) -> bool:
        """检查该数据集是否有正在运行或排队的任务。"""
        for job in self.jobs.values():
            # A cancel request keeps the job running until its child process has
            # actually exited, so edits cannot race a still-writing CLI process.
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
        now_str = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        suffix = os.urandom(3).hex()
        # A sweep dataset is usually a comma-delimited list and must never be
        # mirrored into a directory name.  Keep it in metadata only.
        if job_type == "sweep":
            job_id = f"sweep_{now_str}_{suffix}"
        else:
            safe_dataset = re.sub(r"[^A-Za-z0-9._-]+", "-", dataset or "").strip(".-")
            safe_dataset = safe_dataset[:48]
            ds_part = f"_{safe_dataset}" if safe_dataset else ""
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

        # 确保 worker 调度循环已就绪
        self.start_worker()

        lane = get_job_lane(job_type)
        target_queue = self.queues[lane]

        # 安全入队（兼容跨线程或异步事件循环环境）
        if self.loop is not None and self.loop.is_running():
            try:
                curr_loop = asyncio.get_running_loop()
                if curr_loop is self.loop:
                    target_queue.put_nowait(job_id)
                else:
                    self.loop.call_soon_threadsafe(target_queue.put_nowait, job_id)
            except RuntimeError:
                self.loop.call_soon_threadsafe(target_queue.put_nowait, job_id)
        else:
            target_queue.put_nowait(job_id)

        queue_pos = target_queue.qsize()
        return job.to_summary(queue_pos=queue_pos)

    async def _queue_worker(self, lane: str) -> None:
        target_queue = self.queues[lane]
        while True:
            job_id = await target_queue.get()
            job = self.jobs.get(job_id)
            if not job or job.status == "canceled":
                target_queue.task_done()
                continue

            self.running_job_ids.add(job_id)
            job.status = "running"
            job.started_at = datetime.now(timezone.utc).isoformat()
            job.save_meta()
            job.broadcast("status", {"status": "running", "started_at": job.started_at})

            try:
                await self._execute_job(job)
            except asyncio.CancelledError:
                job.cancel_requested = True
                await self._stop_process(job, grace_period=0.5)
                await self._finalize_job(job, exit_code=getattr(job.proc, "returncode", None), force_status="canceled")
                raise
            except Exception as e:
                job.status = "failed"
                job.progress_msg = f"执行异常: {e}"
                job.finished_at = datetime.now(timezone.utc).isoformat()
                job.save_meta()
                job.broadcast("status", {"status": "failed", "error": str(e)})
            finally:
                self.running_job_ids.discard(job_id)
                target_queue.task_done()

    async def _stop_process(self, job: Job, grace_period: float = 5.0) -> None:
        """Request process termination and wait for a definitive exit.

        Keep ``job.status`` as running during this operation.  This is
        intentional: callers relying on dataset locks must not see an idle
        dataset while the child can still modify its files.
        """
        proc = job.proc
        if proc is None or proc.returncode is not None:
            return
        try:
            if sys.platform == "win32":
                proc.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                proc.terminate()
        except (ProcessLookupError, OSError):
            return

        try:
            await asyncio.wait_for(proc.wait(), timeout=grace_period)
            return
        except asyncio.TimeoutError:
            pass

        try:
            proc.kill()
        except (ProcessLookupError, OSError):
            return
        try:
            await proc.wait()
        except (ProcessLookupError, OSError):
            pass

    def _build_cmd(self, job: Job) -> list[str]:
        cmd = ["python", "-m", "eval_vlm", job.type]
        if job.dataset:
            cmd.extend(["-d", job.dataset])

        # 仅在 settings.workspace 确实与全局默认 workspace 不同时才附加 --workspace
        try:
            from eval_vlm import workspace
            global_ws = workspace.resolve_workspace(None, workspace.load_global_config())
            if self.settings.workspace.resolve() != global_ws.resolve():
                cmd.extend(["--workspace", str(self.settings.workspace)])
        except Exception:
            pass

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
        # 强制添加 -u 参数确保 Python 子进程标准输入输出完全无缓冲 (Unbuffered binary stdout/stderr)
        exec_cmd = [sys.executable, "-u"] + cmd[1:]
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

        start_line1 = f"=== 正在启动任务: {' '.join(cmd)} ===\n"
        start_line2 = f"=== 工作区目录: {self.settings.workspace} ===\n"
        start_line3 = f"=== 日志文件: {job.log_file.resolve()} ===\n"
        with job.log_file.open("a", encoding="utf-8") as f_log:
            f_log.write(start_line1)
            f_log.write(start_line2)
            f_log.write(start_line3)
            f_log.flush()

        job.broadcast("log", start_line1)
        job.broadcast("log", start_line2)
        job.broadcast("log", start_line3)

        try:
            proc = await asyncio.create_subprocess_exec(*exec_cmd, **kwargs)
        except Exception as e:
            err_line = f"=== 启动进程失败: {e} ===\n"
            with job.log_file.open("a", encoding="utf-8") as f_log:
                f_log.write(err_line)
                f_log.flush()
            job.broadcast("log", err_line)
            job.status = "canceled" if job.cancel_requested else "failed"
            job.progress_msg = "任务在启动前已取消" if job.cancel_requested else f"启动失败: {e}"
            job.finished_at = datetime.now(timezone.utc).isoformat()
            job.save_meta()
            job.broadcast("status", {"status": job.status, "error": str(e)})
            return

        job.proc = proc
        job.pid = proc.pid
        pid_line = f"=== 任务进程已就绪 (PID: {proc.pid})，开始执行并实时推流 ===\n\n"
        with job.log_file.open("a", encoding="utf-8") as f_log:
            f_log.write(pid_line)
            f_log.flush()
        job.broadcast("log", pid_line)

        job.save_meta()
        job.broadcast("started", {
            "status": "running",
            "command": cmd,
            "log_file": str(job.log_file.resolve()),
            "pid": proc.pid,
        })

        # Cancellation can arrive in the small window between setting running
        # and creating the subprocess.  Honour it before consuming output.
        if job.cancel_requested:
            await self._stop_process(job)

        # Waiting for process exit and draining stdout are independent.  In
        # particular, inherited stdout handles can keep readline() blocked
        # after the child has already exited.
        drain_task = asyncio.create_task(self._drain_stdout(job, proc))
        try:
            exit_code = await proc.wait()
            try:
                await asyncio.wait_for(asyncio.shield(drain_task), timeout=0.75)
            except asyncio.TimeoutError:
                drain_task.cancel()
                await asyncio.gather(drain_task, return_exceptions=True)
            await self._finalize_job(job, exit_code=exit_code)
        except asyncio.CancelledError:
            drain_task.cancel()
            await asyncio.gather(drain_task, return_exceptions=True)
            raise
        finally:
            if not drain_task.done():
                drain_task.cancel()

    async def _drain_stdout(self, job: Job, proc: asyncio.subprocess.Process) -> None:
        assert proc.stdout is not None
        with job.log_file.open("a", encoding="utf-8") as f_log:
            while line_bytes := await proc.stdout.readline():
                line_str = line_bytes.decode("utf-8", errors="replace")
                f_log.write(line_str)
                f_log.flush()
                job.broadcast("log", line_str)

    async def _finalize_job(
        self, job: Job, *, exit_code: Optional[int], force_status: Optional[str] = None
    ) -> None:
        """Persist and announce one terminal state exactly once."""
        if job._finalized:
            return
        job.exit_code = exit_code
        job.finished_at = job.finished_at or datetime.now(timezone.utc).isoformat()
        if force_status:
            job.status = force_status
        elif job.cancel_requested:
            job.status = "canceled"
        else:
            job.status = "succeeded" if exit_code == 0 else "failed"
        finish_line = f"\n=== 任务执行结束 (PID: {job.pid}, 状态: {job.status}, 退出码: {exit_code}) ===\n"
        with job.log_file.open("a", encoding="utf-8") as f_log:
            f_log.write(finish_line)
            f_log.flush()
        job.broadcast("log", finish_line)
        job.save_meta()
        job.broadcast("status", {"status": job.status, "exit_code": exit_code, "finished_at": job.finished_at})
        job._finalized = True

    async def cancel_job(self, job_id: str) -> bool:
        job = self.jobs.get(job_id)
        if not job or job.status not in ("queued", "running"):
            return False

        cancel_msg = f"\n=== 任务已手动取消终止 ({datetime.now(timezone.utc).isoformat()}) ===\n"
        try:
            with job.log_file.open("a", encoding="utf-8") as f_log:
                f_log.write(cancel_msg)
                f_log.flush()
        except Exception:
            pass
        job.broadcast("log", cancel_msg)

        if job.status == "queued":
            job.status = "canceled"
            job.finished_at = datetime.now(timezone.utc).isoformat()
            job.save_meta()
            job.broadcast("status", {"status": "canceled"})
            return True

        job.cancel_requested = True
        # Do not mark canceled early: the dataset remains busy until its child
        # exits.  Once it has exited, expose the terminal state immediately;
        # the worker will subsequently finish draining and persist its log.
        await self._stop_process(job)
        if job.proc is not None and job.proc.returncode is not None:
            await self._finalize_job(job, exit_code=job.proc.returncode, force_status="canceled")
        return True

    async def shutdown(self) -> None:
        """Stop accepting work and reclaim the worker and active child process."""
        self._shutdown = True
        for job in self.jobs.values():
            if job.status == "queued":
                job.status = "canceled"
                job.finished_at = datetime.now(timezone.utc).isoformat()
                job.save_meta()
                job.broadcast("status", {"status": "canceled"})

        for job_id in list(self.running_job_ids):
            running_job = self.jobs.get(job_id)
            if running_job and running_job.status == "running":
                running_job.cancel_requested = True
                await self._stop_process(running_job)

        for task in list(self.worker_tasks.values()):
            if task and not task.done():
                try:
                    await asyncio.wait_for(asyncio.shield(task), timeout=1.0)
                except asyncio.TimeoutError:
                    pass
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    def delete_job(self, job_id: str) -> bool:
        """删除任务记录及其日志文件目录。正在运行的任务不可直接删除，需先取消。"""
        job = self.jobs.get(job_id)
        if not job:
            # 尝试清理可能遗留的孤立磁盘目录
            target_dir = self.settings.jobs_dir / job_id
            if target_dir.exists():
                shutil.rmtree(target_dir, ignore_errors=True)
                return True
            return False

        # 如果任务正在运行或排队，拒绝删除，要求先终止
        if job.status in ("queued", "running"):
            raise ValueError("正在运行或排队中的任务无法直接删除，请先点击停止")

        # 广播关闭任何存留的 SSE 订阅
        try:
            job.broadcast("status", {"status": "deleted"})
        except Exception:
            pass

        # 移除内存引用
        self.jobs.pop(job_id, None)

        # 清除磁盘文件及目录
        if job.dir.exists():
            try:
                shutil.rmtree(job.dir, ignore_errors=True)
            except Exception:
                pass

        return True

    def list_jobs(self) -> list[JobSummary]:
        summaries: list[JobSummary] = []
        for jid, job in sorted(self.jobs.items(), key=lambda x: x[1].created_at, reverse=True):
            self._reconcile_finished_process(job)
            summaries.append(job.to_summary())
        return summaries

    def get_job(self, job_id: str) -> Optional[Job]:
        job = self.jobs.get(job_id)
        if job:
            self._reconcile_finished_process(job)
        return job

    def _reconcile_finished_process(self, job: Job) -> None:
        """Cheap defensive convergence for callers that observe a stale running job."""
        if job.status != "running" or job.proc is None or job.proc.returncode is None:
            return
        # The active worker owns the brief post-exit stdout drain window and
        # will finalize within its bounded grace period. Avoid closing SSE
        # early and hiding the last buffered log lines from viewers.
        lane = get_job_lane(job.type)
        worker = self.worker_tasks.get(lane)
        if job.id in self.running_job_ids and worker and not worker.done():
            return
        job.exit_code = job.proc.returncode
        job.finished_at = datetime.now(timezone.utc).isoformat()
        job.status = "canceled" if job.cancel_requested else (
            "succeeded" if job.exit_code == 0 else "failed"
        )
        job.progress_msg = job.progress_msg or "进程已退出，日志流正在收尾"
        job.save_meta()
        job.broadcast("status", {"status": job.status, "exit_code": job.exit_code, "finished_at": job.finished_at})

    async def stream_job_logs(
        self, job_id: str, offset: int = 0
    ) -> AsyncGenerator[str, None]:
        """SSE 事件流生成器。"""
        job = self.jobs.get(job_id)
        if not job:
            yield f"event: error\ndata: {json.dumps({'message': 'Job not found'})}\n\n"
            return

        # Subscribe first: status can transition while disk history is being
        # replayed.  The queue closes that race (a duplicate tail is harmless).
        q: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        job.subscribers.append(q)
        self._reconcile_finished_process(job)
        try:
            # 1. 首次连线：回放历史日志
            if job.log_file.exists():
                with job.log_file.open("r", encoding="utf-8", errors="replace") as f:
                    if offset > 0:
                        f.seek(offset)
                    content = f.read()
                    if content:
                        yield f"event: log\ndata: {json.dumps(content)}\n\n"
            elif job.status == "queued":
                cmd_preview = " ".join(job.command) if job.command else "—"
                queued_msg = f"=== 任务已进入调度队列等待执行 (ID: {job.id}) ===\n=== 预备执行: {cmd_preview} ===\n\n"
                yield f"event: log\ndata: {json.dumps(queued_msg)}\n\n"

            # 推送当前状态. The subscriber is already attached, so a terminal
            # transition cannot be lost between this snapshot and waiting.
            yield f"event: status\ndata: {json.dumps({'status': job.status, 'exit_code': job.exit_code})}\n\n"

            if job.status not in ("queued", "running"):
                return

            while True:
                try:
                    msg = await asyncio.wait_for(q.get(), timeout=15.0)
                except asyncio.TimeoutError:
                    yield ": ping\n\n"
                    continue
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
