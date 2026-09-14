"""FastAPI 应用工厂与路由注册。"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import json
from pathlib import Path
from typing import Any, Optional

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response, status
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from ..config import Config
from ..workspace import global_config_path, scan_local_models, set_global_value
from .automation import check_dataset_health, compare_runs
from .configio import read_config_info, update_dataset_config
from .datasets import get_samples_page, list_datasets, serve_image
from .deps import get_dataset_cfg
from .editing import delete_sample, list_trash, restore_sample
from .jobs import JobManager, get_job_manager
from .locks import dataset_lock
from .models import (
    ConfigUpdateRequest,
    DeleteSampleRequest,
    DeleteSampleResponse,
    GlobalSplitConfig,
    GGUFConvertRequest,
    JobCreateRequest,
    JobSummary,
    RestoreRequest,
    RestoreResponse,
    SamplesResponse,
    SettingsResponse,
    SettingsUpdateRequest,
)
from .runsio import (
    get_field_metrics_detail,
    get_field_mismatches_records,
    get_metrics_detail,
    get_scored_records,
    list_dataset_html_files,
    list_runs,
    serve_dataset_html,
    serve_failures_html,
    serve_field_mismatches_html,
)
from .settings import Settings, get_settings, set_settings


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    if settings is None:
        settings = get_settings()
    else:
        set_settings(settings)

    job_manager = get_job_manager(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        loop = asyncio.get_running_loop()
        job_manager.start_worker(loop=loop)
        try:
            yield
        finally:
            await job_manager.shutdown()

    app = FastAPI(
        title="eval_vlm Web UI",
        description="VLM 测试集评测与可视化审查平台",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.dependency_overrides[get_settings] = lambda: settings

    app.add_middleware(GZipMiddleware, minimum_size=800)

    # HTML is the entry point and must be revalidated; fingerprinted assets
    # can be retained indefinitely. API and streaming responses are untouched.
    @app.middleware("http")
    async def static_cache_middleware(request: Request, call_next):
        response = await call_next(request)
        path = request.url.path
        if path in ("/", "/index.html"):
            response.headers["Cache-Control"] = "no-cache, max-age=0, must-revalidate"
        elif path in ("/app.js", "/styles.css") and request.query_params.get("v"):
            response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        return response

    # -----------------------------------------------------------------------
    # 全局设置与模型管理
    # -----------------------------------------------------------------------
    @app.get("/api/settings")
    def api_get_settings(
        st: Settings = Depends(get_settings),
    ) -> SettingsResponse:
        st.reload_global_config()
        cfg_path = global_config_path()
        raw_text = cfg_path.read_text(encoding="utf-8") if cfg_path.exists() else ""
        split_dict = st.split if isinstance(st.split, dict) else {}
        split_obj = GlobalSplitConfig(
            train=float(split_dict.get("train", 0.95)),
            test=float(split_dict.get("test", 0.05)),
            val=float(split_dict.get("val", 0.0)),
            seed=int(split_dict.get("seed", 42)),
            stratify_by=split_dict.get("stratify_by") or None,
        )
        return SettingsResponse(
            workspace=str(st.workspace),
            media_root=st.media_root,
            image_strip_prefix=st.image_strip_prefix,
            hf_models_dir=st.hf_models_dir,
            mnn_models_dir=st.mnn_models_dir,
            llamacpp_models_dir=st.llamacpp_models_dir,
            train_out_dir=st.train_out_dir,
            val_out_dir=st.val_out_dir,
            test_out_dir=st.test_out_dir,
            split=split_obj,
            config_file=str(cfg_path),
            raw_yaml=raw_text,
        )

    @app.put("/api/settings")
    def api_update_settings(
        body: SettingsUpdateRequest,
        st: Settings = Depends(get_settings),
    ) -> SettingsResponse:
        fields = {
            "media_root": body.media_root,
            "image_strip_prefix": body.image_strip_prefix,
            "hf_models_dir": body.hf_models_dir,
            "mnn_models_dir": body.mnn_models_dir,
            "llamacpp_models_dir": body.llamacpp_models_dir,
            "train_out_dir": body.train_out_dir,
            "val_out_dir": body.val_out_dir,
            "test_out_dir": body.test_out_dir,
        }
        if body.workspace is not None and body.workspace.strip():
            fields["workspace"] = body.workspace.strip()

        for k, v in fields.items():
            if v is not None:
                if isinstance(v, list):
                    clean_list = [str(x).strip() for x in v if str(x).strip()]
                    val_to_set = clean_list if clean_list else None
                elif isinstance(v, str):
                    s = v.strip()
                    if s.lower() in ("", "null", "none"):
                        val_to_set = None
                    elif "\n" in s:
                        lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
                        val_to_set = lines if len(lines) > 1 else (lines[0] if lines else None)
                    else:
                        val_to_set = s
                else:
                    val_to_set = v
                set_global_value(k, val_to_set)

        if body.split and isinstance(body.split, dict):
            for sk, sv in body.split.items():
                if sv is not None:
                    sv_str = None if (isinstance(sv, str) and sv.strip().lower() in ("", "null", "none")) else str(sv).strip()
                    set_global_value(f"split.{sk}", sv_str)

        st.reload_global_config()
        cfg_path = global_config_path()
        raw_text = cfg_path.read_text(encoding="utf-8") if cfg_path.exists() else ""
        split_dict = st.split if isinstance(st.split, dict) else {}
        split_obj = GlobalSplitConfig(
            train=float(split_dict.get("train", 0.95)),
            test=float(split_dict.get("test", 0.05)),
            val=float(split_dict.get("val", 0.0)),
            seed=int(split_dict.get("seed", 42)),
            stratify_by=split_dict.get("stratify_by") or None,
        )
        return SettingsResponse(
            workspace=str(st.workspace),
            media_root=st.media_root,
            image_strip_prefix=st.image_strip_prefix,
            hf_models_dir=st.hf_models_dir,
            mnn_models_dir=st.mnn_models_dir,
            llamacpp_models_dir=st.llamacpp_models_dir,
            train_out_dir=st.train_out_dir,
            val_out_dir=st.val_out_dir,
            test_out_dir=st.test_out_dir,
            split=split_obj,
            config_file=str(cfg_path),
            raw_yaml=raw_text,
        )

    @app.get("/api/models")
    def api_get_models(
        st: Settings = Depends(get_settings),
    ) -> dict[str, Any]:
        st.reload_global_config()
        result = scan_local_models(
            hf_dir=st.hf_models_dir,
            mnn_dir=st.mnn_models_dir,
            llamacpp_dir=st.llamacpp_models_dir,
        )
        return {
            "hf_dir": st.hf_models_dir,
            "mnn_dir": st.mnn_models_dir,
            "llamacpp_dir": st.llamacpp_models_dir,
            "hf_models": result.get("hf_models", []),
            "mnn_models": result.get("mnn_models", []),
            "llamacpp_models": result.get("llamacpp_models", []),
        }

    @app.post("/api/tools/convert-gguf")
    async def api_convert_gguf(
        body: GGUFConvertRequest,
    ) -> JobSummary:
        """提交一个异步 HF 转 GGUF 任务。"""
        job_manager.start_worker(asyncio.get_running_loop())
        params = {
            "hf_path": body.hf_path,
        }
        if body.name:
            params["name"] = body.name
        if body.out_dir:
            params["out_dir"] = body.out_dir
        if body.outtype:
            params["outtype"] = body.outtype
        if body.is_multimodal:
            params["mmproj"] = True
        else:
            params["no_mmproj"] = True
        if body.mmproj_outtype:
            params["mmproj_outtype"] = body.mmproj_outtype
        if body.mmproj_type:
            params["mmproj_type"] = body.mmproj_type
        if body.quantize:
            params["quantize"] = body.quantize
        if body.clean_intermediate:
            params["clean_intermediate"] = True
        if body.llama_cpp_dir:
            params["llama_cpp_dir"] = body.llama_cpp_dir

        return job_manager.submit_job(
            job_type="convert-gguf",
            dataset=None,
            params=params,
            user="local",
        )

    # -----------------------------------------------------------------------
    # 数据集
    # -----------------------------------------------------------------------
    @app.get("/api/datasets")
    def api_list_datasets(
        st: Settings = Depends(get_settings),
    ) -> list[dict[str, Any]]:
        return [ds.model_dump() for ds in list_datasets(st)]

    @app.get("/api/datasets/{name}")
    def api_get_dataset(
        cfg: Config = Depends(get_dataset_cfg),
    ) -> dict[str, Any]:
        from .locks import calc_file_sha256
        from ..data.splitter import load_split_meta

        runs = list_runs(cfg)
        return {
            "name": cfg.dataset_dir.name,
            "path": str(cfg.dataset_dir),
            "test_sha256": calc_file_sha256(cfg.test_path),
            "split_meta": load_split_meta(cfg),
            "runs": [r.model_dump() for r in runs],
            "health": check_dataset_health(cfg),
        }

    @app.get("/api/datasets/{name}/samples")
    def api_get_samples(
        offset: int = 0,
        limit: int = 50,
        filter: Optional[str] = Query(None, description="搜索关键词"),
        cfg: Config = Depends(get_dataset_cfg),
    ) -> SamplesResponse:
        return get_samples_page(cfg, offset=offset, limit=limit, filter_text=filter)

    @app.get("/api/datasets/{name}/image")
    def api_get_image(
        ref: str = Query(..., description="图片引用或路径"),
        thumb: int = Query(0, description="是否返回缩略图 (1=是)"),
        cfg: Config = Depends(get_dataset_cfg),
    ) -> Response:
        return serve_image(cfg, ref=ref, thumb=bool(thumb))

    # -----------------------------------------------------------------------
    # 配置
    # -----------------------------------------------------------------------
    @app.get("/api/datasets/{name}/config")
    def api_get_config(
        cfg: Config = Depends(get_dataset_cfg),
    ) -> dict[str, Any]:
        return read_config_info(cfg)

    @app.put("/api/datasets/{name}/config")
    async def api_put_config(
        body: ConfigUpdateRequest,
        cfg: Config = Depends(get_dataset_cfg),
        st: Settings = Depends(get_settings),
    ) -> dict[str, Any]:
        return await update_dataset_config(cfg, st, body.updates, user="local")

    # -----------------------------------------------------------------------
    # 样本删除与回收站 (核心)
    # -----------------------------------------------------------------------
    @app.delete("/api/datasets/{name}/samples/{sample_id}")
    async def api_delete_sample(
        name: str,
        sample_id: str,
        body: DeleteSampleRequest,
        cfg: Config = Depends(get_dataset_cfg),
        st: Settings = Depends(get_settings),
    ) -> DeleteSampleResponse:
        if job_manager.is_dataset_busy(name):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"数据集 '{name}' 当前有正在运行的任务，禁止删除样本以防污染结果",
            )
        async with dataset_lock(name, st):
            return delete_sample(
                cfg=cfg,
                settings=st,
                sample_id=sample_id,
                expected_sha256=body.expected_sha256,
                user="local",
                reason=body.reason,
                mode=body.mode,
                image_index=body.image_index,
            )

    @app.get("/api/datasets/{name}/trash")
    def api_list_trash(
        cfg: Config = Depends(get_dataset_cfg),
        st: Settings = Depends(get_settings),
    ) -> list[dict[str, Any]]:
        return list_trash(cfg, st)

    @app.post("/api/datasets/{name}/trash/{trash_id}/restore")
    async def api_restore_sample(
        name: str,
        trash_id: str,
        body: RestoreRequest,
        cfg: Config = Depends(get_dataset_cfg),
        st: Settings = Depends(get_settings),
    ) -> RestoreResponse:
        if job_manager.is_dataset_busy(name):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"数据集 '{name}' 当前有正在运行的任务，禁止恢复样本",
            )
        async with dataset_lock(name, st):
            return restore_sample(
                cfg=cfg,
                settings=st,
                trash_id=trash_id,
                expected_sha256=body.expected_sha256,
                user="local",
            )

    # -----------------------------------------------------------------------
    # 运行结果与分析
    # -----------------------------------------------------------------------
    @app.get("/api/datasets/{name}/runs")
    def api_get_runs(
        cfg: Config = Depends(get_dataset_cfg),
    ) -> list[dict[str, Any]]:
        return [r.model_dump() for r in list_runs(cfg)]

    @app.get("/api/datasets/{name}/runs/{model}/{backend}/metrics")
    def api_get_run_metrics(
        model: str,
        backend: str,
        cfg: Config = Depends(get_dataset_cfg),
    ) -> dict[str, Any]:
        return get_metrics_detail(cfg, model, backend)

    @app.get("/api/datasets/{name}/runs/{model}/{backend}/scored")
    def api_get_run_scored(
        model: str,
        backend: str,
        offset: int = 0,
        limit: int = 50,
        min_score: Optional[float] = None,
        max_score: Optional[float] = None,
        order: str = "default",
        cfg: Config = Depends(get_dataset_cfg),
    ) -> dict[str, Any]:
        return get_scored_records(
            cfg,
            model,
            backend,
            offset=offset,
            limit=limit,
            min_score=min_score,
            max_score=max_score,
            order_by=order,
        )

    @app.get("/api/datasets/{name}/runs/{model}/{backend}/failures.html")
    @app.get("/api/datasets/{name}/runs/{model}/{backend}/failure.html")
    def api_get_run_failures_html(
        model: str,
        backend: str,
        cfg: Config = Depends(get_dataset_cfg),
    ) -> FileResponse:
        return serve_failures_html(cfg, model, backend)

    @app.get("/api/datasets/{name}/html-files")
    def api_list_dataset_html_files(
        cfg: Config = Depends(get_dataset_cfg),
    ) -> list[dict[str, Any]]:
        return list_dataset_html_files(cfg)

    @app.get("/api/datasets/{name}/html-view")
    def api_serve_dataset_html_view(
        path: str = Query(..., description="相对数据集目录的 HTML 报告路径"),
        cfg: Config = Depends(get_dataset_cfg),
    ) -> FileResponse:
        return serve_dataset_html(cfg, path)

    @app.get("/api/datasets/{name}/failures.html")
    @app.get("/api/datasets/{name}/failure.html")
    def api_serve_dataset_root_failures_html(
        cfg: Config = Depends(get_dataset_cfg),
    ) -> FileResponse:
        return serve_dataset_html(cfg, "failures.html")

    @app.get("/api/datasets/{name}/runs/{model}/{backend}/field-metrics")
    def api_get_run_field_metrics(
        model: str,
        backend: str,
        cfg: Config = Depends(get_dataset_cfg),
    ) -> dict[str, Any]:
        return get_field_metrics_detail(cfg, model, backend)

    @app.get("/api/datasets/{name}/runs/{model}/{backend}/field-mismatches")
    def api_get_run_field_mismatches(
        model: str,
        backend: str,
        offset: int = 0,
        limit: int = 50,
        filter_state: Optional[str] = None,
        cfg: Config = Depends(get_dataset_cfg),
    ) -> dict[str, Any]:
        return get_field_mismatches_records(
            cfg,
            model,
            backend,
            offset=offset,
            limit=limit,
            filter_state=filter_state,
        )

    @app.get("/api/datasets/{name}/runs/{model}/{backend}/field-mismatches.html")
    @app.get("/api/datasets/{name}/runs/{model}/{backend}/field-mismatch.html")
    def api_get_run_field_mismatches_html(
        model: str,
        backend: str,
        cfg: Config = Depends(get_dataset_cfg),
    ) -> FileResponse:
        return serve_field_mismatches_html(cfg, model, backend)

    # -----------------------------------------------------------------------
    # 自动化与对比
    # -----------------------------------------------------------------------
    @app.get("/api/datasets/{name}/health")
    def api_dataset_health(
        cfg: Config = Depends(get_dataset_cfg),
    ) -> dict[str, Any]:
        return check_dataset_health(cfg)

    @app.get("/api/datasets/{name}/diff")
    def api_run_diff(
        a: str = Query(..., description="Run A, 如 'Qwen_Qwen2-VL/openai'"),
        b: str = Query(..., description="Run B, 如 'qwen-mnn/mnn'"),
        cfg: Config = Depends(get_dataset_cfg),
    ) -> dict[str, Any]:
        return compare_runs(cfg, a, b)

    # -----------------------------------------------------------------------
    # 任务执行 (Jobs)
    # -----------------------------------------------------------------------
    @app.post("/api/datasets/{name}/jobs")
    async def api_create_dataset_job(
        name: str,
        body: JobCreateRequest,
    ) -> JobSummary:
        job_manager.start_worker(asyncio.get_running_loop())
        return job_manager.submit_job(
            job_type=body.type,
            dataset=name,
            params=body.params or {},
            user="local",
        )

    @app.post("/api/sweep/jobs")
    async def api_create_sweep_job(
        body: JobCreateRequest,
    ) -> JobSummary:
        job_manager.start_worker(asyncio.get_running_loop())
        return job_manager.submit_job(
            job_type="sweep",
            dataset=body.dataset,
            params=body.params or {},
            user="local",
        )

    @app.post("/api/jobs")
    async def api_create_job(
        body: JobCreateRequest,
    ) -> JobSummary:
        job_manager.start_worker(asyncio.get_running_loop())
        return job_manager.submit_job(
            job_type=body.type,
            dataset=body.dataset,
            params=body.params or {},
            user="local",
        )

    @app.get("/api/jobs")
    async def api_list_jobs() -> list[JobSummary]:
        return job_manager.list_jobs()

    @app.get("/api/jobs/{job_id}")
    async def api_get_job(
        job_id: str,
    ) -> JobSummary:
        job = job_manager.get_job(job_id)
        if not job:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="任务不存在")
        return job.to_summary()

    @app.get("/api/jobs/{job_id}/stream")
    async def api_job_stream(
        job_id: str,
        offset: int = 0,
    ) -> StreamingResponse:
        job = job_manager.get_job(job_id)
        if not job:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="任务不存在")
        return StreamingResponse(
            job_manager.stream_job_logs(job_id, offset=offset),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.post("/api/jobs/{job_id}/cancel")
    async def api_cancel_job(
        job_id: str,
    ) -> dict[str, Any]:
        ok = await job_manager.cancel_job(job_id)
        return {"success": ok, "job_id": job_id}

    @app.delete("/api/jobs/{job_id}")
    async def api_delete_job(
        job_id: str,
    ) -> dict[str, Any]:
        """删除指定任务记录及日志文件。"""
        try:
            ok = job_manager.delete_job(job_id)
            if not ok:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="任务不存在或已被删除")
            return {"success": True, "job_id": job_id}
        except ValueError as e:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    @app.delete("/api/jobs")
    async def api_clear_finished_jobs(
    ) -> dict[str, Any]:
        """批量清理所有已结束（非运行、非排队）的历史任务。"""
        deleted_ids: list[str] = []
        for job in list(job_manager.jobs.values()):
            if job.status not in ("queued", "running"):
                try:
                    if job_manager.delete_job(job.id):
                        deleted_ids.append(job.id)
                except Exception:
                    pass
        return {"success": True, "deleted_count": len(deleted_ids), "deleted_ids": deleted_ids}

    @app.post("/api/jobs/{job_id}/resume")
    async def api_resume_job(
        job_id: str,
    ) -> JobSummary:
        job_manager.start_worker(asyncio.get_running_loop())
        old_job = job_manager.get_job(job_id)
        if not old_job:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="任务不存在")
        return job_manager.submit_job(
            job_type=old_job.type,
            dataset=old_job.dataset,
            params=old_job.params,
            user="local",
        )

    # -----------------------------------------------------------------------
    # 静态前端资源挂载
    # -----------------------------------------------------------------------
    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")

    return app
