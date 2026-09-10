"""FastAPI 应用工厂与路由注册。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from ..config import Config
from ..workspace import global_config_path, scan_local_models, set_global_value
from .auth import User, get_current_user, require_editor, require_viewer
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
    JobCreateRequest,
    JobSummary,
    RestoreRequest,
    RestoreResponse,
    SamplesResponse,
    SettingsResponse,
    SettingsUpdateRequest,
)
from .runsio import get_field_metrics_detail, get_field_mismatches_records, get_metrics_detail, get_scored_records, list_runs, serve_failures_html, serve_field_mismatches_html
from .settings import Settings, get_settings, set_settings


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    if settings is None:
        settings = get_settings()
    else:
        set_settings(settings)

    app = FastAPI(
        title="eval_vlm Web UI",
        description="VLM 测试集评测与可视化审查平台",
        version="0.1.0",
    )
    app.dependency_overrides[get_settings] = lambda: settings

    # 跨域配置 (本地同源亦保留兼容性)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 禁用静态资源浏览器过激缓存，确保代码更新即刻生效
    @app.middleware("http")
    async def no_cache_static_middleware(request: Request, call_next):
        response = await call_next(request)
        path = request.url.path
        if path in ("/", "/index.html", "/app.js", "/styles.css") or path.endswith((".js", ".css", ".html")):
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response

    job_manager = get_job_manager(settings)

    # -----------------------------------------------------------------------
    # 鉴权与当前用户
    # -----------------------------------------------------------------------
    @app.get("/api/whoami")
    def whoami(user: User = Depends(get_current_user)) -> dict[str, Any]:
        return {"username": user.username, "role": user.role}

    # -----------------------------------------------------------------------
    # 全局设置与模型管理
    # -----------------------------------------------------------------------
    @app.get("/api/settings")
    def api_get_settings(
        _user: User = Depends(require_viewer),
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
        _user: User = Depends(require_editor),
        st: Settings = Depends(get_settings),
    ) -> SettingsResponse:
        fields = {
            "media_root": body.media_root,
            "image_strip_prefix": body.image_strip_prefix,
            "hf_models_dir": body.hf_models_dir,
            "mnn_models_dir": body.mnn_models_dir,
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
            train_out_dir=st.train_out_dir,
            val_out_dir=st.val_out_dir,
            test_out_dir=st.test_out_dir,
            split=split_obj,
            config_file=str(cfg_path),
            raw_yaml=raw_text,
        )

    @app.get("/api/models")
    def api_get_models(
        _user: User = Depends(require_viewer),
        st: Settings = Depends(get_settings),
    ) -> dict[str, Any]:
        st.reload_global_config()
        result = scan_local_models(hf_dir=st.hf_models_dir, mnn_dir=st.mnn_models_dir)
        return {
            "hf_dir": st.hf_models_dir,
            "mnn_dir": st.mnn_models_dir,
            "hf_models": result.get("hf_models", []),
            "mnn_models": result.get("mnn_models", []),
        }

    # -----------------------------------------------------------------------
    # 数据集
    # -----------------------------------------------------------------------
    @app.get("/api/datasets")
    def api_list_datasets(
        _user: User = Depends(require_viewer),
        st: Settings = Depends(get_settings),
    ) -> list[dict[str, Any]]:
        return [ds.model_dump() for ds in list_datasets(st)]

    @app.get("/api/datasets/{name}")
    def api_get_dataset(
        cfg: Config = Depends(get_dataset_cfg),
        _user: User = Depends(require_viewer),
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
        _user: User = Depends(require_viewer),
    ) -> SamplesResponse:
        return get_samples_page(cfg, offset=offset, limit=limit, filter_text=filter)

    @app.get("/api/datasets/{name}/image")
    def api_get_image(
        ref: str = Query(..., description="图片引用或路径"),
        thumb: int = Query(0, description="是否返回缩略图 (1=是)"),
        cfg: Config = Depends(get_dataset_cfg),
        _user: User = Depends(require_viewer),
    ) -> Response:
        return serve_image(cfg, ref=ref, thumb=bool(thumb))

    # -----------------------------------------------------------------------
    # 配置
    # -----------------------------------------------------------------------
    @app.get("/api/datasets/{name}/config")
    def api_get_config(
        cfg: Config = Depends(get_dataset_cfg),
        _user: User = Depends(require_viewer),
    ) -> dict[str, Any]:
        return read_config_info(cfg)

    @app.put("/api/datasets/{name}/config")
    async def api_put_config(
        body: ConfigUpdateRequest,
        cfg: Config = Depends(get_dataset_cfg),
        user: User = Depends(require_editor),
        st: Settings = Depends(get_settings),
    ) -> dict[str, Any]:
        return await update_dataset_config(cfg, st, body.updates, user=user.username)

    # -----------------------------------------------------------------------
    # 样本删除与回收站 (核心)
    # -----------------------------------------------------------------------
    @app.delete("/api/datasets/{name}/samples/{sample_id}")
    async def api_delete_sample(
        name: str,
        sample_id: str,
        body: DeleteSampleRequest,
        cfg: Config = Depends(get_dataset_cfg),
        user: User = Depends(require_editor),
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
                user=user.username,
                reason=body.reason,
                mode=body.mode,
                image_index=body.image_index,
            )

    @app.get("/api/datasets/{name}/trash")
    def api_list_trash(
        cfg: Config = Depends(get_dataset_cfg),
        _user: User = Depends(require_viewer),
        st: Settings = Depends(get_settings),
    ) -> list[dict[str, Any]]:
        return list_trash(cfg, st)

    @app.post("/api/datasets/{name}/trash/{trash_id}/restore")
    async def api_restore_sample(
        name: str,
        trash_id: str,
        body: RestoreRequest,
        cfg: Config = Depends(get_dataset_cfg),
        user: User = Depends(require_editor),
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
                user=user.username,
            )

    # -----------------------------------------------------------------------
    # 运行结果与分析
    # -----------------------------------------------------------------------
    @app.get("/api/datasets/{name}/runs")
    def api_get_runs(
        cfg: Config = Depends(get_dataset_cfg),
        _user: User = Depends(require_viewer),
    ) -> list[dict[str, Any]]:
        return [r.model_dump() for r in list_runs(cfg)]

    @app.get("/api/datasets/{name}/runs/{model}/{backend}/metrics")
    def api_get_run_metrics(
        model: str,
        backend: str,
        cfg: Config = Depends(get_dataset_cfg),
        _user: User = Depends(require_viewer),
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
        _user: User = Depends(require_viewer),
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
    def api_get_run_failures_html(
        model: str,
        backend: str,
        cfg: Config = Depends(get_dataset_cfg),
        _user: User = Depends(require_viewer),
    ) -> FileResponse:
        return serve_failures_html(cfg, model, backend)

    @app.get("/api/datasets/{name}/runs/{model}/{backend}/field-metrics")
    def api_get_run_field_metrics(
        model: str,
        backend: str,
        cfg: Config = Depends(get_dataset_cfg),
        _user: User = Depends(require_viewer),
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
        _user: User = Depends(require_viewer),
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
    def api_get_run_field_mismatches_html(
        model: str,
        backend: str,
        cfg: Config = Depends(get_dataset_cfg),
        _user: User = Depends(require_viewer),
    ) -> FileResponse:
        return serve_field_mismatches_html(cfg, model, backend)

    # -----------------------------------------------------------------------
    # 自动化与对比
    # -----------------------------------------------------------------------
    @app.get("/api/datasets/{name}/health")
    def api_dataset_health(
        cfg: Config = Depends(get_dataset_cfg),
        _user: User = Depends(require_viewer),
    ) -> dict[str, Any]:
        return check_dataset_health(cfg)

    @app.get("/api/datasets/{name}/diff")
    def api_run_diff(
        a: str = Query(..., description="Run A, 如 'Qwen_Qwen2-VL/openai'"),
        b: str = Query(..., description="Run B, 如 'qwen-mnn/mnn'"),
        cfg: Config = Depends(get_dataset_cfg),
        _user: User = Depends(require_viewer),
    ) -> dict[str, Any]:
        return compare_runs(cfg, a, b)

    # -----------------------------------------------------------------------
    # 任务执行 (Jobs)
    # -----------------------------------------------------------------------
    @app.post("/api/datasets/{name}/jobs")
    def api_create_dataset_job(
        name: str,
        body: JobCreateRequest,
        user: User = Depends(require_editor),
    ) -> JobSummary:
        return job_manager.submit_job(
            job_type=body.type,
            dataset=name,
            params=body.params or {},
            user=user.username,
        )

    @app.post("/api/sweep/jobs")
    def api_create_sweep_job(
        body: JobCreateRequest,
        user: User = Depends(require_editor),
    ) -> JobSummary:
        return job_manager.submit_job(
            job_type="sweep",
            dataset=body.dataset,
            params=body.params or {},
            user=user.username,
        )

    @app.post("/api/jobs")
    def api_create_job(
        body: JobCreateRequest,
        user: User = Depends(require_editor),
    ) -> JobSummary:
        return job_manager.submit_job(
            job_type=body.type,
            dataset=body.dataset,
            params=body.params or {},
            user=user.username,
        )

    @app.get("/api/jobs")
    def api_list_jobs(_user: User = Depends(require_viewer)) -> list[JobSummary]:
        return job_manager.list_jobs()

    @app.get("/api/jobs/{job_id}")
    def api_get_job(
        job_id: str,
        _user: User = Depends(require_viewer),
    ) -> JobSummary:
        job = job_manager.get_job(job_id)
        if not job:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="任务不存在")
        return job.to_summary()

    @app.get("/api/jobs/{job_id}/stream")
    async def api_job_stream(
        job_id: str,
        offset: int = 0,
        _user: User = Depends(require_viewer),
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
    def api_cancel_job(
        job_id: str,
        _user: User = Depends(require_editor),
    ) -> dict[str, Any]:
        ok = job_manager.cancel_job(job_id)
        return {"success": ok, "job_id": job_id}

    @app.post("/api/jobs/{job_id}/resume")
    def api_resume_job(
        job_id: str,
        user: User = Depends(require_editor),
    ) -> JobSummary:
        old_job = job_manager.get_job(job_id)
        if not old_job:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="任务不存在")
        return job_manager.submit_job(
            job_type=old_job.type,
            dataset=old_job.dataset,
            params=old_job.params,
            user=user.username,
        )

    # -----------------------------------------------------------------------
    # 静态前端资源挂载
    # -----------------------------------------------------------------------
    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")

    return app
