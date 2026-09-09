"""Pydantic 数据模型定义。"""
from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field


class DatasetSummary(BaseModel):
    name: str
    path: str
    test_count: int = 0
    train_count: int = 0
    val_count: int = 0
    run_count: int = 0
    test_sha256: Optional[str] = None
    media_root: Optional[str] = None
    source: Optional[str] = None
    has_dirty_runs: bool = False


class ImageRefInfo(BaseModel):
    ref: str
    exists: bool = True
    is_http: bool = False
    url: str


class SampleItem(BaseModel):
    id: str
    position: int
    turns: list[dict[str, Any]]
    images: list[ImageRefInfo]
    targets: list[dict[str, Any]]
    meta: dict[str, Any] = Field(default_factory=dict)


class SamplesResponse(BaseModel):
    total: int
    offset: int
    limit: int
    test_sha256: str
    samples: list[SampleItem]


class DeleteSampleRequest(BaseModel):
    expected_sha256: str
    reason: Optional[str] = None
    mode: str = "record"  # "record" | "image"
    image_index: Optional[int] = None


class DeleteSampleResponse(BaseModel):
    success: bool
    new_sha256: str
    deleted_id: str
    mode: str
    shifted_ids_count: int
    invalidated_runs: list[str] = Field(default_factory=list)


class RestoreRequest(BaseModel):
    expected_sha256: str


class RestoreResponse(BaseModel):
    success: bool
    new_sha256: str
    restored_id: str
    restored_position: int
    invalidated_runs: list[str] = Field(default_factory=list)


class TrashItem(BaseModel):
    trash_id: str
    sample_id: str
    deleted_by: str
    reason: Optional[str] = None
    ts: str
    mode: str
    position: int
    record: Optional[dict[str, Any]] = None


class ConfigUpdateItem(BaseModel):
    key: str
    value: Any


class ConfigUpdateRequest(BaseModel):
    updates: list[ConfigUpdateItem]


class JobCreateRequest(BaseModel):
    type: str  # "split" | "pred" | "eval" | "field-eval" | "sweep"
    dataset: Optional[str] = None
    params: Optional[dict[str, Any]] = Field(default_factory=dict)


class JobSummary(BaseModel):
    id: str
    type: str
    dataset: Optional[str] = None
    user: str = "anonymous"
    status: str  # "queued" | "running" | "succeeded" | "failed" | "canceled" | "interrupted"
    created_at: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    exit_code: Optional[int] = None
    progress: Optional[float] = None
    progress_msg: Optional[str] = None
    queue_position: Optional[int] = None


class RunSummary(BaseModel):
    model: str
    backend: str
    path: str
    has_metrics: bool = False
    has_scored: bool = False
    has_failures_html: bool = False
    is_stale: bool = False
    stale_reason: Optional[str] = None
    metrics_summary: Optional[dict[str, Any]] = None
