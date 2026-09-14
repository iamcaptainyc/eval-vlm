"""Pydantic 数据模型定义。"""
from __future__ import annotations

from typing import Any, Literal, Optional, Union
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
    # convert-gguf is submitted only by its dedicated endpoint: it has a
    # different, validated request model and must not be a generic CLI escape hatch.
    type: Literal["split", "pred", "score", "eval", "field-eval", "sweep"]
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
    command: list[str] = Field(default_factory=list)
    log_file: Optional[str] = None
    params: dict[str, Any] = Field(default_factory=dict)


class RunSummary(BaseModel):
    model: str
    backend: str
    path: str
    has_eval: bool = False
    has_field_eval: bool = False
    has_metrics: bool = False
    has_scored: bool = False
    has_failures_html: bool = False
    has_field_metrics: bool = False
    has_field_mismatches_html: bool = False
    has_field_mismatches_json: bool = False
    is_stale: bool = False
    stale_reason: Optional[str] = None
    metrics_summary: Optional[dict[str, Any]] = None
    field_metrics_summary: Optional[dict[str, Any]] = None


class GlobalSplitConfig(BaseModel):
    train: float = 0.95
    test: float = 0.05
    val: float = 0.0
    seed: int = 42
    stratify_by: Optional[str] = None


class SettingsResponse(BaseModel):
    workspace: str
    media_root: Optional[str] = None
    image_strip_prefix: Optional[str] = None
    hf_models_dir: Optional[Union[str, list[str]]] = None
    mnn_models_dir: Optional[Union[str, list[str]]] = None
    llamacpp_models_dir: Optional[Union[str, list[str]]] = None
    train_out_dir: Optional[str] = None
    val_out_dir: Optional[str] = None
    test_out_dir: Optional[str] = None
    split: Optional[GlobalSplitConfig] = None
    config_file: Optional[str] = None
    raw_yaml: Optional[str] = None


class SettingsUpdateRequest(BaseModel):
    workspace: Optional[str] = None
    media_root: Optional[str] = None
    image_strip_prefix: Optional[str] = None
    hf_models_dir: Optional[Union[str, list[str]]] = None
    mnn_models_dir: Optional[Union[str, list[str]]] = None
    llamacpp_models_dir: Optional[Union[str, list[str]]] = None
    train_out_dir: Optional[str] = None
    val_out_dir: Optional[str] = None
    test_out_dir: Optional[str] = None
    split: Optional[dict[str, Any]] = None


class GGUFConvertRequest(BaseModel):
    hf_path: str
    name: Optional[str] = None
    out_dir: Optional[str] = None
    outtype: str = "bf16"
    is_multimodal: bool = True
    mmproj_outtype: str = "f16"
    mmproj_type: Optional[str] = None
    quantize: Optional[str] = None
    clean_intermediate: bool = False
    llama_cpp_dir: Optional[str] = None



