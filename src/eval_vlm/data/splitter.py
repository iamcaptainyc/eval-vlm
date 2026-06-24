"""把源 LlamaFactory JSON 纯分割成 train / test(val 可选)。

设计要点:
- 只做"分割",不做解析:每条记录**原样**写入对应文件(答案、对话结构、图片路径全不动),
  因此 train.json / val.json / test.json 都是**合法的 LlamaFactory 数据集**,
  train.json 可直接拿去 LlamaFactory 训练,test.json 在 run/score 阶段按 LlamaFactory 格式读取。
- 确定性:固定 seed,同配置同结果;可按某字段分层抽样。
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Optional

from ..config import Config
from .loader import load_raw_records, source_hash


def _normalize_ratios(train: float, val: float, test: float) -> dict[str, float]:
    parts = {"train": max(0.0, train), "val": max(0.0, val), "test": max(0.0, test)}
    total = sum(parts.values())
    if total <= 0:
        raise ValueError("split 比例非法:train/val/test 至少有一个 > 0")
    return {k: v / total for k, v in parts.items()}


def _slice_indices(indices: list[int], ratios: dict[str, float],
                   rng: random.Random) -> dict[str, list[int]]:
    """对一组下标按比例三路切分(test 取余数,保证不丢样本)。"""
    idx = indices[:]
    rng.shuffle(idx)
    n = len(idx)
    n_train = round(n * ratios["train"])
    n_val = round(n * ratios["val"])
    # 防越界
    n_train = min(n_train, n)
    n_val = min(n_val, n - n_train)
    train = idx[:n_train]
    val = idx[n_train:n_train + n_val]
    test = idx[n_train + n_val:]
    return {"train": train, "val": val, "test": test}


def _partition(n: int, ratios: dict[str, float], seed: int,
               strata: Optional[list[str]]) -> dict[str, list[int]]:
    rng = random.Random(seed)
    if strata is None:
        parts = _slice_indices(list(range(n)), ratios, rng)
    else:
        groups: dict[str, list[int]] = defaultdict(list)
        for i, key in enumerate(strata):
            groups[key].append(i)
        parts = {"train": [], "val": [], "test": []}
        for key in sorted(groups):
            sub = _slice_indices(groups[key], ratios, rng)
            for split_name in parts:
                parts[split_name].extend(sub[split_name])
    return {k: sorted(v) for k, v in parts.items()}


def split_dataset(cfg: Config) -> dict:
    """执行划分,把 train/test(+val) 以 LlamaFactory 格式写到 dataset_dir,返回 meta。"""
    records: list[Any] = load_raw_records(cfg.source_path)
    n = len(records)
    if n == 0:
        raise ValueError("数据源为空,无法划分")

    ratios = _normalize_ratios(cfg.split.train, cfg.split.val, cfg.split.test)

    stratify_by = cfg.split.stratify_by
    strata: Optional[list[str]] = None
    if stratify_by:
        strata = [str(_get_field(r, stratify_by)) for r in records]

    parts = _partition(n, ratios, cfg.split.seed, strata)

    # split 产物落数据集文件夹本身(各模型共享),不进 <模型> 子目录。
    cfg.dataset_dir.mkdir(parents=True, exist_ok=True)

    written: dict[str, str] = {}
    # train / test 必出;val 仅在比例 > 0 时产出。
    targets = {"train": cfg.train_path, "test": cfg.test_path}
    if ratios["val"] > 0 and parts["val"]:
        targets["val"] = cfg.val_path

    for name, path in targets.items():
        subset = [records[i] for i in parts[name]]   # 原样切片,verbatim
        path.parent.mkdir(parents=True, exist_ok=True)   # 支持自定义 *_out 指向任意目录
        with path.open("w", encoding="utf-8") as f:
            json.dump(subset, f, ensure_ascii=False, indent=2)
        written[name] = str(path)

    meta = {
        "schema_version": 2,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": str(cfg.source_path),
        "source_sha256": source_hash(cfg),
        "media_root": str(cfg.media_root_path),
        "total_samples": n,
        "seed": cfg.split.seed,
        "stratify_by": stratify_by,
        "ratios": ratios,
        "counts": {k: len(v) for k, v in parts.items()},
        "indices": parts,            # 原始源中的下标,便于审计/复现
        "files": written,
        "format": "llamafactory",    # 三份均为原样 LlamaFactory 格式
    }
    with cfg.split_meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return meta


def _get_field(record: Any, key: str) -> Any:
    if isinstance(record, dict):
        return record.get(key, "__none__")
    return "__none__"


def load_split_meta(cfg: Config) -> dict:
    """读取 split_meta.json(若存在),否则返回空 dict。"""
    if not cfg.split_meta_path.exists():
        return {}
    with cfg.split_meta_path.open("r", encoding="utf-8") as f:
        return json.load(f)
