"""测试结果的读写。

- predictions.jsonl : 每行一条预测(追加写,支持断点续跑)
- metrics.json      : 聚合指标
- scored.jsonl      : 逐样本得分
- summary.md        : 人类可读摘要
- run_meta.json     : 运行元信息(模型/配置/时间/计数),可复现
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from ..data.schema import Prediction


# ---------------------------------------------------------------------------
# predictions.jsonl
# ---------------------------------------------------------------------------
def load_prediction_ids(path: Path) -> set[str]:
    """读取已成功完成预测的 id 集合(忽略轮维度;保留向后兼容)。"""
    return {sid for sid, _turn in load_prediction_keys(path)}


def load_prediction_keys(path: Path) -> set[tuple[str, int]]:
    """读取已成功完成预测的 (id, turn) 集合(用于多轮断点续跑跳过)。

    只把成功的算作已完成;有 error 的允许重跑。
    """
    done: set[tuple[str, int]] = set()
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("error") is None and "id" in obj:
                done.add((str(obj["id"]), int(obj.get("turn", -1))))
    return done


def _prediction_from_datadir_record(obj: dict) -> Prediction:
    """把 pred --datadir 的 LlamaFactory 对话记录解析成 Prediction。

    datadir 产物是对话格式 {id, images, messages, latency}:模型输出在 messages
    最后一个 assistant 轮里,**无顶层 prediction/turn 字段**。这里取该轮 content 作
    prediction、其下标作 turn。用于 precision --datadir(否则 from_dict 会读成空串,
    使两端都成 "" 而误报 100% 一致)。
    """
    msgs = obj.get("messages") or []
    last_idx, content = -1, ""
    for i, m in enumerate(msgs):
        if isinstance(m, dict) and m.get("role") == "assistant":
            last_idx, content = i, m.get("content", "")
    return Prediction(
        id=str(obj["id"]),
        turn=last_idx,
        prediction=content,
        images=list(obj.get("images", [])),
        latency=obj.get("latency"),
        error=obj.get("error"),
        raw=obj.get("raw"),
    )


def load_predictions(path: Path, *, datadir_format: bool = False) -> list[Prediction]:
    """读取全部预测(取每个 (id, turn) 的最后一条,后写覆盖先写)。

    datadir_format=True 时按 pred --datadir 的对话格式解析(输出在 messages 最后一个
    assistant 轮);默认按 run / pred --dataset 的 Prediction.to_dict 格式解析。
    """
    by_key: dict[tuple[str, int], Prediction] = {}
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            pred = (_prediction_from_datadir_record(obj) if datadir_format
                    else Prediction.from_dict(obj))
            by_key[(pred.id, pred.turn)] = pred
    return list(by_key.values())


class PredictionWriter:
    """追加式 jsonl 写入器,每条 flush,保证中断不丢已写结果。"""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = path.open("a", encoding="utf-8")

    def write(self, pred: Prediction) -> None:
        self._fh.write(json.dumps(pred.to_dict(), ensure_ascii=False) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> "PredictionWriter":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


# ---------------------------------------------------------------------------
# metrics / scored / summary / run_meta
# ---------------------------------------------------------------------------
def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(text)


# ---------------------------------------------------------------------------
# 产物目录发现(跨模型/后端)
# ---------------------------------------------------------------------------
# 数据集级产物(各模型共享),不是「某模型某后端」的运行结果,枚举时要跳过。
_DATASET_LEVEL_FILES = {"config.yaml", "report.md", "report.json"}
# 判定「这是一个真正的运行结果目录」的标志产物(任一存在即算)。
_RUN_MARKER_FILES = ("metrics.json", "precision.json", "run_meta.json", "pred_meta.json")


def discover_run_dirs(dataset_dir: Path) -> list[tuple[str, str, Path]]:
    """枚举 <dataset_dir>/<模型名>/<后端>/ 两级产物目录。

    返回按 (model, backend) 排序的 [(model_name, backend, dir), ...],
    只收含 _RUN_MARKER_FILES 任一(metrics/precision/run_meta/pred_meta)的叶子目录。
    数据集级文件(train/test/val/split_meta/config.yaml/report.*)都在 dataset_dir 顶层,
    不是二级目录,天然不会被纳入。跨格式合并报告(report 命令)据此发现全部已跑格式。
    """
    if not dataset_dir.is_dir():
        return []
    found: list[tuple[str, str, Path]] = []
    for model_dir in sorted(dataset_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        for backend_dir in sorted(model_dir.iterdir()):
            if not backend_dir.is_dir():
                continue
            if any((backend_dir / f).exists() for f in _RUN_MARKER_FILES):
                found.append((model_dir.name, backend_dir.name, backend_dir))
    return found
