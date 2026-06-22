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


def load_predictions(path: Path) -> list[Prediction]:
    """读取全部预测(取每个 (id, turn) 的最后一条,后写覆盖先写)。"""
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
            pred = Prediction.from_dict(obj)
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
