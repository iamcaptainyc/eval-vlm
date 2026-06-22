"""跨阶段流转的统一数据结构。

Sample 是 loader/runner/scorer 共同认的中间表示:它持有**完整对话**(turns)
以及**要评测的 assistant 轮**(targets)。这样既支持只评最后一轮(标签),
也支持逐轮评测(轮1描述 + 轮2标签),且每个目标轮可单独对齐、单独评分。
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Optional


@dataclass
class Turn:
    """一轮对话(user 或 assistant)。"""
    role: str          # "user" | "assistant"
    content: str       # 文本内容(可能含 <image> 占位符)


@dataclass
class EvalTurn:
    """一个待预测 + 评分的 assistant 轮。

    turn_index — 该轮在 Sample.turns 中的下标。
    reference  — 该轮的标准答案(数据集里这一轮 assistant 的原文)。
    """
    turn_index: int
    reference: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "EvalTurn":
        return cls(turn_index=int(d["turn_index"]), reference=d.get("reference", ""))


@dataclass
class Sample:
    """一条评测样本。

    id      — 稳定标识,贯穿 split/run/score 对齐结果。
    turns   — 完整对话(全部 user + assistant 轮,<image> 占位符原样保留)。
    images  — 该样本引用的图片路径(相对 media_root)。
    targets — 要评测的 assistant 轮列表(按对话顺序)。
    meta    — 附加信息(如 task_type),供分层抽样 / 分组评分用。
    """
    id: str
    turns: list[Turn] = field(default_factory=list)
    images: list[str] = field(default_factory=list)
    targets: list[EvalTurn] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def reference(self) -> Optional[str]:
        """便捷属性:最后一个目标轮的标准答案(兼容单轮 scorer / fake 后端)。"""
        return self.targets[-1].reference if self.targets else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "turns": [asdict(t) for t in self.turns],
            "images": list(self.images),
            "targets": [t.to_dict() for t in self.targets],
            "meta": dict(self.meta),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Sample":
        return cls(
            id=str(d["id"]),
            turns=[Turn(**t) for t in d.get("turns", [])],
            images=list(d.get("images", [])),
            targets=[EvalTurn.from_dict(t) for t in d.get("targets", [])],
            meta=dict(d.get("meta", {})),
        )


@dataclass
class Prediction:
    """单条推理结果。

    turn   — 该预测对应的目标轮下标(Sample.turns 中的 index)。
             多轮 rollout 下,一条样本会产生多条 Prediction,用 (id, turn) 唯一标识。
             -1 表示未指定(单轮兼容)。
    images — 该预测所属样本引用的原始图片地址(数据集里的原样路径/URL)。
             随预测落盘,使每条结果都能追溯回原图,便于人工核查。
    """
    id: str
    turn: int = -1
    prediction: str = ""
    images: list[str] = field(default_factory=list)
    latency: Optional[float] = None
    error: Optional[str] = None
    raw: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Prediction":
        return cls(
            id=str(d["id"]),
            turn=int(d.get("turn", -1)),
            prediction=d.get("prediction", ""),
            images=list(d.get("images", [])),
            latency=d.get("latency"),
            error=d.get("error"),
            raw=d.get("raw"),
        )
