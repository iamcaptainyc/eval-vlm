"""测试 / dry-run 用的回显后端,不联网。

策略:直接回显该轮的标准答案 expected(若有),否则回显最后一个 user 文本。
这样跑 dry-run 时各轮 exact_match / token_f1 都会接近满分,便于验证全链路打通。
"""
from __future__ import annotations

from typing import Optional

from ..data.schema import Prediction, Turn
from .base import InferenceBackend


class FakeBackend(InferenceBackend):
    def complete(
        self,
        context: list[Turn],
        images: list[str],
        sample_id: str,
        expected: Optional[str] = None,
    ) -> Prediction:
        if expected is not None:
            text = expected
        else:
            user_turns = [t.content for t in context if t.role == "user"]
            text = (user_turns[-1] if user_turns else "").replace("<image>", "").strip()
        return Prediction(id=sample_id, prediction=text, latency=0.0,
                          raw={"backend": "fake"})
