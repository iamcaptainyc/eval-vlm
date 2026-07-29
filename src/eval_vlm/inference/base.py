"""推理后端抽象基类。"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from ..config import Config
from ..data.schema import Prediction, Turn


class InferenceBackend(ABC):
    """所有推理后端的统一接口。

    runner 只依赖这个接口,不关心底层是 OpenAI API 还是本地回显,
    从而让"执行测试"与"如何调用模型"解耦。

    多轮 rollout 下,runner 会**逐轮**调用 complete:每次传入截至当前待预测轮
    之前的对话上下文(context),后端据此生成**一个** assistant 轮的回答。
    """

    # 后端是否可在多线程下并发调用。OpenAI/fake 走 HTTP/纯函数,可并发(True);
    # 本地有状态后端(如 MNN:单个 Llm 对象 + KV cache)必须串行(置 False),
    # 编排层(predict_folder / run_inference)据此把并发降为 1。
    thread_safe: bool = True

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    @abstractmethod
    def complete(
        self,
        context: list[Turn],
        images: list[str],
        sample_id: str,
        expected: Optional[str] = None,
    ) -> Prediction:
        """根据对话上下文生成下一个 assistant 轮。

        context  — 截至待预测轮之前的对话(含已填入的历史 assistant 轮)。
        images   — 该样本引用的图片(按 context 中 <image> 出现顺序消费)。
        expected — 该轮的标准答案;仅 fake 后端用于回显,真实后端忽略。

        实现方应捕获自身异常并写入 Prediction.error,而不是抛出,
        以免中断整批评测。返回的 Prediction 不需要设置 turn,由 runner 填。

        例外:在 except 块里先调 self._raise_if_fail_fast() —— fail-fast 模式下它会
        原样抛出当前异常(带完整 traceback),否则直接返回、照常记 error 继续跑。
        """
        raise NotImplementedError

    def _raise_if_fail_fast(self) -> None:
        """fail-fast 模式:重新抛出**正在处理中的**异常,让完整 traceback 冒泡到 CLI。

        必须在 except 块内(或其调用链上)调用:无参 `raise` 会重新抛出当前活跃异常,
        且保留其原始 traceback(指向真正出错的那一行,如模型内部的 reshape)。
        默认(非 fail-fast)是 no-op —— 由调用方接着记 error、不中断整批。
        """
        if self.cfg.inference.fail_fast:
            raise  # re-raise active exception, original traceback intact

    def close(self) -> None:
        """可选:释放资源(连接池等)。"""
