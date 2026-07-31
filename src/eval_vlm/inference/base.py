"""推理后端抽象基类。"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from ..config import Config
from ..data.schema import Prediction, Turn


@dataclass
class BatchItem:
    """一次批处理里的单条请求(供 complete_batch 用)。"""
    context: list[Turn]
    images: list[str]
    sample_id: str
    expected: Optional[str] = None


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

    # 后端是否支持**真·批处理**(一次把多条 prompt 交给引擎,如离线 vLLM 的 continuous
    # batching)。为 True 时 runner 走批量路径(按轮次跨样本批处理),忽略线程并发;
    # 后端应覆写 complete_batch。默认 False:runner 走 per-sample 路径。
    supports_batch: bool = False

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

    def complete_batch(self, items: list["BatchItem"]) -> list[Prediction]:
        """一次处理**多条**请求,返回与 items 等长、同序的 Prediction 列表。

        默认实现:逐条调 complete(不做真批处理)。支持批处理的后端(supports_batch=True,
        如离线 vLLM)覆写本方法,一次把所有 prompt 交给引擎做 continuous batching 提速。
        runner 仅在 supports_batch=True 时才调用它。
        """
        return [self.complete(it.context, it.images, it.sample_id, it.expected)
                for it in items]

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
