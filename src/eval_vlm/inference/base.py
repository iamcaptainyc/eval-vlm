"""推理后端抽象基类。"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from ..config import Config
from ..data.schema import Prediction, Turn


@dataclass
class BatchItem:
    """一条待推理请求(complete 的参数打包版),供 complete_batch 批量消费。

    字段与 complete 一一对应:一个 BatchItem = 一次原本会调 complete 的调用。
    编排层(predict_folder)把待处理项打包成 list[BatchItem] 交给支持批量的后端,
    后端可在**原生层并行**处理整批(如 cmnn:N 个 Llm 实例 + 线程池,无 GIL 真并行)。
    """
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

    # 后端是否支持「批量推理」:一次收多条请求、在后端内部并行处理(如 cmnn 的 C++
    # 多实例线程池)。为 True 时编排层改走批量路径(complete_batch);为 False 则
    # 逐条调 complete。默认 False:现有 openai/mnn/fake 后端无需改动。
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
        """
        raise NotImplementedError

    def complete_batch(self, items: list[BatchItem]) -> list[Prediction]:
        """批量推理:输入 N 条请求,返回**等长同序**的 N 条 Prediction。

        默认实现逐条转调 complete(串行),因此任何后端都天然「支持」批量调用、
        编排层可无差别地走批量路径。真正的并行加速由 supports_batch=True 的后端
        (如 cmnn)覆盖本方法、在原生层并行完成。

        约定:返回列表长度与 items 一致、顺序对应(便于按下标写回);实现方应像
        complete 一样把单条失败写进该条 Prediction.error,而非抛出中断整批。
        """
        return [
            self.complete(it.context, it.images, it.sample_id, it.expected)
            for it in items
        ]

    def close(self) -> None:
        """可选:释放资源(连接池等)。"""
