"""runner 的批处理路径(_rollout_batch):按轮次跨样本批量 + 多轮 rollout 上下文串联。"""
from __future__ import annotations

from eval_vlm import runner
from eval_vlm.config import Config
from eval_vlm.data.schema import Prediction, Sample, Turn


class _StubWriter:
    def __init__(self):
        self.rows = []

    def write(self, pred):
        self.rows.append(pred)


class _BatchStub:
    """支持批处理的假后端:记录每轮收到的 items,预测回显该轮上下文长度与最后一轮内容。"""
    supports_batch = True

    def __init__(self):
        self.calls = []          # 每次 complete_batch 收到的 items

    def complete_batch(self, items):
        self.calls.append(items)
        out = []
        for it in items:
            # 预测里带上下文里最后一个 assistant 的内容(若有),便于验证 rollout 串联
            prev_assist = [t.content for t in it.context if t.role == "assistant"]
            tag = f"gen[{it.sample_id}]<-{prev_assist[-1] if prev_assist else 'NONE'}"
            out.append(Prediction(id=it.sample_id, prediction=tag))
        return out


def _two_turn_sample(sid: str) -> Sample:
    # user0 / assistant0(目标) / user1 / assistant1(目标)
    turns = [Turn("user", "<image>描述"), Turn("assistant", "GOLD0"),
             Turn("user", "追问"), Turn("assistant", "GOLD1")]
    from eval_vlm.data.schema import EvalTurn
    return Sample(id=sid, turns=turns, images=[f"{sid}.jpg"],
                  targets=[EvalTurn(1, "GOLD0"), EvalTurn(3, "GOLD1")])


def test_rollout_batch_batches_per_round_and_threads_context():
    cfg = Config()
    cfg.eval.context = "rollout"
    todo = [_two_turn_sample("a"), _two_turn_sample("b")]
    backend = _BatchStub()
    writer = _StubWriter()

    n_ok, n_err = runner._rollout_batch(cfg, backend, todo, writer)

    # 2 轮,每轮一次 complete_batch,每次含 2 个样本(跨样本批量)
    assert len(backend.calls) == 2
    assert [len(c) for c in backend.calls] == [2, 2]
    assert n_ok == 4 and n_err == 0                 # 2 样本 × 2 目标轮

    # 第 0 轮上下文无 assistant 历史
    round0 = {it.sample_id: it for it in backend.calls[0]}
    assert all(t.role != "assistant" for t in round0["a"].context)

    # 第 1 轮上下文里应含**上一轮生成**(rollout),而非 GOLD0
    round1 = {it.sample_id: it for it in backend.calls[1]}
    a_ctx_assist = [t.content for t in round1["a"].context if t.role == "assistant"]
    assert a_ctx_assist == ["gen[a]<-NONE"]         # 用了 a 自己第0轮的输出,不是 GOLD0

    # 写盘:每条 pred 带正确 turn / images
    by = {(p.id, p.turn) for p in writer.rows}
    assert by == {("a", 1), ("a", 3), ("b", 1), ("b", 3)}
    assert all(p.images for p in writer.rows)


def test_rollout_batch_gold_context_mode():
    """gold 模式:第 1 轮上下文用数据集原文(GOLD0),不注入模型输出。"""
    cfg = Config()
    cfg.eval.context = "gold"
    todo = [_two_turn_sample("a")]
    backend = _BatchStub()
    writer = _StubWriter()
    runner._rollout_batch(cfg, backend, todo, writer)
    round1 = backend.calls[1][0]
    a_ctx_assist = [t.content for t in round1.context if t.role == "assistant"]
    assert a_ctx_assist == ["GOLD0"]
