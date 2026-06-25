"""执行测试:读 test.json -> 逐轮 rollout 推理 -> 追加写 predictions.jsonl。

多轮对话(如 轮1描述 -> 轮2标签)按 assistant 轮**逐轮 rollout**:
每个目标轮单独生成、单独记录一条 Prediction(用 (id, turn) 唯一标识)。
rollout 模式下,后续轮的上下文用模型**自己生成**的前文;gold 模式用数据集原文。

并发在**样本级**(rollout 有轮间依赖,单条样本内部顺序执行)。
断点续跑:某样本所有目标轮都已成功则跳过,否则整条样本重跑(保证 rollout 一致)。
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

from tqdm import tqdm

from .config import Config
from .data.loader import load_samples
from .data.schema import Prediction, Sample, Turn
from .data.splitter import load_split_meta
from .inference import build_backend, worker_count
from .inference.base import InferenceBackend
from .results import store


def _build_context(sample: Sample, upto: int, predicted: dict[int, str],
                   mode: str) -> list[Turn]:
    """构造截至 turns[upto] 之前的对话上下文。

    历史 assistant 轮的内容:rollout 模式优先用模型已生成的预测(predicted),
    否则(gold 模式 / 非目标轮 / 缺失)回落到数据集原文。
    """
    ctx: list[Turn] = []
    for j in range(upto):
        turn = sample.turns[j]
        if turn.role == "assistant" and mode == "rollout" and j in predicted:
            ctx.append(Turn(role="assistant", content=predicted[j]))
        else:
            ctx.append(turn)
    return ctx


def _rollout_sample(cfg: Config, backend: InferenceBackend,
                    sample: Sample) -> list[Prediction]:
    """对单条样本逐轮 rollout,返回每个目标轮一条 Prediction。"""
    mode = cfg.eval.context
    predicted: dict[int, str] = {}
    preds: list[Prediction] = []
    for target in sample.targets:
        ctx = _build_context(sample, target.turn_index, predicted, mode)
        pred = backend.complete(ctx, sample.images, sample.id,
                                expected=target.reference)
        pred.turn = target.turn_index
        # 记录原始图片地址,使每条预测都能追溯回原图(人工核查用)。
        pred.images = list(sample.images)
        preds.append(pred)
        # 成功才登记,供后续轮在 rollout 模式下作上下文;失败则后续轮回落 gold。
        if not pred.error:
            predicted[target.turn_index] = pred.prediction
    return preds


def run_inference(cfg: Config) -> dict:
    """对测试集执行推理,返回统计信息。"""
    if not cfg.test_path.exists():
        raise FileNotFoundError(
            f"未找到测试集 {cfg.test_path},请先运行: python -m eval_vlm split"
        )
    samples = load_samples(cfg, source=cfg.test_path)   # 按 LlamaFactory 格式读 test.json
    meta = load_split_meta(cfg)

    done_keys = store.load_prediction_keys(cfg.predictions_path)

    def sample_done(s: Sample) -> bool:
        return bool(s.targets) and all(
            (s.id, t.turn_index) in done_keys for t in s.targets
        )

    todo: list[Sample] = [s for s in samples if not sample_done(s)]
    n_targets = sum(len(s.targets) for s in samples)

    print(f"[run] 待推理 {len(todo)} 条样本(已完成跳过 {len(samples) - len(todo)} 条),"
          f"正在加载后端/模型({cfg.inference.backend})...", flush=True)
    backend = build_backend(cfg)
    max_workers = worker_count(backend, cfg.inference.max_concurrency)
    if cfg.inference.max_concurrency > 1 and max_workers == 1:
        import warnings
        warnings.warn(
            f"[run] 注意: max_concurrency={cfg.inference.max_concurrency} 对 "
            f"{cfg.inference.backend} 后端不生效(该后端为有状态单对象,必须串行推理),"
            f"实际并发=1。如需并行加速,请换用 openai/vllm 后端。",
            stacklevel=1,
        )
    print("[run] 后端就绪,开始推理。", flush=True)
    started = datetime.now(timezone.utc).isoformat()
    n_ok = 0
    n_err = 0

    if not todo:
        print(f"全部 {len(samples)} 条样本({n_targets} 个目标轮)已完成,无需推理(断点续跑)。")
    else:
        with store.PredictionWriter(cfg.predictions_path) as writer:
            try:
                if max_workers == 1:
                    pbar = tqdm(todo, total=len(todo), desc="inference", unit="sample")
                    for s in pbar:
                        pbar.set_postfix_str(s.id)
                        for pred in _rollout_sample(cfg, backend, s):
                            writer.write(pred)
                            if pred.error:
                                n_err += 1
                            else:
                                n_ok += 1
                else:
                    with ThreadPoolExecutor(max_workers=max_workers) as pool:
                        futures = {pool.submit(_rollout_sample, cfg, backend, s): s for s in todo}
                        try:
                            for fut in tqdm(as_completed(futures), total=len(futures),
                                            desc="inference", unit="sample"):
                                for pred in fut.result():
                                    writer.write(pred)
                                    if pred.error:
                                        n_err += 1
                                    else:
                                        n_ok += 1
                        except KeyboardInterrupt:
                            for fut in futures:
                                fut.cancel()
                            raise
            except KeyboardInterrupt:
                print(f"\n[run] 已中断:本轮成功 {n_ok} 条、失败 {n_err} 条均已落盘;"
                      f"重跑同一命令即可断点续跑。", flush=True)
    backend.close()

    stats = {
        "run_name": cfg.run_name,
        "model": cfg.inference.result_name,
        "backend": cfg.inference.backend,
        "base_url": getattr(cfg.inference.active, "base_url", None),
        "eval_targets": cfg.eval.targets,
        "eval_context": cfg.eval.context,
        "test_size": len(samples),
        "num_targets": n_targets,
        "newly_completed": n_ok,
        "errors": n_err,
        "skipped_samples_already_done": len(samples) - len(todo),
        "started_at": started,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "split_source": meta.get("source"),
        "split_source_sha256": meta.get("source_sha256"),
    }
    store.write_json(cfg.run_meta_path, stats)
    return stats
