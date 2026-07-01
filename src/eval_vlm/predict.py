"""无标注图片描述:遍历一个图片文件夹 -> 调用 VLM -> 落盘描述结果。

与 run/score 的「带标注测试集」链路不同,这里面对的是**无标注照片**:
没有数据集 JSON、没有标准答案、不评分。每张图片各起一段独立对话,
对话结构由 cfg.pred 决定(单轮简写 prompt 或多轮 template):

    user:      <image>请描述图片
    assistant: <模型生成的描述>

产物落在工作目录(workspace)下一个与图片文件夹同名的文件夹里(与 split 一致),
每条结果写成**原样 LlamaFactory 格式**(messages + images),因此可直接当作
一个新数据集复用(例如人工校对后用于训练)。

并发在**图片级**(各图相互独立);断点续跑:predictions.jsonl 里已成功的图片跳过。
"""
from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from .config import Config, PredConfig
from .data.schema import Prediction, Turn
from .inference import build_backend, worker_count
from .inference.base import InferenceBackend
from .results import store

# 识别为图片的扩展名(大小写不敏感)。
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"}


def list_images(datadir: Path) -> list[Path]:
    """列出文件夹内的图片文件(仅当前层,不递归),按文件名排序保证可复现。"""
    return sorted(
        p for p in datadir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def _validate_context(turns: list[Turn]) -> None:
    """校验对话模板合法(在构造时一次性报错,而非每张图都失败)。

    规则(对齐 OpenAIBackend._build_messages 的行为):
      1. 全部轮中 <image> 恰好 1 个 —— 0 个则图片永不发送;>1 个后端会按
         「占位符多于图片」报错(每张图都是单图)。
      2. 含 <image> 的那一轮必须是 user —— assistant/system 轮被原样透传,图会被静默丢弃。
      3. 最后一轮必须是 user —— 否则模型没有可作答的最终提问。
    """
    if not turns:
        raise ValueError("pred 对话模板为空(至少需要一个含 <image> 的 user 轮)")
    img_total = sum(t.content.count("<image>") for t in turns)
    if img_total != 1:
        raise ValueError(
            f"pred 对话中 <image> 占位符必须恰好出现 1 次,当前 {img_total} 次"
        )
    img_turn = next(t for t in turns if "<image>" in t.content)
    if img_turn.role != "user":
        raise ValueError(f"<image> 必须位于 user 轮,当前在 {img_turn.role!r} 轮")
    if turns[-1].role != "user":
        raise ValueError(
            f"对话最后一轮必须是 user(模型据此作答),当前为 {turns[-1].role!r} 轮"
        )


def build_context(pred_cfg: PredConfig, *, prompt_override: Optional[str] = None) -> list[Turn]:
    """据 pred 配置构造**每张图通用**的对话上下文(图片由 <image> 占位)。

    有 template(多轮)则按它构造(覆盖 prompt);否则用单轮 prompt——
    prompt_override(CLI --prompt)优先于 pred_cfg.prompt;若都不含 <image> 则自动前置。
    构造后立即校验合法性。
    """
    if pred_cfg.template:
        turns: list[Turn] = []
        for i, t in enumerate(pred_cfg.template):
            if not isinstance(t, dict) or "role" not in t or "content" not in t:
                raise ValueError(
                    f"pred.template 第 {i} 项必须是含 role/content 的字典,当前: {t!r}"
                )
            turns.append(Turn(role=str(t["role"]), content=str(t["content"])))
    else:
        prompt = prompt_override if prompt_override is not None else pred_cfg.prompt
        content = prompt if "<image>" in prompt else f"<image>{prompt}"
        turns = [Turn(role="user", content=content)]
    _validate_context(turns)
    return turns


def _done_images(path: Path) -> set[str]:
    """读取 predictions.jsonl 中已成功描述的图片名(用于断点续跑跳过)。

    以记录里的 id(= 图片文件名)为准;有 error 字段的不算完成,允许重跑。
    """
    done: set[str] = set()
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
                done.add(str(obj["id"]))
    return done


def _describe_one(
    backend: InferenceBackend, image_name: str, context: list[Turn]
) -> Prediction:
    """对单张图片按给定对话上下文生成描述,返回 Prediction。

    image_name 既作样本 id(可追溯到原图),也作 <image> 占位符消费的图片引用
    (相对 cfg.data.media_root = 图片文件夹解析)。后端自身已捕获异常并写入 error。
    context 在所有图片间共享、只读;后端按需读取,不修改它。
    """
    pred = backend.complete(context, [image_name], image_name)
    pred.turn = len(context)      # assistant 答案的轮下标(turns = context + [assistant])
    pred.images = [image_name]
    return pred


def _to_record(image_name: str, context: list[Turn], pred: Prediction) -> dict:
    """成功预测 -> 原样 LlamaFactory 记录(messages + images,可直接复用为数据集)。

    messages = 全部上下文轮(含 <image> 占位符字面保留)+ 末尾模型答案;
    额外保留 id/latency 便于追溯(LlamaFactory 只读 messages/images,忽略多余键)。
    """
    messages = [{"role": t.role, "content": t.content} for t in context]
    messages.append({"role": "assistant", "content": pred.prediction})
    return {
        "id": image_name,
        "images": [image_name],
        "messages": messages,
        "latency": pred.latency,
    }


def _render_txt_block(rec: dict) -> str:
    """把一条 LlamaFactory 记录渲染成人类可读文本块(predictions.txt 用)。

        【图像名】xxx.jpg
        【对话】:
            user: ...
            assistant: ...

    <image> 占位符对人类阅读是噪音(图片名已在标题里),去掉只留提问文本。
    """
    lines = [f"【图像名】{rec.get('id', '')}", "【对话】:"]
    for m in rec.get("messages", []):
        content = str(m.get("content", "")).replace("<image>", "").strip()
        lines.append(f"\t{m.get('role', '')}: {content}")
    return "\n".join(lines) + "\n\n"


def _rebuild_txt_from_jsonl(jsonl_path: Path, txt_path: Path) -> None:
    """据 predictions.jsonl 整份重建人类可读的 predictions.txt(权威视图)。

    txt 是 jsonl 的纯派生视图:每次跑完按最终 jsonl 重渲染一次,保证两者一致
    (覆盖续跑、--overwrite、以及本特性之前已有的 jsonl)。jsonl 不在则不生成。
    """
    if not jsonl_path.exists():
        return
    blocks: list[str] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("error") is None and "id" in rec:
                blocks.append(_render_txt_block(rec))
    with txt_path.open("w", encoding="utf-8") as tf:
        tf.write("".join(blocks))
        tf.flush()
        try:
            os.fsync(tf.fileno())
        except OSError:
            pass


def predict_folder(cfg: Config, datadir: Path, *, prompt: Optional[str] = None,
                   overwrite: bool = False) -> dict:
    """遍历 datadir 内所有图片,按 cfg.pred 组织对话调用 VLM 描述并落盘。返回统计。

    prompt(可选)为 CLI --prompt 临时覆盖;None 表示用 cfg.pred 的配置。
    overwrite=True 时忽略已有结果、整份重跑(predictions.jsonl 截断重写);
    默认 False 为断点续跑(已成功的图片跳过,追加写)。

    产物(落在 cfg.run_dir = workspace/<图片文件夹名>/<模型>):
      predictions.jsonl — 每行一条成功描述(LlamaFactory 格式,可复用),追加写支持续跑;
      failures.jsonl    — 每行一条失败记录(id/image/error),供排查/重跑;
      pred_meta.json    — 运行元信息(模型/后端/对话结构/计数/时间)。
    """
    images = list_images(datadir)
    if not images:
        raise FileNotFoundError(
            f"图片文件夹内未找到图片: {datadir}"
            f"(支持扩展名: {', '.join(sorted(IMAGE_EXTS))})"
        )

    # system_prompt 运行时映射到当前后端块,复用后端既有系统消息处理
    # (后端块若不支持系统提示——如 mnn——则跳过,不产生「设了不生效」的死配置)。
    if cfg.pred.system_prompt is not None:
        active = cfg.inference.active
        if hasattr(active, "system_prompt"):
            active.system_prompt = cfg.pred.system_prompt
    # 对话上下文每张图通用,循环外构造一次(并发线程间只读共享)。
    context = build_context(cfg.pred, prompt_override=prompt)

    out_dir = cfg.run_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = cfg.predictions_path
    fail_path = out_dir / "failures.jsonl"
    txt_path = out_dir / "predictions.txt"      # jsonl 的人类可读视图

    # overwrite:无视已有结果整份重跑;否则断点续跑跳过已成功项。
    done = set() if overwrite else _done_images(pred_path)
    todo = [p for p in images if p.name not in done]

    # 后端构造可能很慢(如 MNN 的 model.load() 要加载权重);显式打点便于区分
    # "卡在加载" vs "卡在推理"——中断在加载阶段时 predictions.jsonl 本就还没有任何结果。
    print(f"[pred] 待描述 {len(todo)} 张(已完成跳过 {len(images) - len(todo)} 张),"
          f"正在加载后端/模型({cfg.inference.backend})...", flush=True)
    backend = build_backend(cfg)
    max_workers = worker_count(backend, cfg.inference.max_concurrency)
    if cfg.inference.max_concurrency > 1 and max_workers == 1:
        import warnings
        warnings.warn(
            f"[pred] 注意: max_concurrency={cfg.inference.max_concurrency} 对 "
            f"{cfg.inference.backend} 后端不生效(该后端为有状态单对象,必须串行推理),"
            f"实际并发=1。如需并行加速,请换用 openai/vllm 后端。",
            stacklevel=1,
        )
    print("[pred] 后端就绪,开始推理。", flush=True)
    started = datetime.now(timezone.utc).isoformat()
    n_ok = 0
    n_err = 0
    interrupted = False

    if not todo:
        print(f"全部 {len(images)} 张图片已完成描述,无需推理(断点续跑)。")
    else:
        pred_path.parent.mkdir(parents=True, exist_ok=True)

        def _append(fh, obj: dict) -> None:
            """写一条 -> flush -> fsync:即便进程随后崩溃/被杀(MNN 原生 segfault 等),
            已写结果也已落盘,下次同命令断点续跑可据此跳过。"""
            fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
            fh.flush()
            try:
                os.fsync(fh.fileno())
            except OSError:                       # 某些文件系统不支持 fsync,flush 已够进程级崩溃
                pass

        def _record(pred: Prediction, ok_fh, fail_fh, txt_fh) -> None:
            nonlocal n_ok, n_err
            if pred.error:
                n_err += 1
                _append(fail_fh, {"id": pred.id, "image": pred.id, "error": pred.error})
            else:
                n_ok += 1
                rec = _to_record(pred.id, context, pred)
                _append(ok_fh, rec)
                # 同步写人类可读视图(崩溃也保留本轮已生成的);跑完再整份重建保证一致。
                txt_fh.write(_render_txt_block(rec))
                txt_fh.flush()
                try:
                    os.fsync(txt_fh.fileno())
                except OSError:
                    pass

        # predictions:续跑追加写(安全),overwrite 时截断重写;failures 每次重写(只反映本轮未完成项)。
        with pred_path.open("w" if overwrite else "a", encoding="utf-8") as ok_fh, \
                fail_path.open("w", encoding="utf-8") as fail_fh, \
                txt_path.open("w" if overwrite else "a", encoding="utf-8") as txt_fh:
            try:
                if max_workers == 1:
                    # 串行后端(如 MNN):同一线程「算一条 -> 立即写盘」,不经线程池。
                    # 这对 MNN 尤其关键:pymnn 原生推理基本不释放 GIL,旧的线程池写法里
                    # worker 线程会连续跑完多张、主线程的写盘循环却抢不到调度,一旦某张在
                    # 原生层 Segfault(core dump)整进程被杀,之前已生成的结果根本没落盘就丢了。
                    # 这里推理返回即 flush+fsync 落盘,segfault 也只丢正在跑的那一张,其余可续跑。
                    pbar = tqdm(todo, total=len(todo), desc="describe", unit="image")
                    for p in pbar:
                        pbar.set_postfix_str(p.name)   # 崩溃时进度条停留的就是肇事图片
                        _record(_describe_one(backend, p.name, context), ok_fh, fail_fh, txt_fh)
                else:
                    # 线程安全后端:并发推理,每条完成即写盘。
                    with ThreadPoolExecutor(max_workers=max_workers) as pool:
                        futures = {
                            pool.submit(_describe_one, backend, p.name, context): p
                            for p in todo
                        }
                        try:
                            for fut in tqdm(as_completed(futures), total=len(futures),
                                            desc="describe", unit="image"):
                                _record(fut.result(), ok_fh, fail_fh, txt_fh)
                        except KeyboardInterrupt:
                            # 取消尚未开始的任务,避免 shutdown 把整队列跑完却不写盘。
                            for fut in futures:
                                fut.cancel()
                            raise
            except KeyboardInterrupt:
                interrupted = True
                print(f"\n[pred] 已中断:本轮成功 {n_ok} 张、失败 {n_err} 张均已落盘 "
                      f"-> {pred_path};重跑同一命令即可断点续跑补齐剩余。", flush=True)
    backend.close()
    # txt 是 jsonl 的人类可读视图:据最终 jsonl 整份重建,保证两者完全一致
    # (覆盖续跑累积、--overwrite、以及本特性之前已有的 jsonl)。
    _rebuild_txt_from_jsonl(pred_path, txt_path)

    stats = {
        "datadir": str(datadir),
        "model": cfg.inference.result_name,
        "backend": cfg.inference.backend,
        "base_url": getattr(cfg.inference.active, "base_url", None),
        # 对话结构:单轮记 prompt;多轮记轮数(prompt 为 null)。便于复现与排查。
        "prompt": None if cfg.pred.template else (
            prompt if prompt is not None else cfg.pred.prompt),
        "num_context_turns": len(context),
        "system_prompt": cfg.inference.system_prompt,
        "num_images": len(images),
        "newly_completed": n_ok,
        "errors": n_err,
        "skipped_already_done": len(images) - len(todo),
        "interrupted": interrupted,           # 中途被 Ctrl-C:已落盘结果仍有效,可断点续跑
        "predictions_path": str(pred_path),
        "started_at": started,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    store.write_json(out_dir / "pred_meta.json", stats)
    return stats
