"""命令行入口(工作目录模型)。

机器级设置放全局配置(~/.eval_vlm/config.yaml);每个数据集是工作目录下一个
自包含文件夹(含 config.yaml + 全部产物)。各命令对 --dataset / --datadir 的含义:

    config  : 管理全局配置(init / show / set)
    split   : 初始化  —— --dataset = 源数据集 JSON 路径
              在 workspace 下建 <数据集名>/,从内置模板生成 config.yaml,再分割
    pred    : 预测(不评分),二选一:
              --dataset = 数据集名(或文件夹路径);对其 test.json 推理 -> predictions.jsonl
              --datadir = 无标注图片文件夹;逐张单轮描述,产物落 workspace/<同名>/
    score   : 读取已有 —— predictions.jsonl -> metrics.json / scored.jsonl / failures.md / summary.md
    eval    : 读取已有 —— 一键连续执行 pred(--dataset)+ score(不含 split:split 后需先部署模型)

产物按模型分目录:pred/score/eval 的结果落在 工作目录/<数据集>/<模型名>/,
不同模型对同一数据集互不覆盖(openai/vllm/fake 取 inference.openai.model;
mnn 取 inference.mnn.config_path 所在目录名);split 产物(train/test/val)是各模型
共享的,落在数据集文件夹本身。

CLI 覆盖会「永久写回」该数据集 config.yaml(用户参数优先且持久化,不再是临时):
    --base-url / --model     pred/eval 写回 inference.openai.base_url / inference.openai.model
    --scorer                 score/eval 写回 scoring.scorer
    --backend                pred 写回 inference.backend
    --mnn-config / --mnn-image-max-side   pred 写回 inference.mnn.config_path / image_max_side
    --prompt / --system-prompt            pred --datadir 写回 pred.prompt / pred.system_prompt
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Optional

from pathlib import Path

from .config import Config, DEFAULT_PROMPT, load_dataset_config
from .data.splitter import split_dataset
from .runner import run_inference
from .predict import predict_folder
from .evaluate import score_predictions
from .scoring import available_scorers
from . import workspace


# ---------------------------------------------------------------------------
# config:管理全局配置
# ---------------------------------------------------------------------------
def _cmd_config(args: argparse.Namespace) -> int:
    action = args.action
    if action == "init":
        path = workspace.init_global_config(force=args.force)
        print(f"[config] 全局配置 -> {path}")
    elif action == "show":
        path = workspace.global_config_path()
        cfg = workspace.load_global_config()
        print(f"[config] {path}")
        print(json.dumps(cfg, ensure_ascii=False, indent=2))
    elif action == "keys":
        print(workspace.describe_settable_keys())
    elif action == "set":
        if not args.key:
            print("用法: eval-vlm config set <key> <value>", file=sys.stderr)
            return 2
        value = args.value
        if value is not None and value.lower() in ("null", "none", ""):
            value = None
        path = workspace.set_global_value(args.key, value)
        print(f"[config] {args.key} = {value!r} -> {path}")
    return 0


# ---------------------------------------------------------------------------
# split:初始化数据集文件夹 + 分割
# ---------------------------------------------------------------------------
def _cmd_split(args: argparse.Namespace) -> int:
    global_cfg = workspace.load_global_config()
    ws = workspace.resolve_workspace(args.workspace, global_cfg)

    # 仅收集显式提供的 split 参数,写进生成的 config.yaml。
    so: dict = {}
    for key in ("train", "test", "val", "seed", "stratify_by"):
        val = getattr(args, key, None)
        if val is not None:
            so[key] = val

    folder = workspace.init_dataset(
        args.dataset, ws,
        name=args.name,
        split_overrides=so,
        split_defaults=global_cfg.get("split"),   # 不传 --train 等时用全局默认
        media_root=global_cfg.get("media_root", "."),
        image_strip_prefix=global_cfg.get("image_strip_prefix"),
        force=args.force,
    )
    cfg = load_dataset_config(folder)
    # 自定义产物位置(临时覆盖,可直接写到 LlamaFactory data/)。
    if args.train_out:
        cfg.split.train_out = args.train_out
    if args.val_out:
        cfg.split.val_out = args.val_out
    if args.test_out:
        cfg.split.test_out = args.test_out

    meta = split_dataset(cfg)
    counts = meta["counts"]
    files = meta["files"]
    print(f"[split] 数据集 '{folder.name}' -> {folder}")
    print(f"[split] config.yaml 已生成;共 {meta['total_samples']} 条 -> "
          f"train {counts['train']} / val {counts['val']} / test {counts['test']} "
          f"(seed={meta['seed']})")
    for name in ("train", "val", "test"):
        if name in files:
            print(f"        {name}.json (LlamaFactory 格式) -> {files[name]}")
    print(f"[split] 后续: 部署模型 -> eval-vlm eval --dataset {folder.name}")
    return 0


# ---------------------------------------------------------------------------
# run / score / eval:读取已有数据集文件夹
# ---------------------------------------------------------------------------
def _resolve_folder(args: argparse.Namespace) -> Path:
    """把 --dataset 解析成已存在的数据集文件夹(workspace 模型)。"""
    global_cfg = workspace.load_global_config()
    ws = workspace.resolve_workspace(args.workspace, global_cfg)
    return workspace.resolve_dataset_dir(args.dataset, ws)


# CLI flag 名 -> 写回 config.yaml 的点号键(用户参数优先且持久化)。
_PERSIST_MAP: tuple[tuple[str, str], ...] = (
    ("base_url", "inference.openai.base_url"),
    ("model", "inference.openai.model"),
    ("backend", "inference.backend"),
    ("mnn_config", "inference.mnn.config_path"),
    ("mnn_image_max_side", "inference.mnn.image_max_side"),
    ("scorer", "scoring.scorer"),
    ("prompt", "pred.prompt"),
    ("system_prompt", "pred.system_prompt"),
)


def _persist_overrides(folder: Path, args: argparse.Namespace) -> list[str]:
    """把用户显式提供的 CLI 覆盖永久写回该数据集 config.yaml,返回写回的键列表。

    只写回非 None 的 flag;某命令没有的 flag 自动跳过(getattr 兜底)。
    写回后由调用方 load_dataset_config 读回,实现「用户参数优先 + 持久化」。
    """
    persisted: list[str] = []
    for attr, dotted in _PERSIST_MAP:
        val = getattr(args, attr, None)
        if val is not None:
            workspace.set_dataset_value(folder, dotted, val)
            persisted.append(dotted)
    return persisted


def _report_persist(tag: str, persisted: list[str], folder: Path) -> None:
    if persisted:
        print(f"[{tag}] 已将 {', '.join(persisted)} 写回 {folder / 'config.yaml'}(永久生效)")


def _do_run(cfg: Config, tag: str = "pred") -> dict:
    stats = run_inference(cfg)
    print(f"[{tag}] 完成 {stats['newly_completed']} 个目标轮,失败 {stats['errors']} 个,"
          f"跳过(已完成样本) {stats['skipped_samples_already_done']} 条 -> {cfg.predictions_path}")
    if stats["errors"]:
        print(f"[{tag}] 注意:有 {stats['errors']} 条推理失败(已记录 error,可重跑补齐)。")
    return stats


def _do_score(cfg: Config, scorer: Optional[str]) -> dict:
    metrics = score_predictions(cfg, scorer_name=scorer)
    per_turn = metrics.get("per_turn") or {}
    print(f"[score] {len(per_turn)} 个目标轮,总体均分 {metrics.get('overall_mean_score')} "
          f"-> {cfg.metrics_path}")
    n_fail = metrics.get("num_failed_samples", 0)
    if n_fail:
        print(f"[score] exact_match 未命中 {n_fail} 个样本"
              f"(共 {metrics.get('num_failed_targets', 0)} 个错误轮),"
              f"人类可读清单 -> {cfg.failures_path}")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return metrics


def _pred_dataset(args: argparse.Namespace) -> int:
    """pred --dataset:对已有数据集的 test.json 执行推理(只预测,不评分)。

    等价于旧的 `run` 命令。--base-url/--model 等永久写回该数据集 config.yaml。
    """
    folder = _resolve_folder(args)
    persisted = _persist_overrides(folder, args)      # --base-url/--model 永久写回
    cfg = load_dataset_config(folder)
    _report_persist("pred", persisted, folder)
    print(f"[pred] 数据集预测,模型目录(按模型名区分)-> {cfg.run_dir}")
    _do_run(cfg)
    return 0


def _cmd_score(args: argparse.Namespace) -> int:
    folder = _resolve_folder(args)
    persisted = _persist_overrides(folder, args)      # --scorer 永久写回
    cfg = load_dataset_config(folder)
    _report_persist("score", persisted, folder)
    _do_score(cfg, args.scorer)
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    """一键连续执行 预测 + 评分(不含 split)。"""
    folder = _resolve_folder(args)
    persisted = _persist_overrides(folder, args)      # --base-url/--model/--scorer 永久写回
    cfg = load_dataset_config(folder)
    _report_persist("eval", persisted, folder)
    print(f"[eval] 模型目录(按模型名区分)-> {cfg.run_dir}")
    _do_run(cfg, "eval")
    _do_score(cfg, args.scorer)
    return 0


# ---------------------------------------------------------------------------
# pred:统一的「预测(不评分)」命令
#   --dataset = 对已有数据集的 test.json 推理(旧 run)
#   --datadir = 对无标注图片文件夹逐张描述(旧 pred)
# ---------------------------------------------------------------------------
def _cmd_pred(args: argparse.Namespace) -> int:
    """pred 分发:--dataset 走数据集预测;--datadir 走无标注图片描述。

    互斥参数组已保证二者恰好提供其一。
    """
    if args.dataset:
        return _pred_dataset(args)
    return _pred_datadir(args)


def _pred_datadir(args: argparse.Namespace) -> int:
    """遍历 --datadir 内所有图片,按 config.yaml 组织对话调 VLM 描述,落到 workspace/<同名>。

    自包含文件夹模型(同 split→run):首次运行在 workspace/<名>/ 生成 config.yaml
    (含 inference + pred 两段),再次运行直接读它。产物按模型落 workspace/<名>/<模型>/。
    CLI flag(--backend/--base-url/--model/--prompt 等)会永久写回该 config.yaml
    (用户参数优先且持久化);--overwrite 整份重跑覆盖已有结果。
    """
    global_cfg = workspace.load_global_config()
    ws = workspace.resolve_workspace(args.workspace, global_cfg)

    datadir = Path(args.datadir).expanduser().resolve()
    if not datadir.is_dir():
        raise FileNotFoundError(f"--datadir 不是文件夹: {datadir}")

    name = args.name or datadir.name
    out_dir = (ws / name).resolve()
    if out_dir == datadir:
        raise ValueError(
            f"输出文件夹与 --datadir 相同({out_dir});请用 --name 指定不同名字,"
            f"或把 workspace 设为别处(预测产物不应写回原图片文件夹)。"
        )

    # 生成(首次/--force)或沿用已有 config.yaml。
    config_path = out_dir / "config.yaml"
    existed = config_path.exists()
    workspace.init_pred_config(out_dir, datadir, global_cfg, force=args.force)
    action = "重新生成(--force)" if (existed and args.force) else ("沿用" if existed else "首次生成")
    print(f"[pred] {action}配置 -> {config_path}")

    # 用户 CLI 覆盖永久写回 config.yaml(--backend/--base-url/--model/--mnn-config/--prompt/--system-prompt),
    # 再读回成强类型 Config —— 用户参数优先且持久化。
    persisted = _persist_overrides(out_dir, args)
    cfg = load_dataset_config(out_dir)          # 读 config.yaml + 钉 dataset_dir=out_dir
    _report_persist("pred", persisted, out_dir)

    # 图片永远定位到 --datadir(即便 config 里 media_root 被改过)。
    cfg.data.media_root = str(datadir)
    print(f"[pred] 模型目录(按模型名区分)-> {cfg.run_dir}")

    # prompt/system_prompt 已写回 config,故 prompt=None 交给 cfg.pred 驱动。
    stats = predict_folder(cfg, datadir, prompt=None,
                           overwrite=getattr(args, "overwrite", False))
    print(f"[pred] 完成 {stats['newly_completed']} 张描述,失败 {stats['errors']} 张,"
          f"跳过(已完成) {stats['skipped_already_done']} 张 -> {stats['predictions_path']}")
    print(f"[pred] 人类可读视图 -> {cfg.run_dir / 'predictions.txt'}")
    if stats["errors"]:
        print(f"[pred] 注意:有 {stats['errors']} 张失败(已记录到 failures.jsonl,可重跑补齐)。")
    return 0


# ---------------------------------------------------------------------------
# 参数
# ---------------------------------------------------------------------------
def _add_workspace_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument("--workspace", default=None,
                   help="覆盖全局配置中的 workspace(数据集文件夹的父目录)")


def _add_inference_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--base-url", default=None,
                   help="临时覆盖 inference.openai.base_url(部署地址,如 http://localhost:8000/v1)")
    p.add_argument("--model", default=None,
                   help="临时覆盖 inference.openai.model(部署时注册的模型名)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eval_vlm",
        description="解耦的 VLM 测试集评测工具(工作目录模型:config / split / pred / score / eval)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # config
    p_config = sub.add_parser("config", help="管理全局配置(workspace/media_root 等)")
    p_config.add_argument("action", choices=["init", "show", "set", "keys"],
                          help="init=生成默认 / show=查看当前 / set=改一个键 / keys=列出所有可设置键")
    p_config.add_argument("key", nargs="?", default=None,
                          help="set 时的键:workspace/media_root/image_strip_prefix,"
                               "或 split 默认 split.train/split.test/split.val/split.seed/split.stratify_by")
    p_config.add_argument("value", nargs="?", default=None, help="set 时的值(null 表示清空)")
    p_config.add_argument("--force", action="store_true", help="init 时覆盖已有全局配置")
    p_config.set_defaults(func=_cmd_config)

    # split(初始化)
    p_split = sub.add_parser("split", help="初始化数据集文件夹并分割(--dataset=源JSON)")
    p_split.add_argument("--dataset", "-d", required=True, help="源数据集 JSON 文件路径")
    p_split.add_argument("--name", default=None, help="数据集文件夹名(默认取源文件名,不含扩展名)")
    p_split.add_argument("--train", type=float, default=None,
                         help="训练集比例(如 0.8);不传则用全局配置 split.train")
    p_split.add_argument("--test", type=float, default=None,
                         help="测试集比例(如 0.2);不传则用全局配置 split.test")
    p_split.add_argument("--val", type=float, default=None,
                         help="验证集比例(>0 才产出 val.json);不传则用全局配置 split.val")
    p_split.add_argument("--seed", type=int, default=None,
                         help="随机种子(可复现);不传则用全局配置 split.seed")
    p_split.add_argument("--stratify-by", dest="stratify_by", default=None,
                         help="分层抽样字段名(如标签字段);不传则用全局配置 split.stratify_by")
    p_split.add_argument("--train-out", default=None, help="覆盖 train.json 输出路径")
    p_split.add_argument("--val-out", default=None, help="覆盖 val.json 输出路径")
    p_split.add_argument("--test-out", default=None, help="覆盖 test.json 输出路径")
    p_split.add_argument("--force", action="store_true", help="数据集文件夹已存在时重建(覆盖 config.yaml)")
    _add_workspace_arg(p_split)
    p_split.set_defaults(func=_cmd_split)

    # score
    p_score = sub.add_parser("score", help="对已有数据集的预测评分(--dataset=名|路径)")
    p_score.add_argument("--dataset", "-d", required=True, help="数据集名(或文件夹路径)")
    p_score.add_argument("--scorer", default=None,
                         help=f"临时覆盖评分器。可用: {', '.join(available_scorers())}")
    _add_workspace_arg(p_score)
    p_score.set_defaults(func=_cmd_score)

    # eval = run + score
    p_eval = sub.add_parser("eval", help="一键连续执行 run + score(不含 split)")
    p_eval.add_argument("--dataset", "-d", required=True, help="数据集名(或文件夹路径)")
    _add_inference_args(p_eval)
    p_eval.add_argument("--scorer", default=None, help="临时覆盖评分器")
    _add_workspace_arg(p_eval)
    p_eval.set_defaults(func=_cmd_eval)

    # pred(统一预测命令:--dataset=数据集 test.json | --datadir=无标注图片文件夹)
    p_pred = sub.add_parser(
        "pred",
        help="预测(不评分):--dataset=对数据集 test.json 推理 | --datadir=对无标注图片文件夹逐张描述")
    src = p_pred.add_mutually_exclusive_group(required=True)
    src.add_argument("--dataset", default=None,
                     help="数据集名(或文件夹路径):对其 test.json 推理(等价旧 run,只预测不评分)")
    src.add_argument("--datadir", default=None,
                     help="无标注图片文件夹路径:逐张单轮描述(等价旧 pred),产物落 workspace/<同名>")
    p_pred.add_argument("--name", default=None,
                        help="[--datadir] 输出文件夹名(默认取图片文件夹名);产物落在 workspace/<名>/")
    p_pred.add_argument("--prompt", default=None,
                        help=f"[--datadir] 临时覆盖单轮提示词(不传则用文件夹 config.yaml 的 pred.prompt,"
                             f"默认 {DEFAULT_PROMPT!r};config 里设了多轮 template 时此项无效)")
    p_pred.add_argument("--system-prompt", dest="system_prompt", default=None,
                        help="[--datadir] 临时覆盖系统提示(不传则用 config.yaml 的 pred.system_prompt)")
    p_pred.add_argument("--backend", default=None,
                        choices=["openai", "vllm", "mnn", "fake"],
                        help="临时覆盖推理后端:openai/vllm(调 OpenAI 兼容 API,vllm 为别名)| "
                             "mnn(本地 pymnn 推理转换后的 mnn 模型,需 --mnn-config)| "
                             "fake(回显,不联网,自检用)")
    p_pred.add_argument("--mnn-config", dest="mnn_config", default=None,
                        help="backend=mnn 时:转换产物目录里 config.json 的路径"
                             "(临时覆盖 inference.mnn.config_path)")
    p_pred.add_argument("--mnn-image-max-side", dest="mnn_image_max_side", type=int,
                        default=None,
                        help="backend=mnn 时:图片最长边像素上限(超大图等比缩放;默认 2048;"
                             "设 0 关闭缩放)。临时覆盖 inference.mnn.image_max_side")
    p_pred.add_argument("--force", action="store_true",
                        help="[--datadir] 重新生成文件夹内 config.yaml(覆盖你的手改)")
    p_pred.add_argument("--overwrite", action="store_true",
                        help="[--datadir] 无视已有结果整份重跑(覆盖 predictions.jsonl);默认断点续跑只补未完成")
    _add_inference_args(p_pred)
    _add_workspace_arg(p_pred)
    p_pred.set_defaults(func=_cmd_pred)

    return parser


def _force_utf8_stdout() -> None:
    """Windows 控制台默认 GBK,会把中文输出变乱码;强制 UTF-8。"""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8")
            except Exception:  # noqa: BLE001
                pass


def main(argv: Optional[list[str]] = None) -> int:
    _force_utf8_stdout()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (FileExistsError, FileNotFoundError, ValueError, KeyError) as e:
        # 已知的用户侧错误:打印简洁信息,不抛完整堆栈。
        print(f"[error] {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
