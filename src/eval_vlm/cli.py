"""命令行入口(工作目录模型)。

机器级设置放全局配置(~/.eval_vlm/config.yaml);每个数据集是工作目录下一个
自包含文件夹(含 config.yaml + 全部产物)。两类命令对 --dataset 的含义不同:

    config  : 管理全局配置(init / show / set)
    split   : 初始化  —— --dataset = 源数据集 JSON 路径
              在 workspace 下建 <数据集名>/,从内置模板生成 config.yaml,再分割
    run     : 读取已有 —— --dataset = 数据集名(或文件夹路径);test.json -> predictions.jsonl
    score   : 读取已有 —— predictions.jsonl -> metrics.json / scored.jsonl / failures.md / summary.md
    eval    : 读取已有 —— 一键连续执行 run + score(不含 split:split 后需先部署模型)
    pred    : 无标注 —— --datadir = 图片文件夹;逐张单轮描述,产物落 workspace/<同名>/(不评分)

临时覆盖(不回写 config.yaml,永久改动请手改文件夹内 config.yaml):
    --base-url / --model   run/eval 注入部署地址与模型名
    --scorer               score/eval 覆盖评分器
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
def _load_existing(args: argparse.Namespace) -> Config:
    global_cfg = workspace.load_global_config()
    ws = workspace.resolve_workspace(args.workspace, global_cfg)
    folder = workspace.resolve_dataset_dir(args.dataset, ws)
    return load_dataset_config(folder)


def _apply_inference_overrides(cfg: Config, args: argparse.Namespace) -> None:
    """部署地址/模型名临时覆盖(不回写 config.yaml)。"""
    if getattr(args, "base_url", None):
        cfg.inference.base_url = args.base_url
    if getattr(args, "model", None):
        cfg.inference.model = args.model


def _do_run(cfg: Config) -> dict:
    stats = run_inference(cfg)
    print(f"[run] 完成 {stats['newly_completed']} 个目标轮,失败 {stats['errors']} 个,"
          f"跳过(已完成样本) {stats['skipped_samples_already_done']} 条 -> {cfg.predictions_path}")
    if stats["errors"]:
        print(f"[run] 注意:有 {stats['errors']} 条推理失败(已记录 error,可重跑补齐)。")
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


def _cmd_run(args: argparse.Namespace) -> int:
    cfg = _load_existing(args)
    _apply_inference_overrides(cfg, args)
    _do_run(cfg)
    return 0


def _cmd_score(args: argparse.Namespace) -> int:
    cfg = _load_existing(args)
    _do_score(cfg, args.scorer)
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    """一键连续执行 run + score(不含 split)。"""
    cfg = _load_existing(args)
    _apply_inference_overrides(cfg, args)
    _do_run(cfg)
    _do_score(cfg, args.scorer)
    return 0


# ---------------------------------------------------------------------------
# pred:无标注图片文件夹 -> 单轮描述(不评分)
# ---------------------------------------------------------------------------
def _cmd_pred(args: argparse.Namespace) -> int:
    """遍历 --datadir 内所有图片,按 config.yaml 组织对话调 VLM 描述,落到 workspace/<同名>。

    自包含文件夹模型(同 split→run):首次运行在 workspace/<名>/ 生成 config.yaml
    (含 inference + pred 两段),再次运行直接读它。改 vLLM API 或对话组织 = 手改该
    config.yaml 重跑。CLI flag 为临时覆盖,优先级:flag > 文件夹 config.yaml > 模板默认。
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

    # 生成(首次/--force)或沿用已有 config.yaml,然后读回成强类型 Config。
    config_path = out_dir / "config.yaml"
    existed = config_path.exists()
    workspace.init_pred_config(out_dir, datadir, global_cfg, force=args.force)
    cfg = load_dataset_config(out_dir)          # 读 config.yaml + 钉 run_dir_path=out_dir
    action = "重新生成(--force)" if (existed and args.force) else ("沿用" if existed else "首次生成")
    print(f"[pred] {action}配置 -> {config_path}")

    # 图片永远定位到 --datadir(即便 config 里 media_root 被改过)。
    cfg.data.media_root = str(datadir)
    # CLI 临时覆盖(不回写 config.yaml)。
    if args.backend is not None:
        cfg.inference.backend = args.backend
    _apply_inference_overrides(cfg, args)       # --base-url / --model
    if args.system_prompt is not None:
        cfg.pred.system_prompt = args.system_prompt

    stats = predict_folder(cfg, datadir, prompt=args.prompt)   # --prompt 显式才覆盖(默认 None)
    print(f"[pred] 完成 {stats['newly_completed']} 张描述,失败 {stats['errors']} 张,"
          f"跳过(已完成) {stats['skipped_already_done']} 张 -> {stats['predictions_path']}")
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
                   help="临时覆盖 inference.base_url(部署地址,如 http://localhost:8000/v1)")
    p.add_argument("--model", default=None, help="临时覆盖 inference.model(部署时注册的模型名)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eval_vlm",
        description="解耦的 VLM 测试集评测工具(工作目录模型:config / split / run / score / eval)",
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

    # run
    p_run = sub.add_parser("run", help="对已有数据集执行推理(--dataset=名|路径)")
    p_run.add_argument("--dataset", "-d", required=True, help="数据集名(或文件夹路径)")
    _add_inference_args(p_run)
    _add_workspace_arg(p_run)
    p_run.set_defaults(func=_cmd_run)

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

    # pred(无标注图片描述)
    p_pred = sub.add_parser(
        "pred", help="对无标注图片文件夹逐张描述(不评分;--datadir=图片文件夹)")
    p_pred.add_argument("--datadir", "-d", required=True, help="待描述的图片文件夹路径")
    p_pred.add_argument("--name", default=None,
                        help="输出文件夹名(默认取图片文件夹名);产物落在 workspace/<名>/")
    p_pred.add_argument("--prompt", default=None,
                        help=f"临时覆盖单轮提示词(不传则用文件夹 config.yaml 的 pred.prompt,"
                             f"默认 {DEFAULT_PROMPT!r};config 里设了多轮 template 时此项无效)")
    p_pred.add_argument("--system-prompt", dest="system_prompt", default=None,
                        help="临时覆盖系统提示(不传则用 config.yaml 的 pred.system_prompt)")
    p_pred.add_argument("--backend", default=None, choices=["openai", "fake"],
                        help="临时覆盖推理后端:openai(调真实 API)| fake(回显,不联网,自检用)")
    p_pred.add_argument("--force", action="store_true",
                        help="重新生成文件夹内 config.yaml(覆盖你的手改)")
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
