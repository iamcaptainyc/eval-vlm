#!/usr/bin/env python
"""Optuna 自动超参搜索:LlamaFactory 训练 → 合并 → 离线 vLLM 评测(eval-vlm)→ 读指标。

单卡单机**串行**:每个 trial 依次跑 train → export(合并)→ eval-vlm pred → eval-vlm
field-eval(或 eval)→ 读 run_dir 里的 json 指标,回传给 Optuna。训练/评测用不同 conda
环境(见 config 的 train_env / eval_env),各步骤都是独立子进程,天然错开显存。

用法:
    conda run -n <driver_env> python tune.py --config config.yaml
断点续跑:重跑同一命令即可。n_trials 是**总预算**——已终结(完成/剪枝/失败)的 trial 都计入,
本次只补齐到总数(而非每次都新跑 n_trials 个);study 持久化在 sqlite。

依赖:optuna、pyyaml(装在跑本驱动的环境即可;vllm/eval-vlm 装在 eval_env,
LlamaFactory 装在 train_env)。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

try:
    import optuna
except ImportError:  # pragma: no cover
    sys.exit("需要 optuna:pip install optuna")


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def suggest(trial, name: str, spec: dict):
    """据 search_space 的一项声明在 trial 上采样。"""
    t = spec["type"]
    if t == "float_log":
        return trial.suggest_float(name, float(spec["low"]), float(spec["high"]), log=True)
    if t == "float":
        return trial.suggest_float(name, float(spec["low"]), float(spec["high"]))
    if t == "int":
        return trial.suggest_int(name, int(spec["low"]), int(spec["high"]), step=int(spec.get("step", 1)))
    if t == "categorical":
        return trial.suggest_categorical(name, spec["choices"])
    raise ValueError(f"未知搜索类型 {t!r}(可选 float_log/float/int/categorical)")


def render_train_args(args: dict) -> list[str]:
    """把 {key: val} 渲染成 ['--key','val',...];bool 用 True/False(与 llamafactory 兼容)。"""
    out: list[str] = []
    for k, v in args.items():
        out.append(f"--{k}")
        out.append("True" if v is True else "False" if v is False else str(v))
    return out


def run(cmd: list[str], cwd: str | None = None, env: dict | None = None) -> None:
    """跑子进程,输出实时透传到终端;非零退出抛异常(由 objective 捕获转 TrialPruned)。

    env 非 None 时作为子进程环境(用于给 vLLM 评测注入 FLASHINFER_* 等变量,须在子进程
    python 启动前设好,才能早于 import vllm 生效)。
    """
    print("\n$ " + " ".join(shlex.quote(c) for c in cmd) + (f"   (cwd={cwd})" if cwd else ""),
          flush=True)
    r = subprocess.run(cmd, cwd=cwd, env=env)
    if r.returncode != 0:
        raise RuntimeError(f"命令失败(exit {r.returncode}):{cmd[:3]}...")


def conda_prefix(cfg: dict, env: str) -> list[str]:
    """conda_run 前缀(如 'conda run --no-capture-output -n')split + 环境名。"""
    return cfg["conda_run"].split() + [env]


def safe_name(s: str) -> str:
    """把名字清洗成合法目录名:只保留字母/数字,**其余特殊符号(含 .、- 等)一律折成 _**。

    与 eval-vlm 的 safe_model_dirname 对齐:本函数产物是其安全字符集的子集,传给
    eval-vlm 不会再被二次改写,保证 tune.py 拼出的 run_dir 与 eval-vlm 实际写入的一致。
    如 Qwen3.5-0.8B -> Qwen3_5_0_8B。
    """
    return re.sub(r"[^0-9A-Za-z]+", "_", s).strip("_") or "model"


def _trial_time(trial: "optuna.Trial") -> str:
    """trial 启动时间(本地),格式 %Y%m%d_%H%M%S——全 [0-9A-Za-z_],与 safe_name 幂等。

    optuna 的 datetime_start 是 UTC,这里转本地(服务器时区);未启动/时区缺失回退 now。
    """
    t = getattr(trial, "datetime_start", None)
    if t is None or t.tzinfo is None:
        t = datetime.now().astimezone()
    else:
        t = t.astimezone()
    return t.strftime("%Y%m%d_%H%M%S")


def trial_artifact_tag(base_tag: str, study_tag: str, number: int, started_at: str) -> str:
    """trial 产物目录名 = 基座名_study名_trial_序号_时间(全 [0-9A-Za-z_],eval-vlm 幂等)。

    序号在前、时间在后,目录按 trial 序号排,一眼看出是第几个 + 何时跑的;时间戳保证
    共享 db/共享 work_root 下每 trial 唯一。
    """
    return f"{base_tag}_{study_tag}_trial_{number:04d}_{started_at}"


_METRIC_MAP = {
    "field_exact_match": "exact_match_rate",
    "field_micro": "micro_accuracy",
    "field_macro": "macro_accuracy",
}


def read_metric(run_dir: Path, objective: dict, field_eval: bool) -> tuple[float, dict]:
    """从 eval-vlm 落盘的 json 读优化目标值 + 完整指标 dict。"""
    metric = objective["metric"]
    if field_eval:
        m = json.loads((run_dir / "field_metrics.json").read_text(encoding="utf-8"))
        if metric not in _METRIC_MAP:
            raise ValueError(f"field_eval=true 时 objective.metric 应为 {list(_METRIC_MAP)},收到 {metric!r}")
        return float(m["overall"][_METRIC_MAP[metric]]), m
    m = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    return float(m["overall_mean_score"]), m


# leaderboard 固定列(与 field_eval 开关无关的稳定超集,避免续跑切换模式时列错位)
# 头部 study/model/started_at 标识该行属于哪个实验——多 study 共享 work_root/db 时行才可分辨。
_LEADERBOARD_FIELDS = ["study", "model", "started_at", "trial", "objective",
                       "exact_match_rate", "micro_accuracy", "macro_accuracy",
                       "per_field", "params", "merged_dir"]


def append_leaderboard(path: Path, row: dict) -> None:
    """把一行 trial 结果追加进 leaderboard.csv(首行写表头);用固定列,缺项留空。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    full = {k: row.get(k, "") for k in _LEADERBOARD_FIELDS}
    with path.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_LEADERBOARD_FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(full)


def validate_config(cfg: dict) -> None:
    """跑前一次性校验必填键,避免 KeyError 被 objective 的 except 吞成"全部 trial 失败"。"""
    for k in ("train_env", "eval_env", "conda_run", "base_model", "llamafactory_dir",
              "eval_vlm_dataset", "work_root", "base_train_args", "merge_args",
              "search_space", "objective", "eval", "optuna"):
        if k not in cfg:
            raise SystemExit(f"config 缺少必填键: {k}")
    for k in ("gpu_memory_utilization", "max_model_len", "image_max_pixels"):
        if k not in cfg["eval"]:
            raise SystemExit(f"config.eval 缺少必填键: {k}")
    if "metric" not in cfg["objective"]:
        raise SystemExit("config.objective 缺少必填键: metric")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def build_objective(cfg: dict):
    work_root = Path(cfg["work_root"])
    eval_ds = cfg["eval_vlm_dataset"]
    ev = cfg["eval"]
    field_eval = bool(ev.get("field_eval", True))
    objective_cfg = cfg["objective"]
    leaderboard = work_root / "leaderboard.csv"
    # 评测子进程环境:在 os.environ 基础上叠加 eval_env_vars(如禁用 flashinfer 的变量)。
    # 子进程 python 启动前即带上,保证早于 import vllm 生效。None 表示不改(继承当前环境)。
    eval_env_vars = cfg.get("eval_env_vars") or {}
    eval_proc_env = ({**os.environ, **{str(k): str(v) for k, v in eval_env_vars.items()}}
                     if eval_env_vars else None)

    # merged 目录名 = 基座名_study名_trial_000N:基座名清洗掉特殊符号,study 名作实验标识,
    # 让同一 work_root 下不同实验(不同 study)不撞名,eval-vlm 的 result_name(取目录名)也带全标识。
    base_tag = safe_name(Path(cfg["base_model"]).name)               # Qwen3.5-0.8B -> Qwen3_5_0_8B
    study_tag = safe_name(cfg["optuna"].get("study_name", "autotune"))  # 与 main() 里 study 名默认值一致

    def objective(trial: "optuna.Trial") -> float:
        params = {name: suggest(trial, name, spec) for name, spec in cfg["search_space"].items()}
        # 派生耦合参数:alpha = lora_alpha_ratio × 采样的 lora_rank(遵循 rank=1/2·alpha 等约定)。
        alpha_ratio = cfg.get("lora_alpha_ratio")
        if alpha_ratio and "lora_rank" in params:
            params["lora_alpha"] = int(round(alpha_ratio * params["lora_rank"]))
        # merged 目录名 = 基座名_study名_trial_000N_时间:同一 work_root 下不同实验不撞名;
        # 时间戳(trial 启动的本地时间)让目录唯一、可追溯"何时跑的";eval-vlm 的
        # result_name(取目录名)随之带全基座+实验标识。adapter 为临时产物,
        # 合并成功后删除,只留合并权重。
        started_at = _trial_time(trial)
        trial_tag = trial_artifact_tag(base_tag, study_tag, trial.number, started_at)
        merged_dir = work_root / trial_tag
        adapter_dir = work_root / f".{trial_tag}.adapter"     # 临时:合并后删
        merge_path = work_root / f"{trial_tag}.merge.yaml"    # 小配置,留存供复现
        work_root.mkdir(parents=True, exist_ok=True)

        try:
            # 1) 训练(train_env)
            train_args = {
                **cfg["base_train_args"], **params,
                "model_name_or_path": cfg["base_model"],
                "output_dir": str(adapter_dir),
                "overwrite_output_dir": True,
            }
            run(conda_prefix(cfg, cfg["train_env"]) + ["llamafactory-cli", "train"]
                + render_train_args(train_args), cwd=cfg["llamafactory_dir"])

            # 2) 合并(train_env)—— 写 merge.yaml 再 export
            merge_yaml = {
                "model_name_or_path": cfg["base_model"],
                "adapter_name_or_path": str(adapter_dir),
                "export_dir": str(merged_dir),
                **cfg["merge_args"],
            }
            merge_path.write_text(yaml.safe_dump(merge_yaml, allow_unicode=True, sort_keys=False),
                                  encoding="utf-8")
            run(conda_prefix(cfg, cfg["train_env"]) + ["llamafactory-cli", "export", str(merge_path)],
                cwd=cfg["llamafactory_dir"])

            # 合并成功 → 删掉临时 adapter,只保留合并后的全精度权重(省磁盘)
            shutil.rmtree(adapter_dir, ignore_errors=True)

            # 3) 评测(eval_env):field-eval 自给自足——没预测就用同一 backend/模型自动跑 pred,
            #    再抽字段算准确率;一条命令,pred 与评测必然同一 run_dir(无不一致风险)。
            #    field_eval=false 时用 eval-vlm eval(pred+score 合一)。
            # 若 image_max_pixels 被搜索,评测端必须用**该 trial 的采样值**(与训练对齐),否则回退静态配置。
            eval_img_max = params.get("image_max_pixels", ev["image_max_pixels"])
            eval_common = [
                "--dataset", eval_ds,
                "--backend", "vllm_offline", "--vllm-model", str(merged_dir),
                "--vllm-gpu-util", str(ev["gpu_memory_utilization"]),
                "--vllm-max-model-len", str(ev["max_model_len"]),
                "--vllm-image-max-pixels", str(eval_img_max),
            ]
            sub = "field-eval" if field_eval else "eval"
            run(conda_prefix(cfg, cfg["eval_env"]) + ["eval-vlm", sub] + eval_common,
                env=eval_proc_env)

            # 4) 读指标(run_dir = <dataset>/<merged名>/vllm_offline)
            run_dir = Path(eval_ds) / merged_dir.name / "vllm_offline"
            value, metrics = read_metric(run_dir, objective_cfg, field_eval)

        except Exception as e:  # noqa: BLE001 - 单个 trial 失败不拖垮整轮搜索
            trial.set_user_attr("error", f"{type(e).__name__}: {e}")
            print(f"[trial {trial.number}] 失败,剪枝:{e}", flush=True)
            raise optuna.TrialPruned()

        # 记录 + leaderboard
        trial.set_user_attr("params", params)
        trial.set_user_attr("merged_dir", str(merged_dir))
        row = {"study": cfg["optuna"].get("study_name", "autotune"),
               "model": Path(cfg["base_model"]).name,
               "started_at": started_at,
               "trial": trial.number, "objective": round(value, 4)}
        if field_eval:
            ov = metrics.get("overall", {})
            per_field = {k: v["accuracy"] for k, v in metrics.get("per_field", {}).items()}
            trial.set_user_attr("overall", ov)
            trial.set_user_attr("per_field", per_field)
            row.update({"exact_match_rate": ov.get("exact_match_rate"),
                        "micro_accuracy": ov.get("micro_accuracy"),
                        "macro_accuracy": ov.get("macro_accuracy"),
                        "per_field": json.dumps(per_field, ensure_ascii=False)})
        row["params"] = json.dumps(params, ensure_ascii=False)
        row["merged_dir"] = str(merged_dir)
        append_leaderboard(leaderboard, row)
        print(f"[trial {trial.number}] objective={value:.4f}", flush=True)
        return value

    return objective


def create_or_load_study(cfg: dict, storage: str, sampler, direction: str) -> "optuna.Study":
    """创建/续跑 study(load_if_exists),并做共享 db 的辅助:列出全部 study + 校验签名。

    同一 sqlite 天然可容纳多个 study;共享时用 optuna-dashboard <db> 统一监控。签名
    (base_model + eval_vlm_dataset) 记在 study.user_attrs:续跑若与首次创建时不同,大声
    警告——默认 study_name='autotune' 最易把两个不同实验静默续进同一 study。
    """
    oc = cfg["optuna"]
    study = optuna.create_study(
        study_name=oc.get("study_name", "autotune"),
        storage=storage,
        sampler=sampler,
        direction=direction,
        load_if_exists=True,   # 断点续跑
    )
    # 共享 db 监控:启动时列出该 db 内全部 study。
    try:
        summaries = optuna.get_all_study_summaries(storage)
    except Exception:  # noqa: BLE001 - 只读辅助,失败不阻塞主流程
        summaries = []
    if len(summaries) > 1:
        db_path = storage.replace("sqlite:///", "")
        print(f"[autotune] storage 共享 {len(summaries)} 个 study"
              f"(统一监控: optuna-dashboard {db_path})")
        for s in summaries:
            print(f"    · {s.study_name}: trials={s.n_trials}")
    # study 签名防护。
    sig = {"base_model": cfg["base_model"], "eval_vlm_dataset": cfg["eval_vlm_dataset"]}
    prev_sig = study.user_attrs.get("_study_sig")
    if prev_sig and prev_sig != sig:
        print(f"[autotune] ⚠️ 警告:study '{study.study_name}' 已存在,但本次配置的 "
              f"base_model/eval_vlm_dataset 与首次创建时不同({prev_sig} vs {sig})——"
              f"继续会把两个不同实验混进同一 study!确认是续跑本实验再继续。", flush=True)
    if not prev_sig:
        study.set_user_attr("_study_sig", sig)
    return study


def main() -> int:
    ap = argparse.ArgumentParser(description="LlamaFactory + eval-vlm 自动超参搜索(Optuna)")
    ap.add_argument("--config", required=True, help="config.yaml 路径")
    ap.add_argument("--n-trials", type=int, default=None, help="覆盖 optuna.n_trials")
    args = ap.parse_args()

    cfg = load_config(args.config)
    validate_config(cfg)                       # 跑前校验必填键,失败早报而非每个 trial 剪枝
    work_root = Path(cfg["work_root"])
    work_root.mkdir(parents=True, exist_ok=True)

    oc = cfg["optuna"]
    storage = oc.get("storage") or f"sqlite:///{(work_root / 'study.db').as_posix()}"
    sampler = (optuna.samplers.TPESampler(seed=oc.get("seed"))
               if oc.get("sampler", "tpe") == "tpe"
               else optuna.samplers.RandomSampler(seed=oc.get("seed")))
    study = create_or_load_study(cfg, storage, sampler,
                                 cfg["objective"].get("direction", "maximize"))
    n_trials = args.n_trials if args.n_trials is not None else oc.get("n_trials", 20)
    # 续跑语义:n_trials 是**总预算**。已终结(完成/剪枝/失败)的 trial 都计入,
    # 本次只补齐到总数,而不是每次都新跑 n_trials 个(Optuna 的 optimize 本身没有这个语义)。
    done = len([t for t in study.trials if t.state.name in ("COMPLETE", "PRUNED", "FAIL")])
    remaining = max(0, n_trials - done)
    print(f"[autotune] study={oc.get('study_name')} storage={storage} "
          f"目标={cfg['objective']['metric']}({cfg['objective'].get('direction')})  "
          f"总预算 n_trials={n_trials},已终结 {done},本次补 {remaining}", flush=True)

    if remaining > 0:
        study.optimize(build_objective(cfg), n_trials=remaining)
    else:
        print(f"[autotune] 已达总预算 {n_trials},无需再跑(要加更多试验请调大 n_trials)。")

    completed = [t for t in study.trials if t.state.name == "COMPLETE"]
    print("\n" + "=" * 60)
    if not completed:
        print("[autotune] 没有成功完成的 trial(全部失败/剪枝),检查上面的报错。")
        return 1
    best = study.best_trial
    print(f"[autotune] 最优 trial #{best.number}  {cfg['objective']['metric']}={best.value:.4f}")
    print(f"[autotune] 最优超参:{json.dumps(best.params, ensure_ascii=False, indent=2)}")
    if best.user_attrs.get("per_field"):
        print(f"[autotune] 逐字段准确率:{json.dumps(best.user_attrs['per_field'], ensure_ascii=False, indent=2)}")
    print(f"[autotune] 最优合并权重:{best.user_attrs.get('merged_dir')}")
    print(f"[autotune] leaderboard:{work_root / 'leaderboard.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
