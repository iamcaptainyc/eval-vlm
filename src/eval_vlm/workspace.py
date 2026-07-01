"""工作目录模型:全局配置 + 数据集文件夹的初始化/定位 + 模板渲染。

设计:
- 机器级设置(workspace / media_root / image_strip_prefix)放全局配置
  (~/.eval_vlm/config.yaml,可用 EVAL_VLM_CONFIG 改路径),所有数据集共享。
- 每个数据集是 workspace 下一个文件夹,内含从内置模板渲染出的 config.yaml + 全部产物。
- `split --dataset <源json>` 初始化文件夹;`run/score/eval --dataset <名|路径>` 读取已存在文件夹。
"""
from __future__ import annotations

import os
import re
import sys
from importlib import resources
from pathlib import Path
from typing import Any, Optional

import yaml

# 全局配置的机器级顶层键(及默认值)。
_TOP_KEYS = ("workspace", "media_root", "image_strip_prefix")
_GLOBAL_DEFAULTS = {
    "workspace": "~/eval_vlm_workspace",
    "media_root": ".",
    "image_strip_prefix": None,
}

# split 默认值(嵌套在全局配置 split: 块下;不传 --train/--test 等时使用)。
# 同时作为 init_dataset 的内置兜底,保证唯一真源。
_SPLIT_DEFAULTS = {
    "train": 0.95,
    "test": 0.05,
    "val": 0.0,
    "seed": 42,
    "stratify_by": None,
}

# 可通过 `eval-vlm config set <key> <value>` 设置的全部键(唯一真源:校验/帮助/文档)。
# (key, 类型, 默认值, 说明)
_KEY_SPECS: tuple[tuple[str, str, Any, str], ...] = (
    ("workspace", "路径", _GLOBAL_DEFAULTS["workspace"],
     "所有数据集文件夹的父目录(split 在此创建 <数据集名>/)"),
    ("media_root", "路径", _GLOBAL_DEFAULTS["media_root"],
     "图片相对路径解析根(写进每个数据集的 config.yaml)"),
    ("image_strip_prefix", "字符串|null", _GLOBAL_DEFAULTS["image_strip_prefix"],
     "跨机训练要剥除的绝对路径前缀;本机不需要则设为 null"),
    ("split.train", "float", _SPLIT_DEFAULTS["train"],
     "默认训练集比例(不传 --train 时用)"),
    ("split.test", "float", _SPLIT_DEFAULTS["test"],
     "默认测试集比例(不传 --test 时用)"),
    ("split.val", "float", _SPLIT_DEFAULTS["val"],
     "默认验证集比例(>0 才产出 val.json;不传 --val 时用)"),
    ("split.seed", "int", _SPLIT_DEFAULTS["seed"],
     "默认随机种子,可复现(不传 --seed 时用)"),
    ("split.stratify_by", "字符串|null", _SPLIT_DEFAULTS["stratify_by"],
     "默认分层抽样字段名;null 表示不分层(不传 --stratify-by 时用)"),
)

# 不在全局配置、只能手改某个数据集文件夹内 config.yaml 的键(说明用,非可设置)。
_DATASET_LEVEL_HINTS: tuple[tuple[str, str], ...] = (
    ("data.mapping.*",
     "字段映射(messages/images/role/content 等),对齐你的数据集格式"),
    ("inference.backend",
     "推理后端 openai/vllm/mnn/cmnn/fake;切换后只读对应块设置(--backend 永久写回)"),
    ("inference.openai.* (base_url / model / api_key_env / system_prompt / "
     "max_concurrency / max_tokens / temperature / request_timeout / max_retries / image_detail)",
     "openai/vllm 后端设置(--base-url/--model 永久写回 openai.base_url/model)"),
    ("inference.mnn.* (config_path / image_max_side / max_tokens / "
     "repetition_penalty / frequency_penalty / presence_penalty / penalty_window / "
     "temperature / top_k / top_p / sampler_config)",
     "mnn 后端设置(--mnn-config/--mnn-image-max-side 永久写回);采样项 value-gated 防小模型满屏换行退化;产物目录名取 config_path 所在目录名"),
    ("inference.cmnn.* (config_path / num_workers / batch_size / image_max_side / "
     "max_tokens / repetition_penalty / … / sampler_config)",
     "cmnn 后端设置(C++ 原生库批量推理,功能同 mnn;--cmnn-config/--cmnn-num-workers/--cmnn-batch-size 永久写回)"),
    ("eval.targets / eval.context",
     "评测哪些 assistant 轮(all|last)、用什么上下文(rollout|gold)"),
    ("scoring.scorer / scoring.turn_scorers",
     "评分器与逐轮评分器(scorer 可用 --scorer 临时覆盖)"),
    ("split.train_out / val_out / test_out",
     "三份产物的输出路径(可用 --train-out/--val-out/--test-out 临时覆盖)"),
)

_DEFAULT_GLOBAL_TEXT = """\
# eval_vlm 全局配置(机器级,所有数据集共享)
# 路径:EVAL_VLM_CONFIG 环境变量优先,否则 ~/.eval_vlm/config.yaml
# 用 `eval-vlm config set <key> <value>` 修改,或直接手改本文件。

workspace: ~/eval_vlm_workspace   # 所有数据集文件夹的父目录(split 在此创建 <数据集名>/)
media_root: .                     # 图片相对路径解析根(写进每个数据集的 config.yaml)
image_strip_prefix: null          # 跨机训练绝对前缀,本机不需要则 null

# split 默认比例/参数:不传 --train/--test/--val/--seed/--stratify-by 时用这里的值。
# 命令行参数优先级更高。改法:eval-vlm config set split.train 0.9
split:
  train: 0.95                     # 训练集比例
  test: 0.05                      # 测试集比例
  val: 0.0                        # 验证集比例(>0 才产出 val.json)
  seed: 42                        # 随机种子(可复现)
  stratify_by: null               # 分层抽样字段名(默认不分层)
"""


# ---------------------------------------------------------------------------
# 全局配置
# ---------------------------------------------------------------------------
def global_config_path() -> Path:
    """全局配置文件路径:EVAL_VLM_CONFIG 优先,否则 ~/.eval_vlm/config.yaml。"""
    env = os.environ.get("EVAL_VLM_CONFIG")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".eval_vlm" / "config.yaml"


def init_global_config(force: bool = False) -> Path:
    """写入带注释的默认全局配置(已存在且非 force 则不动)。返回路径。"""
    path = global_config_path()
    if path.exists() and not force:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_DEFAULT_GLOBAL_TEXT, encoding="utf-8")
    return path


def load_global_config() -> dict[str, Any]:
    """读取全局配置;缺失时自动生成默认并提示,再读回。缺键用默认值兜底。"""
    path = global_config_path()
    if not path.exists():
        init_global_config()
        print(
            f"[eval_vlm] 已生成默认全局配置: {path}\n"
            f"           请按需设置 workspace / media_root("
            f"`eval-vlm config set workspace <dir>`)。",
            file=sys.stderr,
        )
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    cfg: dict[str, Any] = dict(_GLOBAL_DEFAULTS)
    for k in _TOP_KEYS:
        if k in raw:
            cfg[k] = raw[k]
    # 嵌套 split 默认:缺键用内置默认;数值容错(手改成字符串也能用)。
    split = dict(_SPLIT_DEFAULTS)
    raw_split = raw.get("split")
    if isinstance(raw_split, dict):
        for k in _SPLIT_DEFAULTS:
            if k in raw_split:
                try:
                    split[k] = _coerce_split(k, raw_split[k])
                except (ValueError, TypeError):
                    split[k] = _SPLIT_DEFAULTS[k]
    cfg["split"] = split
    return cfg


def _all_keys() -> tuple[str, ...]:
    """可设置键的完整清单(顶层 + 嵌套 split.*),供校验与帮助文本。"""
    return tuple(spec[0] for spec in _KEY_SPECS)


def describe_settable_keys() -> str:
    """渲染「可设置键」清单(类型/默认/说明)+「不可设置(数据集级)」清单。

    供 `eval-vlm config keys` 打印,让用户一眼看清哪些键能全局设、哪些得手改数据集 config.yaml。
    """
    kw = max(len(spec[0]) for spec in _KEY_SPECS)
    out = ["可通过 `eval-vlm config set <key> <value>` 设置的全局键:", ""]
    for key, typ, default, desc in _KEY_SPECS:
        out.append(f"  {key.ljust(kw)}  ({typ}, 默认 {_yaml_scalar(default)})")
        out.append(f"  {' ' * kw}      {desc}")
    out.append("")
    out.append("不属于全局配置(单个数据集独有,需手改该数据集文件夹内的 config.yaml):")
    for name, desc in _DATASET_LEVEL_HINTS:
        out.append(f"  {name}")
        out.append(f"      {desc}")
    out.append("")
    out.append("提示:split 比例命令行参数 > 全局 split.* > 内置默认;pred/score/eval 的 "
               "--base-url/--model/--scorer 等会永久写回该数据集 config.yaml(用户参数优先且持久化)。")
    return "\n".join(out)


def _coerce_top(key: str, value: Optional[str]) -> Any:
    """顶层键类型转换:image_strip_prefix 允许 None,其余转字符串。"""
    if key == "image_strip_prefix":
        return value                           # None 或字符串
    if value is None:
        raise ValueError(f"{key} 不能设为空")
    return str(value)


def _coerce_split(child: str, value: Optional[str]) -> Any:
    """split 子键类型转换:train/test/val -> float, seed -> int, stratify_by -> str|None。"""
    if child == "stratify_by":
        return value                           # None 或字符串
    if value is None:
        raise ValueError(f"split.{child} 不能设为空")
    if child == "seed":
        return int(value)
    return float(value)                        # train / test / val


def set_global_value(key: str, value: Optional[str]) -> Path:
    """设置一个全局配置键(保留注释/其余行)。

    支持顶层键(workspace/media_root/image_strip_prefix)与嵌套 split.<子键>
    (split.train/test/val/seed/stratify_by)。命令行传入的字符串会按键类型转换。
    """
    init_global_config()                       # 确保文件存在(含默认 split 块)
    path = global_config_path()
    text = path.read_text(encoding="utf-8")

    if "." in key:
        parent, child = key.split(".", 1)
        if parent != "split" or child not in _SPLIT_DEFAULTS:
            raise KeyError(f"未知全局配置键: {key}(可选: {', '.join(_all_keys())})")
        text = _set_nested_value(text, "split", child, _coerce_split(child, value))
    else:
        if key not in _TOP_KEYS:
            raise KeyError(f"未知全局配置键: {key}(可选: {', '.join(_all_keys())})")
        text = _update_yaml_value(text, key, _coerce_top(key, value))

    path.write_text(text, encoding="utf-8")
    return path


def _update_yaml_value(text: str, key: str, value: Any) -> str:
    """整行替换某顶层键的值,保留行尾内联注释;缺该键则追加一行。"""
    literal = _yaml_scalar(value)
    pattern = re.compile(rf"^({re.escape(key)}:[ \t]*)([^#\n]*?)([ \t]*#.*)?$", re.MULTILINE)

    def repl(m: re.Match) -> str:
        comment = m.group(3) or ""
        return f"{m.group(1)}{literal}{comment}"

    new_text, n = pattern.subn(repl, text)
    if n == 0:
        sep = "" if text.endswith("\n") else "\n"
        new_text = f"{text}{sep}{key}: {literal}\n"
    return new_text


def _set_nested_value(text: str, parent: str, child: str, value: Any) -> str:
    """替换 `parent:` 块下缩进子键 `child:` 的值,保留行尾内联注释。

    block 不存在则在文末追加 `parent:\\n  child: ...`;block 存在但缺该子键则
    插在 parent 行之后。仅依赖缩进识别 block 范围,匹配本程序生成的全局配置。
    """
    literal = _yaml_scalar(value)
    lines = text.splitlines(keepends=True)
    parent_re = re.compile(rf"^{re.escape(parent)}:[ \t]*(#.*)?\r?\n?$")
    child_re = re.compile(
        rf"^([ \t]+){re.escape(child)}:[ \t]*([^#\n\r]*?)([ \t]*#[^\n\r]*)?(\r?\n?)$"
    )

    parent_idx = next((i for i, ln in enumerate(lines) if parent_re.match(ln)), None)
    if parent_idx is None:
        sep = "" if text.endswith("\n") else "\n"
        return f"{text}{sep}{parent}:\n  {child}: {literal}\n"

    j = parent_idx + 1
    while j < len(lines):
        ln = lines[j]
        if ln.strip() == "":                   # 块内/块后空行,跳过
            j += 1
            continue
        if not re.match(r"^[ \t]", ln):         # 顶到非缩进行 -> 块结束
            break
        m = child_re.match(ln)
        if m:
            indent, comment, eol = m.group(1), m.group(3) or "", m.group(4) or "\n"
            lines[j] = f"{indent}{child}: {literal}{comment}{eol}"
            return "".join(lines)
        j += 1

    lines.insert(parent_idx + 1, f"  {child}: {literal}\n")   # 块内缺该子键 -> 插入
    return "".join(lines)


# ---------------------------------------------------------------------------
# 路径解析
# ---------------------------------------------------------------------------
def resolve_workspace(cli_override: Optional[str], global_cfg: dict[str, Any]) -> Path:
    """工作目录:命令行 --workspace 优先,否则全局配置 workspace。"""
    raw = cli_override if cli_override else global_cfg.get("workspace", _GLOBAL_DEFAULTS["workspace"])
    return Path(str(raw)).expanduser().resolve()


def resolve_dataset_dir(name_or_path: str, workspace: Path) -> Path:
    """run/score/eval:把 --dataset 解析成已存在的数据集文件夹。

    顺序:① 若本身是已存在目录 -> 直接用;② 否则当作名字 -> workspace/<名>。
    """
    p = Path(name_or_path).expanduser()
    if p.is_dir():
        return p.resolve()
    cand = workspace / name_or_path
    if cand.is_dir():
        return cand.resolve()
    raise FileNotFoundError(
        f"未找到数据集 '{name_or_path}'(既不是已存在目录,workspace 下也没有 "
        f"{cand})。请先 eval-vlm split --dataset <源json> 初始化。"
    )


# ---------------------------------------------------------------------------
# 模板渲染 + 数据集初始化
# ---------------------------------------------------------------------------
def _yaml_scalar(value: Any) -> str:
    """把 Python 值渲染成合法 YAML 标量(字符串用单引号,Windows 反斜杠安全)。"""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    s = str(value)
    return "'" + s.replace("'", "''") + "'"


def render_template(values: dict[str, Any]) -> str:
    """读取内置统一模板,把 {{KEY}} 占位符替换为渲染后的 YAML 标量。

    全部命令(split / run / score / eval / pred)共用同一个 config.template.yaml;
    各命令只填自己关心的占位符,其余取默认(与本命令无关的段是惰性的,无副作用)。
    仅替换标量占位符(列表如多轮 pred.template 在模板里静态写,不经此函数)。
    """
    text = (
        resources.files("eval_vlm")
        .joinpath("templates/config.template.yaml")
        .read_text(encoding="utf-8")
    )
    for key, val in values.items():
        text = text.replace("{{" + key + "}}", _yaml_scalar(val))
    return text


def init_dataset(
    source_json: str,
    workspace: Path,
    *,
    name: Optional[str] = None,
    split_overrides: Optional[dict[str, Any]] = None,
    split_defaults: Optional[dict[str, Any]] = None,
    media_root: Any = ".",
    image_strip_prefix: Any = None,
    force: bool = False,
) -> Path:
    """初始化一个数据集文件夹:建目录 + 从模板渲染 config.yaml。返回文件夹路径。

    split 取值优先级:split_overrides(命令行)> split_defaults(全局配置)
    > _SPLIT_DEFAULTS(内置兜底)。不在此执行 split(由调用方拿到 folder 后
    load_dataset_config + split_dataset)。
    """
    src = Path(source_json).expanduser().resolve()
    if not src.is_file():
        raise FileNotFoundError(f"源数据集不是文件: {src}")

    ds_name = name or src.stem
    folder = (workspace / ds_name).resolve()
    config_path = folder / "config.yaml"
    if config_path.exists() and not force:
        raise FileExistsError(
            f"数据集已存在: {folder}(用 --force 重建,将覆盖该文件夹内的 config.yaml)"
        )

    folder.mkdir(parents=True, exist_ok=True)
    # 内置兜底 <- 全局默认 <- 命令行覆盖,逐层合并(仅取认识的子键)。
    sp = dict(_SPLIT_DEFAULTS)
    for src_dict in (split_defaults, split_overrides):
        if src_dict:
            sp.update({k: src_dict[k] for k in _SPLIT_DEFAULTS if k in src_dict})
    values = {
        "RUN_NAME": ds_name,
        "OUTPUT_DIR": str(workspace),
        "SOURCE": str(src),
        "MEDIA_ROOT": media_root,
        "IMAGE_STRIP_PREFIX": image_strip_prefix,
        "TRAIN": sp["train"],
        "TEST": sp["test"],
        "VAL": sp["val"],
        "SEED": sp["seed"],
        "STRATIFY_BY": sp["stratify_by"],
    }
    config_path.write_text(render_template(values), encoding="utf-8")
    return folder


def init_pred_config(
    out_dir: Path,
    datadir: Path,
    global_cfg: dict[str, Any],
    *,
    force: bool = False,
) -> Path:
    """为 pred 在输出文件夹生成 config.yaml(统一模板,只是不评分)。返回其路径。

    与 init_dataset 共用同一个模板:media_root 钉到图片文件夹,image_strip_prefix
    取自全局配置;pred 无源 JSON 故 source 留空,split 段填默认值(对 pred 惰性、无副作用)。
    已存在且非 force 时不动(保留用户手改)。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    config_path = out_dir / "config.yaml"
    if config_path.exists() and not force:
        return config_path
    values = {
        "RUN_NAME": out_dir.name,
        "OUTPUT_DIR": str(out_dir.parent),
        "SOURCE": "",                              # pred 无源数据集 JSON
        "MEDIA_ROOT": str(datadir),
        "IMAGE_STRIP_PREFIX": global_cfg.get("image_strip_prefix"),
        # split 段对 pred 无意义,仅为占位渲染(留默认值,pred 永不读取)。
        "TRAIN": _SPLIT_DEFAULTS["train"],
        "TEST": _SPLIT_DEFAULTS["test"],
        "VAL": _SPLIT_DEFAULTS["val"],
        "SEED": _SPLIT_DEFAULTS["seed"],
        "STRATIFY_BY": _SPLIT_DEFAULTS["stratify_by"],
    }
    config_path.write_text(render_template(values), encoding="utf-8")
    return config_path


def _set_dotted_value(text: str, dotted_key: str, value: Any) -> str:
    """把任意层级点号键(如 inference.openai.base_url)的值写回 YAML,保留行尾注释。

    依赖本程序生成配置的固定缩进(每层 2 空格):逐层定位 `parent:` 块头并收窄
    搜索范围,最后在最内层块里整行替换叶子键的值。中途缺失的块/键会自动按缩进插入,
    因此即便用户精简过 config.yaml(删掉某段)也能补齐。单层键退化为顶层替换。
    """
    parts = dotted_key.split(".")
    if len(parts) == 1:
        return _update_yaml_value(text, parts[0], value)

    literal = _yaml_scalar(value)
    lines = text.splitlines(keepends=True)
    start, end, indent = 0, len(lines), 0

    # 逐层下钻定位父块,收窄 [start, end) 到该块的行范围。
    for depth, part in enumerate(parts[:-1]):
        header_re = re.compile(rf"^{' ' * indent}{re.escape(part)}:[ \t]*(#.*)?\r?\n?$")
        idx = next((i for i in range(start, end) if header_re.match(lines[i])), None)
        if idx is None:
            # 该层块不存在:从当前缩进起,把「剩余各层块头 + 叶子」整段补进**当前父块末尾**
            # (end:顶层块缺失时 end=len(lines) 即文末;子块被手删时 end=父块尾,补回父块内)。
            remaining = parts[depth:]
            tail_lines = [
                f"{' ' * (indent + 2 * k)}{seg}:\n" for k, seg in enumerate(remaining[:-1])
            ]
            tail_lines.append(
                f"{' ' * (indent + 2 * (len(remaining) - 1))}{remaining[-1]}: {literal}\n"
            )
            # 插入点前一行若无换行结尾(文件末尾无换行),补一个,避免与新块连成一行。
            if end > 0 and lines[end - 1] != "" and not lines[end - 1].endswith("\n"):
                lines[end - 1] = lines[end - 1] + "\n"
            lines[end:end] = tail_lines
            return "".join(lines)
        # 块体 = 紧随块头、缩进比块头更深的连续行(空行跳过)。
        j = idx + 1
        while j < end:
            ln = lines[j]
            if ln.strip() == "":
                j += 1
                continue
            cur_indent = len(ln) - len(ln.lstrip(" "))
            if cur_indent <= indent:
                break
            j += 1
        start, end, indent = idx + 1, j, indent + 2

    # 在最内层块里替换(或插入)叶子键。
    child = parts[-1]
    child_re = re.compile(
        rf"^({' ' * indent}){re.escape(child)}:[ \t]*([^#\n\r]*?)([ \t]*#[^\n\r]*)?(\r?\n?)$"
    )
    for i in range(start, end):
        m = child_re.match(lines[i])
        if m:
            comment, eol = m.group(3) or "", m.group(4) or "\n"
            lines[i] = f"{m.group(1)}{child}: {literal}{comment}{eol}"
            return "".join(lines)
    lines.insert(end, f"{' ' * indent}{child}: {literal}\n")
    return "".join(lines)


def set_dataset_value(folder: Path, dotted_key: str, value: Any) -> Path:
    """把一个键的值永久写回某数据集文件夹的 config.yaml(保留注释)。

    用于「用户在命令行显式提供的设置(如 --model/--base-url)应永久生效」:
    写回后,该数据集的后续命令都读到新值,实现「用户参数优先且持久化」。
    支持任意层级点号嵌套键(如 inference.openai.model / inference.mnn.config_path /
    scoring.scorer)与顶层键。块/子键缺失时会自动插入(见 _set_dotted_value)。
    """
    config_path = Path(folder) / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"数据集文件夹缺少 config.yaml: {config_path}")
    text = config_path.read_text(encoding="utf-8")
    text = _set_dotted_value(text, dotted_key, value)
    config_path.write_text(text, encoding="utf-8")
    return config_path
