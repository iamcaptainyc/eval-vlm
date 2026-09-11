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
_TOP_KEYS = ("workspace", "media_root", "image_strip_prefix",
             "train_out_dir", "val_out_dir", "test_out_dir",
             "hf_models_dir", "mnn_models_dir", "llamacpp_models_dir")
_GLOBAL_DEFAULTS = {
    "workspace": "~/eval_vlm_workspace",
    "media_root": ".",
    "image_strip_prefix": None,
    # 光杆旗标 --train-out/--val-out/--test-out 的默认落地目录:设了目录后,
    # `eval-vlm split -d xxx.json --train-out`(不带路径)= 把 train 产物写到
    # <该目录>/xxx_train.json(xxx=数据集名)。留空则该旗标只能带完整路径用。
    "train_out_dir": None,
    "val_out_dir": None,
    "test_out_dir": None,
    "hf_models_dir": None,
    "mnn_models_dir": None,
    "llamacpp_models_dir": None,
}

# 允许显式设为 null 的顶层键(其余顶层键不能为空)。
_TOP_NULLABLE = frozenset(
    {"image_strip_prefix", "train_out_dir", "val_out_dir", "test_out_dir",
     "hf_models_dir", "mnn_models_dir", "llamacpp_models_dir"}
)

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
    ("train_out_dir", "路径|null", _GLOBAL_DEFAULTS["train_out_dir"],
     "光杆旗标 --train-out 的落地目录;设了后该旗标 = <目录>/<数据集名>_train.json"),
    ("val_out_dir", "路径|null", _GLOBAL_DEFAULTS["val_out_dir"],
     "光杆旗标 --val-out 的落地目录;设了后该旗标 = <目录>/<数据集名>_val.json"),
    ("test_out_dir", "路径|null", _GLOBAL_DEFAULTS["test_out_dir"],
     "光杆旗标 --test-out 的落地目录;设了后该旗标 = <目录>/<数据集名>_test.json"),
    ("hf_models_dir", "路径|null", _GLOBAL_DEFAULTS["hf_models_dir"],
     "HuggingFace/vLLM/OpenAI 模型权重存储目录(WebUI 模型浏览器)"),
    ("mnn_models_dir", "路径|null", _GLOBAL_DEFAULTS["mnn_models_dir"],
     "MNN 模型存储根目录(WebUI 模型浏览器)"),
    ("llamacpp_models_dir", "路径|null", _GLOBAL_DEFAULTS["llamacpp_models_dir"],
     "llama.cpp GGUF 模型存储根目录(含多模态成对模型文件夹)"),
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
     "推理后端 openai/vllm/mnn/fake;切换后只读对应块设置(--backend 永久写回)"),
    ("inference.openai.* (base_url / model / api_key_env / system_prompt / "
     "max_concurrency / max_tokens / temperature / request_timeout / max_retries / image_detail)",
     "openai/vllm 后端设置(--base-url/--model 永久写回 openai.base_url/model)"),
    ("inference.mnn.* (config_path / image_max_pixels / image_min_pixels / "
     "image_max_side / system_prompt / max_tokens / "
     "repetition_penalty / frequency_penalty / presence_penalty / penalty_window / "
     "temperature / top_k / top_p / sampler_config)",
     "mnn 后端设置(--mnn-config/--mnn-image-max-side 永久写回);图片预处理项对齐 LlamaFactory 训练规则;采样项 value-gated 防小模型满屏换行退化;产物目录名取 config_path 所在目录名"),
    ("eval.targets / eval.context",
     "评测哪些 assistant 轮(all|last|first|数字=第N轮)、用什么上下文(rollout|gold)"),
    ("scoring.scorer / scoring.turn_scorers",
     "评分器与逐轮评分器(scorer 可用 --scorer 临时覆盖)"),
    ("split.train_out / val_out / test_out",
     "三份产物的输出路径(可用 --train-out/--val-out/--test-out 临时覆盖:带路径=该路径;"
     "光杆旗标=落到全局 train_out_dir/val_out_dir/test_out_dir 下并自动命名 <数据集名>_<份>.json)"),
)

_DEFAULT_GLOBAL_TEXT = """\
# eval_vlm 全局配置(机器级,所有数据集共享)
# 路径:EVAL_VLM_CONFIG 环境变量优先,否则 ~/.eval_vlm/config.yaml
# 用 `eval-vlm config set <key> <value>` 修改,或直接手改本文件。

workspace: ~/eval_vlm_workspace   # 所有数据集文件夹的父目录(split 在此创建 <数据集名>/)
media_root: .                     # 图片相对路径解析根(写进每个数据集的 config.yaml)
image_strip_prefix: null          # 跨机训练绝对前缀,本机不需要则 null

# 光杆旗标 --train-out/--val-out/--test-out 的默认落地目录(不带路径时用)。
# 设了目录后:eval-vlm split -d xxx.json --train-out
#   -> 把 train 产物写到 <train_out_dir>/xxx_train.json(xxx=数据集名)。
# 留 null 则这些旗标必须带完整路径。改法:eval-vlm config set train_out_dir <目录>
train_out_dir: null               # 例:/root/autodl-tmp/LlamaFactory/data
val_out_dir: null
test_out_dir: null

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


def _coerce_top(key: str, value: Optional[Any]) -> Any:
    """顶层键类型转换:可空键(前缀/各 *_out_dir/模型目录)允许 None 或原样(字符串/列表)，其余转字符串。"""
    if key in _TOP_NULLABLE:
        if value is None:
            return None
        if isinstance(value, str) and value.strip().lower() in ("", "null", "none"):
            return None
        return value                           # None 或字符串或列表
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
    """整行替换某顶层键的值,保留行尾内联注释;支持单行标量和多行/列表;缺该键则追加一行。"""
    literal = _yaml_scalar(value)
    lines = text.splitlines(keepends=True)
    key_re = re.compile(rf"^{re.escape(key)}:[ \t]*(.*?)([ \t]*#.*)?(\r?\n?)$")

    found_idx = None
    comment = ""
    eol = "\n"
    for idx, line in enumerate(lines):
        m = key_re.match(line)
        if m:
            found_idx = idx
            comment = m.group(2) or ""
            eol = m.group(3) or "\n"
            break

    if found_idx is None:
        sep = "" if text.endswith("\n") else "\n"
        return f"{text}{sep}{key}: {literal}\n"

    # 如果找到，检查后续行是否有缩进内容（例如旧的 YAML 列表项）
    end_idx = found_idx + 1
    while end_idx < len(lines):
        ln = lines[end_idx]
        if re.match(r"^[ \t]+", ln):
            end_idx += 1
        elif ln.strip() == "":
            if end_idx + 1 < len(lines) and re.match(r"^[ \t]+", lines[end_idx + 1]):
                end_idx += 1
            else:
                break
        else:
            break

    new_line = f"{key}: {literal}{comment}{eol}"
    lines[found_idx:end_idx] = [new_line]
    return "".join(lines)


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


def resolve_split_out(
    value: Optional[str], kind: str, ds_name: str, global_cfg: dict[str, Any]
) -> Optional[str]:
    """把 --train-out/--val-out/--test-out 的取值解析成最终输出路径(或 None)。

    kind ∈ {"train","val","test"}。value 语义(来自 argparse nargs="?"):
      None -> 未提供该参数,不覆盖(返回 None,落默认 <数据集>/<份>.json)。
      ""   -> 光杆旗标(如 `--train-out`,不带路径):落到全局 <kind>_out_dir 下,
              自动命名 <数据集名>_<kind>.json(消除手工改名 + 复制到 LlamaFactory data/)。
      其它  -> 用户显式给的完整路径,原样返回(向后兼容旧用法)。
    """
    if value is None:
        return None
    if value != "":
        return value                           # 显式路径:原样
    key = f"{kind}_out_dir"
    out_dir = global_cfg.get(key)
    if not out_dir:
        raise ValueError(
            f"--{kind}-out 作为旗标(不带路径)使用需先设置全局 {key}:\n"
            f"       eval-vlm config set {key} <目录>\n"
            f"    或直接给出完整路径:--{kind}-out <路径>"
        )
    return str(Path(str(out_dir)).expanduser() / f"{ds_name}_{kind}.json")


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
    """把 Python 值渲染成合法 YAML 标量(字符串用单引号,Windows 反斜杠安全;列表格式化为行内数组)。"""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_yaml_scalar(x) for x in value) + "]"
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


def _normalize_dirs(raw_val: Any) -> list[Path]:
    """把各种可能的多目录输入转换为 Path 列表。
    支持:
    - None / ""
    - Path 对象
    - 列表/元组: [path1, path2, ...]
    - 字符串: 支持分号 ';', 换行 '\n', 逗号 ',' 分隔的多个路径
    """
    if not raw_val:
        return []
    items: list[str] = []
    if isinstance(raw_val, (list, tuple)):
        for item in raw_val:
            if item:
                items.append(str(item))
    elif isinstance(raw_val, Path):
        items.append(str(raw_val))
    elif isinstance(raw_val, str):
        parts = re.split(r"[;\n\r,]+", raw_val)
        items.extend(p.strip() for p in parts if p.strip())
    else:
        items.append(str(raw_val))

    paths: list[Path] = []
    seen: set[str] = set()
    for item in items:
        clean = item.strip().strip("'\"")
        if not clean or clean.lower() in ("null", "none"):
            continue
        try:
            p = Path(clean).expanduser().resolve()
            p_str = str(p).lower() if os.name == "nt" else str(p)
            if p_str not in seen:
                seen.add(p_str)
                paths.append(p)
        except Exception:
            pass
    return paths


def _check_and_add_hf(
    d: Path,
    root: Path,
    out: list[dict[str, Any]],
    seen_paths: set[str],
    prefix: str = "",
) -> bool:
    """检查目录 d 是否为 HF 模型。"""
    markers = ("config.json", "model.safetensors", "pytorch_model.bin", "tokenizer.json")
    try:
        if any((d / m).exists() for m in markers):
            p_key = str(d.resolve()).lower() if os.name == "nt" else str(d.resolve())
            if p_key in seen_paths:
                return True
            seen_paths.add(p_key)
            base_name = d.name if d == root else d.relative_to(root).as_posix()
            disp_name = f"{prefix}{base_name}" if prefix else base_name
            out.append({
                "name": disp_name,
                "path": str(d),
                "type": "hf",
            })
            return True
    except Exception:
        pass
    return False


def _check_and_add_mnn(
    d: Path,
    root: Path,
    out: list[dict[str, Any]],
    seen_paths: set[str],
    prefix: str = "",
) -> bool:
    """检查目录 d 是否为 MNN 模型目录 (含 config.json 或 *.mnn)。"""
    try:
        cfg_file = d / "config.json"
        has_mnn = any(f.suffix.lower() == ".mnn" for f in d.iterdir() if f.is_file())
        if cfg_file.exists():
            p_key = str(cfg_file.resolve()).lower() if os.name == "nt" else str(cfg_file.resolve())
            if p_key in seen_paths:
                return True
            seen_paths.add(p_key)
            base_name = d.name if d == root else d.relative_to(root).as_posix()
            disp_name = f"{prefix}{base_name}" if prefix else base_name
            out.append({
                "name": disp_name,
                "path": str(cfg_file),
                "type": "mnn",
            })
            return True
        elif has_mnn and d != root:
            p_key = str(d.resolve()).lower() if os.name == "nt" else str(d.resolve())
            if p_key in seen_paths:
                return True
            seen_paths.add(p_key)
            base_name = d.relative_to(root).as_posix()
            disp_name = f"{prefix}{base_name}" if prefix else base_name
            out.append({
                "name": disp_name,
                "path": str(d),
                "type": "mnn",
            })
            return True
    except Exception:
        pass
    return False


def scan_local_models(
    hf_dir: Optional[Any] = None,
    mnn_dir: Optional[Any] = None,
    llamacpp_dir: Optional[Any] = None,
) -> dict[str, list[dict[str, Any]]]:
    """探测并枚举本地存储的 HF/vLLM 模型、MNN 模型与 llama.cpp GGUF 模型。

    支持配置多个目录(列表或换行/分号/逗号分隔的字符串)。
    llama.cpp 模型按照模型子文件夹组织(例如 <llamacpp_models_dir>/<A>/):
      内部包含 A_Q4_K_M.gguf / A_bf16_mmproj.gguf 等,系统自动提取父文件夹名 A 作为模型名,
      并智能成对匹配主模型与 mmproj 投影器。
    返回:
      {
        "hf_models": [{"name": "...", "path": "...", "type": "hf"}],
        "mnn_models": [{"name": "...", "path": "...", "type": "mnn"}],
        "llamacpp_models": [
            {
                "name": "A (A_Q4_K_M.gguf + A_bf16_mmproj.gguf)",
                "model_name": "A",
                "path": ".../A/A_Q4_K_M.gguf",
                "mmproj_path": ".../A/A_bf16_mmproj.gguf",
                "type": "llamacpp"
            }
        ]
      }
    """
    hf_models: list[dict[str, Any]] = []
    mnn_models: list[dict[str, Any]] = []
    llamacpp_models: list[dict[str, Any]] = []
    seen_hf_paths: set[str] = set()
    seen_mnn_paths: set[str] = set()
    seen_llamacpp_pairs: set[str] = set()

    # 1. 扫描 HF / vLLM / transformers 模型
    hf_roots = _normalize_dirs(hf_dir)
    hf_multi = len(hf_roots) > 1
    for hp in hf_roots:
        try:
            if hp.is_dir():
                prefix = f"[{hp.name}] " if hf_multi else ""
                if not _check_and_add_hf(hp, hp, hf_models, seen_hf_paths, prefix):
                    for p1 in sorted(hp.iterdir()):
                        if not p1.is_dir():
                            continue
                        if _check_and_add_hf(p1, hp, hf_models, seen_hf_paths, prefix):
                            continue
                        for p2 in sorted(p1.iterdir()):
                            if p2.is_dir():
                                _check_and_add_hf(p2, hp, hf_models, seen_hf_paths, prefix)
        except Exception:
            pass

    # 2. 扫描 MNN 模型
    mnn_roots = _normalize_dirs(mnn_dir)
    mnn_multi = len(mnn_roots) > 1
    for mp in mnn_roots:
        try:
            if mp.is_dir():
                prefix = f"[{mp.name}] " if mnn_multi else ""
                if not _check_and_add_mnn(mp, mp, mnn_models, seen_mnn_paths, prefix):
                    for p1 in sorted(mp.iterdir()):
                        if p1.is_file() and p1.suffix.lower() == ".mnn":
                            p_key = str(p1.resolve()).lower() if os.name == "nt" else str(p1.resolve())
                            if p_key not in seen_mnn_paths:
                                seen_mnn_paths.add(p_key)
                                disp_name = f"{prefix}{p1.name}" if prefix else p1.name
                                mnn_models.append({
                                    "name": disp_name,
                                    "path": str(p1),
                                    "type": "mnn",
                                    "model_name": p1.stem,
                                })
                        elif p1.is_dir():
                            if _check_and_add_mnn(p1, mp, mnn_models, seen_mnn_paths, prefix):
                                continue
                            for p2 in sorted(p1.iterdir()):
                                if p2.is_file() and p2.suffix.lower() == ".mnn":
                                    p_key = str(p2.resolve()).lower() if os.name == "nt" else str(p2.resolve())
                                    if p_key not in seen_mnn_paths:
                                        seen_mnn_paths.add(p_key)
                                        disp_name = f"{prefix}{p1.name}/{p2.name}" if prefix else f"{p1.name}/{p2.name}"
                                        mnn_models.append({
                                            "name": disp_name,
                                            "path": str(p2),
                                            "type": "mnn",
                                            "model_name": p1.name,
                                        })
                                elif p2.is_dir():
                                    _check_and_add_mnn(p2, mp, mnn_models, seen_mnn_paths, prefix)
        except Exception:
            pass

    # 3. 扫描 llama.cpp GGUF 模型 (子文件夹成对识别)
    llamacpp_roots = _normalize_dirs(llamacpp_dir)
    llamacpp_multi = len(llamacpp_roots) > 1
    for lp in llamacpp_roots:
        try:
            if not lp.is_dir():
                continue
            prefix = f"[{lp.name}] " if llamacpp_multi else ""

            # 收集每个包含 gguf 的子文件夹
            # 扫描深度: 支持 lp 目录直接包含模型子文件夹(如 lp/A/*.gguf)
            # 以及两层(如 lp/group/A/*.gguf)
            candidate_dirs: list[Path] = []
            for child in sorted(lp.iterdir()):
                if child.is_dir():
                    candidate_dirs.append(child)
                    for subchild in sorted(child.iterdir()):
                        if subchild.is_dir():
                            candidate_dirs.append(subchild)

            # 遍历候选模型文件夹
            for mdir in candidate_dirs:
                try:
                    gguf_files = [f for f in mdir.iterdir() if f.is_file() and f.suffix.lower() == ".gguf"]
                    if not gguf_files:
                        continue

                    # 分离 mmproj 投影器与语言主干模型
                    mmproj_files = [f for f in gguf_files if "mmproj" in f.name.lower()]
                    main_files = [f for f in gguf_files if "mmproj" not in f.name.lower()]

                    # 模型标识 A 取自文件夹名
                    model_id = mdir.name
                    rel_dir = mdir.relative_to(lp).as_posix()
                    display_folder = f"{prefix}{rel_dir}" if prefix else rel_dir

                    # 选出最匹配的 mmproj (若有多个优先取非中间临时文件或第一个)
                    default_mmproj = str(mmproj_files[0].resolve()) if mmproj_files else None
                    default_mmproj_name = mmproj_files[0].name if mmproj_files else None

                    if main_files:
                        for mf in main_files:
                            m_key = f"{str(mf.resolve())}::{default_mmproj or ''}".lower()
                            if m_key in seen_llamacpp_pairs:
                                continue
                            seen_llamacpp_pairs.add(m_key)

                            if default_mmproj_name:
                                disp_name = f"{display_folder} ({mf.name} + {default_mmproj_name})"
                            else:
                                disp_name = f"{display_folder} ({mf.name})"

                            llamacpp_models.append({
                                "name": disp_name,
                                "model_name": model_id,
                                "path": str(mf.resolve()),
                                "mmproj_path": default_mmproj,
                                "type": "llamacpp",
                            })
                    elif mmproj_files:
                        # 仅有 mmproj 没有独立语言模型的情况
                        for mpf in mmproj_files:
                            m_key = f"::{str(mpf.resolve())}".lower()
                            if m_key in seen_llamacpp_pairs:
                                continue
                            seen_llamacpp_pairs.add(m_key)
                            llamacpp_models.append({
                                "name": f"{display_folder} ({mpf.name}) [仅投影器]",
                                "model_name": model_id,
                                "path": "",
                                "mmproj_path": str(mpf.resolve()),
                                "type": "llamacpp",
                            })
                except Exception:
                    pass

            # 也支持根目录平铺的单个 .gguf
            for root_f in lp.iterdir():
                if root_f.is_file() and root_f.suffix.lower() == ".gguf":
                    is_mm = "mmproj" in root_f.name.lower()
                    m_key = f"{str(root_f.resolve())}::".lower() if not is_mm else f"::{str(root_f.resolve())}".lower()
                    if m_key not in seen_llamacpp_pairs:
                        seen_llamacpp_pairs.add(m_key)
                        disp_name = f"{prefix}{root_f.name}" if prefix else root_f.name
                        llamacpp_models.append({
                            "name": disp_name,
                            "model_name": root_f.stem,
                            "path": str(root_f.resolve()) if not is_mm else "",
                            "mmproj_path": str(root_f.resolve()) if is_mm else None,
                            "type": "llamacpp",
                        })
        except Exception:
            pass

    return {
        "hf_models": hf_models,
        "mnn_models": mnn_models,
        "llamacpp_models": llamacpp_models,
    }

