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

# 全局配置的机器级键(及默认值)。
_GLOBAL_KEYS = ("workspace", "media_root", "image_strip_prefix")
_GLOBAL_DEFAULTS = {
    "workspace": "~/eval_vlm_workspace",
    "media_root": ".",
    "image_strip_prefix": None,
}

_DEFAULT_GLOBAL_TEXT = """\
# eval_vlm 全局配置(机器级,所有数据集共享)
# 路径:EVAL_VLM_CONFIG 环境变量优先,否则 ~/.eval_vlm/config.yaml
# 用 `eval-vlm config set <key> <value>` 修改,或直接手改本文件。

workspace: ~/eval_vlm_workspace   # 所有数据集文件夹的父目录(split 在此创建 <数据集名>/)
media_root: .                     # 图片相对路径解析根(写进每个数据集的 config.yaml)
image_strip_prefix: null          # 跨机训练绝对前缀,本机不需要则 null
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
    cfg = dict(_GLOBAL_DEFAULTS)
    for k in _GLOBAL_KEYS:
        if k in raw:
            cfg[k] = raw[k]
    return cfg


def set_global_value(key: str, value: Optional[str]) -> Path:
    """设置一个全局配置键(保留注释/其余行);value 为 None 写成 null。"""
    if key not in _GLOBAL_KEYS:
        raise KeyError(f"未知全局配置键: {key}(可选: {', '.join(_GLOBAL_KEYS)})")
    init_global_config()                       # 确保文件存在
    path = global_config_path()
    text = path.read_text(encoding="utf-8")
    text = _update_yaml_value(text, key, value)
    path.write_text(text, encoding="utf-8")
    return path


def _update_yaml_value(text: str, key: str, value: Optional[str]) -> str:
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
    """读取内置模板,把 {{KEY}} 占位符替换为渲染后的 YAML 标量。"""
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
    media_root: Any = ".",
    image_strip_prefix: Any = None,
    force: bool = False,
) -> Path:
    """初始化一个数据集文件夹:建目录 + 从模板渲染 config.yaml。返回文件夹路径。

    不在此执行 split(由调用方拿到 folder 后 load_dataset_config + split_dataset)。
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
    so = split_overrides or {}
    values = {
        "RUN_NAME": ds_name,
        "OUTPUT_DIR": str(workspace),
        "SOURCE": str(src),
        "MEDIA_ROOT": media_root,
        "IMAGE_STRIP_PREFIX": image_strip_prefix,
        "TRAIN": so.get("train", 0.95),
        "TEST": so.get("test", 0.05),
        "VAL": so.get("val", 0.0),
        "SEED": so.get("seed", 42),
        "STRATIFY_BY": so.get("stratify_by", None),
    }
    config_path.write_text(render_template(values), encoding="utf-8")
    return folder
