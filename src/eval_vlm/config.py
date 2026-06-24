"""配置加载与校验。

YAML -> 强类型 dataclass。配置是三步共享的中心,但每步只读自己关心的段,
因此即便某段缺失(例如只跑 split 时不关心 inference),也允许用默认值兜底。
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Optional

import yaml

# pred 默认单轮提示词(唯一真源;predict.py / cli.py import 本常量)。
DEFAULT_PROMPT = "请描述图片"

# 文件名非法字符(Windows 最严):路径分隔符、保留符号、控制字符。
_UNSAFE_DIR_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]+')


def safe_model_dirname(model: str) -> str:
    """把 inference.model 转成合法的文件夹名(用作 数据集/<模型>/ 子目录)。

    模型名常含 '/'(如 Qwen/Qwen2-VL-7B)或 ':' 等,直接做目录名会越界或非法。
    这里把所有非法字符折成 '_',并去掉 Windows 不允许的结尾点/空格;空则回落 'default'。
    """
    name = _UNSAFE_DIR_CHARS.sub("_", str(model or "").strip())
    name = name.strip(" ._")          # 去掉首尾的分隔/折叠符与 Windows 不允许的结尾点/空格
    return name or "default"


@dataclass
class Tags:
    role: str = "role"
    content: str = "content"
    user: str = "user"
    assistant: str = "assistant"


@dataclass
class Mapping:
    messages: str = "messages"
    images: str = "images"
    tags: Tags = field(default_factory=Tags)


@dataclass
class DataConfig:
    source: str = ""
    media_root: str = "."
    mapping: Mapping = field(default_factory=Mapping)
    # 跨机器图片路径处理:若图片是训练机绝对路径(如 /root/autodl-tmp/.../images/x.jpg),
    # 评测机上不存在,可填该前缀将其剥掉,剩余部分再相对 media_root 定位。
    image_strip_prefix: Optional[str] = None


@dataclass
class SplitConfig:
    """三路划分比例。train/test 必出,val 可选(val<=0 时不产出 val.json)。
    比例会自动归一化(无需严格相加为 1)。"""
    train: float = 0.8
    test: float = 0.2
    val: float = 0.0
    seed: int = 42
    stratify_by: Optional[str] = None
    # 自定义每份的输出路径/文件名(消除"改名 + 复制到 LlamaFactory data/"的手工步骤)。
    # 留空则落到默认 <output_dir>/<run_name>/{train,val,test}.json。
    # 可填绝对路径或相对 CWD 的路径,父目录会自动创建。
    # 例:train_out: /root/autodl-tmp/LlamaFactory/data/emotion_train.json
    train_out: Optional[str] = None
    val_out: Optional[str] = None
    test_out: Optional[str] = None


@dataclass
class InferenceConfig:
    backend: str = "openai"
    base_url: str = "http://localhost:8000/v1"
    model: str = "trained-vlm"
    api_key_env: str = "OPENAI_API_KEY"
    system_prompt: Optional[str] = None
    max_concurrency: int = 8
    max_tokens: int = 512
    temperature: float = 0.0
    request_timeout: float = 120.0
    max_retries: int = 3
    image_detail: str = "auto"
    # MNN(pymnn)后端专用:训练后转 mnn 的模型目录里 config.json 的路径,
    # 传给 MNN.llm.create()。仅 backend=mnn 时使用;openai/vllm/fake 忽略。
    # 可用 pred 的 --mnn-config 临时覆盖。
    mnn_config_path: Optional[str] = None
    # MNN 后端:图片最长边的像素上限。超大图(如几千×几千、几十 MB)原样喂进
    # pymnn 的 vision 编码器会在原生层 OOM/越界 -> Segmentation fault 直接 core dump
    # 整个进程(Python 捕获不到)。超过此上限的图先等比缩放再推理,从根上避免崩溃。
    # 正常尺寸图不受影响;设 <=0 关闭缩放(回到原样喂入,风险自负)。
    mnn_image_max_side: int = 2048


@dataclass
class EvalConfig:
    """多轮评测策略。

    targets — 评测哪些 assistant 轮:
        "all"  : 每个 assistant 轮都评(默认,轮1描述 + 轮2标签 ...)。
        "last" : 仅最后一个 assistant 轮(退回旧的"只评标签"行为)。
    context — 生成某一轮时,前面 assistant 轮用什么内容作上下文:
        "rollout" : 用模型**自己生成**的前文(真·连续对话,误差会累积,默认)。
        "gold"    : 用数据集里的标准前文(教师强制,各轮独立评测)。
    """
    targets: str = "all"
    context: str = "rollout"


@dataclass
class ScoringConfig:
    scorer: str = "exact_match"
    # 逐轮指定 scorer(按目标顺序);某轮缺省时回落到 scorer。
    # 例:[token_f1, exact_match] -> 轮1描述用 token_f1,轮2标签用 exact_match。
    turn_scorers: list[str] = field(default_factory=list)


@dataclass
class PredConfig:
    """无标注图片描述(pred 命令)的对话组织。

    两种写法,二选一:
      - 单轮简写:只设 prompt(+可选 system_prompt)。每张图发一轮
        user 消息;若 prompt 不含 <image> 占位符,自动前置一个。
      - 多轮模板:设 template(role/content 字典列表),覆盖 prompt。
        可含纯文本 assistant/user 轮做 few-shot 引导;<image> 标记目标图位置。

    约束(由 predict.build_context 校验):全部轮中 <image> 恰好 1 个且在 user 轮;
    最后一轮必须是 user(模型据此作答)。system_prompt 运行时映射到
    inference.system_prompt,复用后端既有系统消息处理。
    """
    prompt: str = DEFAULT_PROMPT
    system_prompt: Optional[str] = None
    # list[dict{role, content}];设置后覆盖 prompt。保持原始 dict(无需 dataclass 递归)。
    template: Optional[list] = None


@dataclass
class Config:
    run_name: str = "default_run"
    output_dir: str = "outputs"
    data: DataConfig = field(default_factory=DataConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    pred: PredConfig = field(default_factory=PredConfig)

    # 配置文件所在目录,用于把相对路径解析成绝对路径。
    config_dir: Path = field(default_factory=lambda: Path.cwd())

    # 显式钉死的产物目录(工作目录模型下 = 数据集文件夹本身)。
    # 一旦设置,run_dir 直接返回它,与 output_dir/run_name 解耦——
    # 这样数据集文件夹可整体搬移/改名,而 run/score 仍按"config.yaml 所在文件夹"定位产物。
    run_dir_path: Optional[Path] = None

    @property
    def dataset_dir(self) -> Path:
        """数据集文件夹本身(含 config.yaml + 各模型共享的 split 产物)。

        有显式 run_dir_path(工作目录模型)时直接用它;
        否则回落到旧行为 <output_dir>/<run_name>/(兼容 --config / 程序化用法)。
        """
        if self.run_dir_path is not None:
            return self.run_dir_path
        base = self._resolve(self.output_dir)
        return base / self.run_name

    @property
    def run_dir(self) -> Path:
        """本次 run/score/pred 的产物目录 = 数据集文件夹 / <模型名>。

        按 inference.model 分目录,使不同模型对同一数据集的结果互不覆盖
        (组织为 工作目录/数据集/模型)。split 产物(train/test/val/split_meta)
        是各模型共享的,落在 dataset_dir,不进这个子目录。
        """
        return self.dataset_dir / safe_model_dirname(self.inference.model)

    # ---- 产物路径(三步之间的解耦契约) ----
    # split 产物:数据集级,各模型共享 -> 落在 dataset_dir。
    @property
    def train_path(self) -> Path:
        if self.split.train_out:
            return self._resolve(self.split.train_out)
        return self.dataset_dir / "train.json"

    @property
    def val_path(self) -> Path:
        if self.split.val_out:
            return self._resolve(self.split.val_out)
        return self.dataset_dir / "val.json"

    @property
    def test_path(self) -> Path:
        if self.split.test_out:
            return self._resolve(self.split.test_out)
        return self.dataset_dir / "test.json"

    @property
    def split_meta_path(self) -> Path:
        return self.dataset_dir / "split_meta.json"

    @property
    def predictions_path(self) -> Path:
        return self.run_dir / "predictions.jsonl"

    @property
    def metrics_path(self) -> Path:
        return self.run_dir / "metrics.json"

    @property
    def scored_path(self) -> Path:
        return self.run_dir / "scored.jsonl"

    @property
    def failures_path(self) -> Path:
        """exact_match 未命中样本的人类可读清单(按 id 分组,含完整对话),供人工审核。"""
        return self.run_dir / "failures.md"

    @property
    def summary_path(self) -> Path:
        return self.run_dir / "summary.md"

    @property
    def run_meta_path(self) -> Path:
        return self.run_dir / "run_meta.json"

    def _resolve(self, p: str | os.PathLike[str]) -> Path:
        """相对路径相对当前工作目录(CWD)解析,绝对路径原样返回。

        即配置中的数据/输出路径写成相对仓库根目录的形式,从仓库根运行即可。
        """
        path = Path(p)
        return path if path.is_absolute() else (Path.cwd() / path)

    @property
    def source_path(self) -> Path:
        return self._resolve(self.data.source)

    @property
    def media_root_path(self) -> Path:
        return self._resolve(self.data.media_root)


def _build(cls: type, data: dict[str, Any]) -> Any:
    """递归地把 dict 构造成嵌套 dataclass,忽略未知键,缺失键用默认值。"""
    kwargs: dict[str, Any] = {}
    type_hints = {f.name: f.type for f in fields(cls)}
    nested = {"data": DataConfig, "split": SplitConfig, "inference": InferenceConfig,
              "eval": EvalConfig, "scoring": ScoringConfig, "pred": PredConfig,
              "mapping": Mapping, "tags": Tags}
    for key, value in (data or {}).items():
        if key not in type_hints:
            continue
        if key in nested and isinstance(value, dict):
            kwargs[key] = _build(nested[key], value)
        else:
            kwargs[key] = value
    return cls(**kwargs)


def load_config(path: str | os.PathLike[str]) -> Config:
    """从 YAML 文件加载配置。"""
    config_path = Path(path).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    cfg = _build(Config, raw)
    cfg.config_dir = config_path.parent
    return cfg


def load_dataset_config(folder: str | os.PathLike[str]) -> Config:
    """加载数据集文件夹内的 config.yaml,并把产物目录钉到该文件夹。

    工作目录模型下,每个数据集文件夹自包含 config.yaml + 全部产物。
    run/score/eval 用这个入口:run_dir 固定为该文件夹本身,
    因此文件夹可整体搬移/改名,产物始终落在它内部。
    """
    folder_path = Path(folder).expanduser().resolve()
    config_path = folder_path / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"数据集文件夹缺少 config.yaml: {config_path}"
            f"(请先运行 eval-vlm split --dataset <源json> 初始化)"
        )
    cfg = load_config(config_path)
    cfg.run_dir_path = folder_path
    return cfg
