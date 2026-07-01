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
    """把 inference.result_name 转成合法的文件夹名(用作 数据集/<模型>/ 子目录)。

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
class OpenAIBackendConfig:
    """openai / vllm 后端(及 fake 自检)的全部设置,独立成块,与其它后端互不干扰。"""
    base_url: str = "http://localhost:8000/v1"
    model: str = "trained-vlm"                  # 也用作产物子目录名 <数据集>/<model>/
    api_key_env: str = "OPENAI_API_KEY"
    system_prompt: Optional[str] = None
    max_concurrency: int = 8
    max_tokens: int = 512
    temperature: float = 0.0
    request_timeout: float = 120.0
    max_retries: int = 3
    image_detail: str = "auto"


@dataclass
class MNNBackendConfig:
    """MNN(pymnn)本地后端的全部设置,独立成块。

    只含 mnn 真正会用到的项(无 base_url/model/并发等无意义字段),
    避免「切到 mnn 后某些设置其实不生效」的困惑。
    """
    # 训练后转 mnn 的模型目录里 config.json 的路径,传给 MNN.llm.create()。
    # 也据此(其所在目录名)决定产物子目录名;可用 pred 的 --mnn-config 临时覆盖。
    config_path: Optional[str] = None
    # 图片最长边的像素上限。超大图(如几千×几千、几十 MB)原样喂进 pymnn 的 vision
    # 编码器会在原生层 OOM/越界 -> Segmentation fault 直接 core dump 整个进程
    # (Python 捕获不到)。超过此上限的图先等比缩放再推理,从根上避免崩溃。
    # 正常尺寸图不受影响;设 <=0 关闭缩放(回到原样喂入,风险自负)。
    image_max_side: int = 2048
    max_tokens: int = 512                       # 作为 response 的 max_new_tokens

    # --- 采样 / 重复抑制(value-gated:每个旋钮按值开关,后端自动翻译成 MNN 采样管线)---
    # 无需关心 MNN 的 sampler_type / mixed_samplers:后端据下面哪些值被打开,自动拼装
    # MNN 的 mixed 流水线(见 mnn_backend._apply_sampler_config)。规则一句话:
    #   penalty(repetition/frequency/presence 任一开)→ 加 penalty 步;
    #   top_k / top_p 设值 → 加对应截断步;
    #   temperature>0 → 末步随机采样;temperature 未设/<=0 → 末步 argmax(确定、可复现)。
    # 小模型(如 0.8B)贪心解码遇到「没把握」的图易陷入 "\n\n\n…" 复读到 max_tokens;
    # 默认「仅开重复惩罚 1.1 + 确定性选词」即可止住复读且结果可复现。各项 __init__ 时经 set_config 下发。
    repetition_penalty: float = 1.1     # >1 惩罚已出现 token(含换行);<=1 关闭。复读顽固可调 1.3~1.5
    frequency_penalty: float = 0.0      # >0 按出现次数累加惩罚(专治同一符号刷屏);0 关闭
    presence_penalty: float = 0.0       # >0 对出现过的 token 一次性惩罚;0 关闭
    penalty_window: int = 0             # 惩罚只回看最近 N 个 token;0=整段历史
    temperature: Optional[float] = None # None/<=0 => 确定性 argmax(可复现);>0 => 温度随机采样
    top_k: Optional[int] = None         # 设值则启用 top-k 截断(只在前 K 个候选里选);仅随机采样(temperature>0)有意义
    top_p: Optional[float] = None       # 设值则启用 nucleus 截断(累积概率到 p);仅随机采样(temperature>0)有意义
    # 高级逃生口:非空 dict 会**原样**下发给 MNN set_config、跳过上面的自动翻译(可直接写
    # sampler_type / mixed_samplers 等 MNN 原生键);设为 {} 则一概不下发、沿用模型 config.json 自带采样。
    sampler_config: Optional[dict] = None


@dataclass
class CMNNBackendConfig(MNNBackendConfig):
    """cmnn(C++ 批量 MNN)后端设置。

    功能与 mnn 一致(多模态单图 + 同一套 value-gated 采样/重复抑制旋钮 +
    图片缩放),故直接继承 MNNBackendConfig 的全部字段;区别只在**执行方式**:
    cmnn 走原生 C++ 库,起 num_workers 个 Llm 实例 + 线程池并行处理整批
    (绕开 pymnn 的 GIL 串行瓶颈)。多出两个旋钮:

      num_workers — 原生库内并发的 Llm 实例数(=真并行度)。受显存/内存约束:
                    每个实例各自持有一份 KV cache(权重尽量共享)。
      batch_size  — 编排层一次交给原生库的请求条数(分块喂,便于增量落盘/断点续跑);
                    宜 >= num_workers 以喂满所有实例。
    """
    num_workers: int = 4
    batch_size: int = 16


@dataclass
class InferenceConfig:
    """推理设置:顶层只选 backend,各后端的参数归入各自的子块。

    切换 backend 只读对应块,各后端设置互不冲突、不会「设了却不生效」。
    新增后端只需加一个子块 + 一个分支。
    """
    backend: str = "openai"
    openai: OpenAIBackendConfig = field(default_factory=OpenAIBackendConfig)
    mnn: MNNBackendConfig = field(default_factory=MNNBackendConfig)
    cmnn: CMNNBackendConfig = field(default_factory=CMNNBackendConfig)

    @property
    def active(self) -> Any:
        """当前 backend 对应的设置块(openai/vllm/fake -> openai;mnn -> mnn;cmnn -> cmnn)。"""
        if self.backend in ("openai", "vllm", "fake"):
            return self.openai
        if self.backend == "mnn":
            return self.mnn
        if self.backend == "cmnn":
            return self.cmnn
        raise ValueError(
            f"未知推理后端: {self.backend!r}(可选: openai, vllm, mnn, cmnn, fake)"
        )

    @property
    def result_name(self) -> str:
        """产物子目录名(<数据集>/<result_name>/),按后端取其模型标识。

        openai/vllm/fake -> openai.model;mnn/cmnn -> config_path 所在目录名
        (如 /x/qwen2-vl-mnn/config.json -> qwen2-vl-mnn),缺省回落 '<backend>-model'。
        """
        if self.backend in ("mnn", "cmnn"):
            cp = self.active.config_path
            return Path(cp).expanduser().parent.name if cp else f"{self.backend}-model"
        if self.backend in ("openai", "vllm", "fake"):
            return self.openai.model
        # 未知后端:与 active 一致地报错,而不是伪装成 openai 给出一个看似正常的目录名。
        raise ValueError(
            f"未知推理后端: {self.backend!r}(可选: openai, vllm, mnn, cmnn, fake)"
        )

    @property
    def max_tokens(self) -> int:
        """编排/统计层用的生成上限(取当前后端块的 max_tokens)。"""
        return self.active.max_tokens

    @property
    def max_concurrency(self) -> int:
        """编排层用的并发数(当前后端块若无此项则为 1,如 mnn 串行)。"""
        return getattr(self.active, "max_concurrency", 1)

    @property
    def system_prompt(self) -> Optional[str]:
        """当前后端块的系统提示(无则 None,如 mnn 不支持系统提示)。"""
        return getattr(self.active, "system_prompt", None)


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
        """本次 pred/score/eval 的产物目录 = 数据集文件夹 / <模型名>。

        按 inference.result_name 分目录(openai/vllm/fake 取 openai.model;mnn 取
        config_path 所在目录名),使不同模型对同一数据集的结果互不覆盖
        (组织为 工作目录/数据集/模型)。split 产物(train/test/val/split_meta)
        是各模型共享的,落在 dataset_dir,不进这个子目录。
        """
        return self.dataset_dir / safe_model_dirname(self.inference.result_name)

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
              "mapping": Mapping, "tags": Tags,
              "openai": OpenAIBackendConfig, "mnn": MNNBackendConfig,
              "cmnn": CMNNBackendConfig}
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
