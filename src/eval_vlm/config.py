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
DEFAULT_PROMPT = "请描述这张图片。"

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
    top_p: float = 1.0                          # nucleus 采样(累积概率到 p);1.0 = 不截断
    request_timeout: float = 120.0
    max_retries: int = 3
    image_detail: str = "auto"


@dataclass
class HFBackendConfig:
    """HuggingFace(transformers)本地参考后端设置,独立成块。

    用途:作为「转换前·训练态」的**参考基准**,与 mnn(转换后)后端做精度对比
    (见 precision 命令)。用 transformers 直接加载本地权重推理,复刻 LlamaFactory
    的图片预处理(image_max_pixels 规则),使其行为尽量贴近训练时。可选依赖:
    未装 transformers/torch 时构造后端才报错,不影响其它后端。
    """
    model_path: Optional[str] = None            # 本地 HF 权重目录;也据其目录名定产物子目录
    max_tokens: int = 512                       # generate 的 max_new_tokens
    # 图片预处理:与 MNNBackendConfig 同一套规则(对齐 LlamaFactory 训练),
    # 下发给 processor 的 min_pixels/max_pixels,使参考端与候选端预处理一致。
    image_max_pixels: int = 768 * 768           # 总像素上限(超过按 sqrt 因子缩小);<=0 关闭
    image_min_pixels: int = 32 * 32             # 总像素下限(不足按 sqrt 因子放大);<=0 关闭
    # 图片最长边像素上限:**纯等比缩放,不做 patch 对齐**(同 mnn 的 image_max_side)。
    # 与上面的 image_max/min_pixels 是两条独立路径:后者会作为 max_pixels/min_pixels
    # 下发给 processor(Qwen2-VL 专属旋钮),对 MiniCPM-V 这类切片模型会污染切片网格;
    # 本项则在**交给 processor 之前**由本后端自己等比缩小图片,patch/切片对齐仍交给模型
    # 自己的 processor。因此非 Qwen 模型(如 MiniCPM-V)建议:image_max/min_pixels 设 0
    # 关掉,改用本项控制输入尺寸。<=0 关闭(默认关,不改变既有 Qwen 行为)。
    image_max_side: int = 0
    system_prompt: Optional[str] = None         # 系统提示(应与训练一致);None=不加
    device: str = "auto"                        # auto/cuda/cpu,传给 from_pretrained(device_map)
    dtype: str = "auto"                         # auto/bfloat16/float16/float32
    greedy: bool = True                         # True=贪心解码(确定可复现);False=按模型默认采样


@dataclass
class MNNBackendConfig:
    """MNN(pymnn)本地后端的全部设置,独立成块。

    只含 mnn 真正会用到的项(无 base_url/model/并发等无意义字段),
    避免「切到 mnn 后某些设置其实不生效」的困惑。
    """
    # 训练后转 mnn 的模型目录里 config.json 的路径,传给 MNN.llm.create()。
    # 也据此(其所在目录名)决定产物子目录名;可用 pred 的 --mnn-config 临时覆盖。
    config_path: Optional[str] = None
    # --- 图片预处理(对齐 LlamaFactory 训练时的 mm_plugin 规则)---
    # 训练时 LlamaFactory 先把总像素超过 image_max_pixels 的图按 sqrt 因子缩小
    # (默认 768*768=589824),再交给 HF processor;MNN 引擎自身只按 28 对齐取整、
    # 不限总像素。不做这步缩小,高分辨率图会喂给视觉编码器远多于训练分布的 token,
    # 推理结果明显偏离训练效果(丢失训练要点)。<=0 关闭。
    image_max_pixels: int = 768 * 768
    # 总像素下限(同 LlamaFactory 默认 32*32=1024,过小的图按 sqrt 因子放大)。<=0 关闭。
    image_min_pixels: int = 32 * 32
    # 图片最长边的像素上限(本工具的 native OOM 兜底,与训练预处理无关)。超大图
    # (如几千×几千、几十 MB)原样喂进 pymnn 的 vision 编码器会在原生层 OOM/越界
    # -> Segmentation fault 直接 core dump 整个进程(Python 捕获不到)。超过此上限
    # 的图先等比缩放再推理。正常经 image_max_pixels 缩小后不会触发;设 <=0 关闭。
    image_max_side: int = 2048
    # 系统提示:经 set_config 下发给 MNN 引擎,由其 apply_chat_template 拼进对话
    # 模板。训练若带 system(LlamaFactory qwen2_vl 模板默认
    # "You are a helpful assistant."),推理也应保持一致,否则小模型行为易漂移。
    # None/空串 = 不下发,沿用模型 config.json 自带的 system_prompt。
    system_prompt: Optional[str] = None
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
    # 量化配方标签(如 "hqq-4bit" / "hqq-8bit")。**纯记录用,不参与推理**:落进 run_meta/
    # pred_meta,供 report 命令在质量并排表里标注该 MNN 变体是哪档量化,便于归因量化损失。
    quant: Optional[str] = None


@dataclass
class VLLMOfflineBackendConfig:
    """离线 vLLM(在进程内 `LLM(...)` 加载权重)本地推理后端设置,独立成块。

    与 openai/vllm(HTTP 别名)不同:**不起服务**,直接用 vllm 的 offline `LLM` 引擎
    在本机加载合并后的全精度权重推理。主要给自动调参闭环用(训练→合并→离线 vLLM 评测)。
    单模型对象有状态、不可并发(见 vllm_offline_backend.thread_safe=False)。
    """
    model_path: Optional[str] = None            # 合并后全精度权重目录(传给 LLM(model=...))
    # --- LLM(...) 引擎参数(默认取用户给的离线示例值)---
    gpu_memory_utilization: float = 0.9
    max_model_len: int = 4096
    max_num_seqs: int = 128
    max_num_batched_tokens: int = 20480
    # Qwen-VL processor 的像素上下限(经 mm_processor_kwargs 下发);建议与训练 image_max_pixels 对齐。
    image_min_pixels: int = 28 * 28
    image_max_pixels: int = 720 * 28 * 28
    dtype: str = "auto"
    trust_remote_code: bool = True
    # vLLM 相关环境变量:在 import vllm / 建 LLM **之前**写入 os.environ(部分 vllm/flashinfer 变量在
    # import 期就读取,晚设无效)。例:{"FLASHINFER_DISABLE_VERSION_CHECK": "1",
    # "VLLM_ATTENTION_BACKEND": "FLASH_ATTN"} 用来绕过/禁用 flashinfer。null=不设。
    env: Optional[dict] = None
    # 高级逃生口:非空 dict 原样合并进 LLM(**vllm_kwargs)(可写 enforce_eager 等 vllm 原生键)。
    vllm_kwargs: Optional[dict] = None
    # --- 生成参数(SamplingParams)---
    max_tokens: int = 512
    temperature: float = 0.0                    # 0 => 贪心(可复现)
    top_p: float = 1.0
    top_k: int = -1                             # -1 => 关闭
    repetition_penalty: float = 1.0             # 1.0 => 关闭
    system_prompt: Optional[str] = None         # 应与训练一致;None/空 = 不加


@dataclass
class InferenceConfig:
    """推理设置:顶层只选 backend,各后端的参数归入各自的子块。

    切换 backend 只读对应块,各后端设置互不冲突、不会「设了却不生效」。
    新增后端只需加一个子块 + 一个分支。
    """
    backend: str = "openai"
    # fail_fast:任一条推理出错就立即抛出**原始异常(带完整 traceback)**并中断整批,
    # 而非默认的「记 error 继续跑」。仅供调试用(如定位模型 reshape 报错),由 CLI 的
    # --fail-fast 运行时置位,不写回 config.yaml。跨后端通用,故放在顶层而非各后端块。
    fail_fast: bool = False
    openai: OpenAIBackendConfig = field(default_factory=OpenAIBackendConfig)
    mnn: MNNBackendConfig = field(default_factory=MNNBackendConfig)
    hf: HFBackendConfig = field(default_factory=HFBackendConfig)
    vllm_offline: VLLMOfflineBackendConfig = field(default_factory=VLLMOfflineBackendConfig)

    @property
    def active(self) -> Any:
        """当前 backend 对应的设置块(openai/vllm/fake -> openai;mnn -> mnn;hf -> hf;vllm_offline -> vllm_offline)。"""
        if self.backend in ("openai", "vllm", "fake"):
            return self.openai
        if self.backend == "mnn":
            return self.mnn
        if self.backend == "hf":
            return self.hf
        if self.backend == "vllm_offline":
            return self.vllm_offline
        raise ValueError(
            f"未知推理后端: {self.backend!r}(可选: openai, vllm, mnn, hf, vllm_offline, fake)"
        )

    @property
    def result_name(self) -> str:
        """产物子目录名(<数据集>/<result_name>/),按后端取其模型标识。

        openai/vllm/fake -> openai.model;mnn -> config_path 所在目录名
        (如 /x/qwen2-vl-mnn/config.json -> qwen2-vl-mnn),缺省回落 'mnn-model';
        hf -> model_path 目录名,缺省回落 'hf-model';
        vllm_offline -> model_path 目录名,缺省回落 'vllm-offline-model'。
        """
        if self.backend == "mnn":
            cp = self.mnn.config_path
            return Path(cp).expanduser().parent.name if cp else "mnn-model"
        if self.backend == "hf":
            mp = self.hf.model_path
            return Path(mp).expanduser().name if mp else "hf-model"
        if self.backend == "vllm_offline":
            mp = self.vllm_offline.model_path
            return Path(mp).expanduser().name if mp else "vllm-offline-model"
        if self.backend in ("openai", "vllm", "fake"):
            return self.openai.model
        # 未知后端:与 active 一致地报错,而不是伪装成 openai 给出一个看似正常的目录名。
        raise ValueError(
            f"未知推理后端: {self.backend!r}(可选: openai, vllm, mnn, hf, vllm_offline, fake)"
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
        """当前后端块的系统提示(openai 走系统消息;mnn 经 set_config 下发)。"""
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
class LabelExtractConfig:
    """描述完成后调远程服务抽取结构化标签(pred --datadir --label-extract 用)。

    参照 app 的 LabelExtractService:POST {base_url}{path},Authorization 头带 bearer
    token,请求体 {"text": <描述>};响应体 code==200 时,data.labels.cn 为「类目 ->
    标签列表」,值为 ["无"] 表示该类目无内容。本工具只取 cn 里非「无」的标签,按类目/
    列表顺序扁平合并成一个列表存入 label.jsonl(见 label_extract.parse_cn_labels)。
    """
    base_url: str = "https://canghai-agent-api-test.aijidou.com/"
    path: str = "api/v1/vlm/label-extract"
    # Authorization 头的完整值(含 bearer 前缀)。这是 app 里的测试 token,会过期;
    # 过期后用 --label-extract-token 覆盖(永久写回本字段),或直接改这里。
    auth_token: str = (
        "bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
        "eyJ1c2VyX2lkIjoyLCJleHAiOjE3ODMzMjM4NDJ9."
        "ggJb41tshiLZ-cmNEobWf9BygqlYynFmfj5OPEQARqs"
    )
    request_timeout: float = 30.0
    max_retries: int = 3
    max_concurrency: int = 4          # 抽取是纯 I/O,可并发(与推理后端是否串行无关)
    none_label: str = "无"            # cn 中表示「无该类目」的占位值,过滤掉
    # field-eval 用的专用路由:返回固定枚举字段(主辅路/道路结构/车道位置/警示标志),
    # 信封同 label-extract(data.labels.cn)。与上面的 path(旧 13 类标签)相互独立。
    value_path: str = "api/v1/vlm/value-extract"


@dataclass
class PrecisionConfig:
    """精度对比(precision 命令)设置:量化 mnn(转换后)相对 hf(转换前)的行为误差。

    对比是**解耦**的:不在同进程跑两个模型,而是各自用 `pred` 产出 predictions.jsonl
    (落在 <数据集>/<模型名>/<后端类型>/,候选走 mnn、参考走 hf 子目录),
    precision 只读这两份产物做对比 + 出报告。
    candidate/reference 指两个**模型名**;留空则默认取当前 mnn/hf 后端的 result_name。
    """
    candidate_dir: Optional[str] = None   # 候选(转换后,MNN)模型名(其 mnn 子目录);None=inference.mnn 的 result_name
    reference_dir: Optional[str] = None   # 参考(转换前,HF)模型名(其 hf 子目录);None=inference.hf 的 result_name
    agreement_min: float = 0.9            # 输出完全一致率低于此值 -> 报告标注「行为偏差偏大」
    token_f1_min: float = 0.9             # 平均 char token-F1 低于此值 -> 标注「文本相似度偏低」
    alignment_strict: bool = True         # True:prompt token 数 / 图片像素数有任一 delta≠0 即标注「预处理不对齐」
    max_worst_samples: int = 20           # precision.md 里展开的「最差样本」条数
    # 净质量Δ:候选(MNN)相对参考(HF)的「净回归率」(HF对了但MNN错了的占比)超过此值 -> 报告标 🔴。
    # 需两端都跑过 score(各自 scored.jsonl 存在)才计算;缺任一则跳过该项。
    quality_regression_max: float = 0.05


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
    label_extract: LabelExtractConfig = field(default_factory=LabelExtractConfig)
    precision: "PrecisionConfig" = field(default_factory=lambda: PrecisionConfig())

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
        """本次 pred/score/eval 的产物目录 = 数据集文件夹 / <模型名> / <后端类型>。

        三级组织(数据集 / 模型名 / 后端类型):模型名按 inference.result_name 取
        (openai/vllm/fake 取 openai.model;mnn 取 config_path 所在目录名;hf 取
        model_path 目录名);后端类型取 inference.backend(openai/vllm/mnn/hf/fake)。
        这样「同一模型的不同后端」结果互不覆盖(如 <模型>/mnn 与 <模型>/hf 并列),
        便于 precision 对比。split 产物(train/test/val/split_meta)是各模型共享的,
        落在 dataset_dir,不进这些子目录。
        """
        return (self.dataset_dir
                / safe_model_dirname(self.inference.result_name)
                / safe_model_dirname(self.inference.backend))

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

    # ---- precision(mnn vs hf 精度对比)----
    def model_run_dir(self, model_name: str, backend: str) -> Path:
        """按模型名 + 后端类型取产物子目录 <数据集>/<安全模型名>/<后端类型>/。

        precision 命令据此定位候选(MNN)/参考(HF)各自的 predictions.jsonl,
        与 run_dir(当前后端的目录)解耦,可对比任意两个已跑过的模型/后端。
        """
        return (self.dataset_dir
                / safe_model_dirname(model_name)
                / safe_model_dirname(backend))

    def predictions_path_for(self, model_name: str, backend: str) -> Path:
        """某模型 + 后端子目录下的 predictions.jsonl(precision 对比读它)。"""
        return self.model_run_dir(model_name, backend) / "predictions.jsonl"

    @property
    def precision_report_path(self) -> Path:
        """机器可读精度对比报告(落在当前后端=候选模型子目录)。"""
        return self.run_dir / "precision.json"

    @property
    def precision_md_path(self) -> Path:
        """人类可读精度对比报告。"""
        return self.run_dir / "precision.md"

    # ---- report(跨格式合并质量报告)----
    # 数据集级(跨模型/后端),落在 dataset_dir 顶层,与各模型子目录并列。
    @property
    def report_md_path(self) -> Path:
        """人类可读合并报告(HF vs 各 MNN 变体的质量并排 + 净质量Δ + 诊断)。"""
        return self.dataset_dir / "report.md"

    @property
    def report_json_path(self) -> Path:
        """机器可读合并报告。"""
        return self.dataset_dir / "report.json"

    @property
    def labels_path(self) -> Path:
        """label-extract 成功结果:每行 {image, labels}(与 predictions.jsonl 同目录)。"""
        return self.run_dir / "label.jsonl"

    @property
    def label_failures_path(self) -> Path:
        """label-extract 失败记录:每行 {image, error},供排查/重跑。"""
        return self.run_dir / "label_failures.jsonl"

    # ---- field-eval(第一轮描述的字段抽取 -> 逐字段准确率)----
    # ref 字段跨模型运行不变,缓存在**数据集级**(dataset_dir),各模型/后端复用;
    # pred 字段与比对结果是**运行级**(run_dir),随模型/后端区分。
    @property
    def field_ref_path(self) -> Path:
        """ref(标准描述)抽取出的字段,数据集级缓存:每行 {id, fields}。"""
        return self.dataset_dir / "fields_ref.jsonl"

    @property
    def field_ref_failures_path(self) -> Path:
        """ref 字段抽取失败记录:每行 {id, error}。"""
        return self.dataset_dir / "fields_ref_failures.jsonl"

    @property
    def field_pred_path(self) -> Path:
        """pred(模型描述)抽取出的字段,运行级:每行 {id, fields}。"""
        return self.run_dir / "fields_pred.jsonl"

    @property
    def field_pred_failures_path(self) -> Path:
        """pred 字段抽取失败记录:每行 {id, error}。"""
        return self.run_dir / "fields_pred_failures.jsonl"

    @property
    def field_metrics_path(self) -> Path:
        """逐字段准确率等聚合指标(机器可读)。"""
        return self.run_dir / "field_metrics.json"

    @property
    def field_summary_path(self) -> Path:
        """逐字段准确率的人类可读摘要。"""
        return self.run_dir / "field_summary.md"

    @property
    def field_mismatches_path(self) -> Path:
        """逐字段失配清单(按 id 列 ref vs pred 每字段值 + ✓/✗),供人工核查。"""
        return self.run_dir / "field_mismatches.md"

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
              "label_extract": LabelExtractConfig, "precision": PrecisionConfig,
              "mapping": Mapping, "tags": Tags,
              "openai": OpenAIBackendConfig, "mnn": MNNBackendConfig,
              "hf": HFBackendConfig, "vllm_offline": VLLMOfflineBackendConfig}
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
