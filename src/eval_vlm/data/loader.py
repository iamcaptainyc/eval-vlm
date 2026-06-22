"""解析 LlamaFactory 数据集格式 JSON -> Sample 列表。

LlamaFactory 的多模态(sharegpt)数据有两种常见写法,区别只在字段名:

  A) mllm_demo 风格(本项目默认):
     {"messages": [{"role": "user", "content": "<image>..."},
                   {"role": "assistant", "content": "..."}],
      "images": ["path/1.jpg"]}

  B) 通用 sharegpt 风格:
     {"conversations": [{"from": "human", "value": "<image>..."},
                        {"from": "gpt", "value": "..."}],
      "images": ["path/1.jpg"]}

两者通过 config 里的 mapping(对齐 LlamaFactory dataset_info.json)统一处理,
改配置即可,无需改代码。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ..config import Config
from .schema import EvalTurn, Sample, Turn


class DataFormatError(ValueError):
    """数据不符合 LlamaFactory 预期格式时抛出。"""


def _stable_id(index: int, record: dict[str, Any]) -> str:
    """为样本生成稳定 id:索引 + 内容哈希前 8 位,既可读又能检测内容漂移。"""
    blob = json.dumps(record, sort_keys=True, ensure_ascii=False)
    digest = hashlib.sha1(blob.encode("utf-8")).hexdigest()[:8]
    return f"{index:06d}-{digest}"


def load_raw_records(path: Path) -> list[dict[str, Any]]:
    """读取 LlamaFactory JSON 为原始记录列表(不解析),供 splitter 原样切分。"""
    if not path.exists():
        raise FileNotFoundError(f"数据文件不存在: {path}")
    with path.open("r", encoding="utf-8") as f:
        records = json.load(f)
    if not isinstance(records, list):
        raise DataFormatError(
            f"LlamaFactory 数据应为 JSON 数组,实际为 {type(records).__name__}"
        )
    return records


def load_samples(cfg: Config, source: Path | None = None) -> list[Sample]:
    """按配置加载并解析 LlamaFactory 数据。

    source 为 None 时读 cfg.source_path(原始数据源);
    run/score 阶段传入 cfg.test_path 来读划分出的 test.json。
    """
    path = Path(source) if source is not None else cfg.source_path
    records = load_raw_records(path)
    m = cfg.data.mapping
    targets_mode = cfg.eval.targets
    samples: list[Sample] = []
    for i, rec in enumerate(records):
        if not isinstance(rec, dict):
            raise DataFormatError(f"第 {i} 条记录不是对象: {rec!r}")
        samples.append(_parse_record(i, rec, m, targets_mode))
    return samples


def _parse_record(index: int, rec: dict[str, Any], m: Any, targets_mode: str) -> Sample:
    conv_key = m.messages
    img_key = m.images
    t = m.tags

    if conv_key not in rec:
        raise DataFormatError(
            f"第 {index} 条记录缺少对话字段 '{conv_key}'(可在 config.data.mapping 中调整)"
        )

    raw_turns = rec.get(conv_key) or []
    images = list(rec.get(img_key) or [])

    # 保留完整对话(角色归一化为标准 user/assistant),记录每个 assistant 轮的下标。
    turns: list[Turn] = []
    assistant_indices: list[int] = []
    for idx, turn in enumerate(raw_turns):
        role_val = turn.get(t.role)
        content_val = turn.get(t.content, "")
        if role_val == t.user:
            norm_role = "user"
        elif role_val == t.assistant:
            norm_role = "assistant"
            assistant_indices.append(idx)
        else:
            norm_role = str(role_val)
        turns.append(Turn(role=norm_role, content=content_val))

    # 选出要评测的 assistant 轮:all=全部,last=仅最后一个。
    if targets_mode == "last":
        chosen = assistant_indices[-1:]
    else:  # "all"(默认)
        chosen = assistant_indices
    targets = [EvalTurn(turn_index=i, reference=turns[i].content) for i in chosen]

    # 校验 <image> 占位符数量与图片数一致(LlamaFactory 约定;对整段对话计数)。
    placeholder_count = sum(turn.content.count("<image>") for turn in turns)
    if images and placeholder_count != len(images):
        raise DataFormatError(
            f"第 {index} 条记录 <image> 占位符数({placeholder_count}) "
            f"与 images 数({len(images)})不一致"
        )

    # meta:保留除对话/图片外的其它字段,供分层抽样 / 分组评分。
    meta = {k: v for k, v in rec.items() if k not in (conv_key, img_key)}

    return Sample(
        id=_stable_id(index, rec),
        turns=turns,
        images=images,
        targets=targets,
        meta=meta,
    )


def source_hash(cfg: Config) -> str:
    """数据源文件内容哈希,写进 split 元信息以便复现/检测漂移。"""
    h = hashlib.sha256()
    with cfg.source_path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_image_path(img: str, cfg: Config) -> Path:
    """把样本里的图片引用解析成评测机上的真实路径。

    顺序:① 可选剥掉训练机绝对前缀 image_strip_prefix;
         ② 绝对路径原样用,相对路径相对 media_root 定位。
    http/https/data URL 由调用方在更上层处理(此函数只管本地路径)。
    """
    prefix = cfg.data.image_strip_prefix
    if prefix and img.startswith(prefix):
        img = img[len(prefix):].lstrip("/\\")
    p = Path(img)
    if not p.is_absolute():
        p = cfg.media_root_path / p
    return p
