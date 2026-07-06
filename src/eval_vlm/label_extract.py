"""描述后调远程服务抽取结构化标签(pred --datadir --label-extract 用)。

参照 app 的 LabelExtractService(Retrofit/OkHttp):对 {base_url}{path} 发 POST,
Authorization 头带 bearer token,Content-Type 为 application/json,请求体 {"text":
<描述>}。响应体形如 {"code":200, "data":{"labels":{"cn":{...},"en":{...}}, ...}}。

本模块是**推理之后的独立流水线**:读 predictions.jsonl(每张图的模型描述),把描述
发给服务、解析出中文标签,写到 label.jsonl(每行 {"image","labels"})。它不依赖具体
推理后端,故:
  - 可独立并发(label_extract.max_concurrency),即便推理后端(如 mnn)是串行的;
  - 天然支持断点续跑(label.jsonl 里已成功的图跳过)与 --overwrite 整份重跑;
  - 抽取失败按用户选择「记 error 跳过、不中断整批」:该条写入 label_failures.jsonl。

只用标准库 urllib 发请求(不引入 requests 依赖)。
"""
from __future__ import annotations

import json
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

from tqdm import tqdm

from .config import Config, LabelExtractConfig
from .results import store


class LabelExtractError(Exception):
    """标签抽取失败(HTTP 错误、超时、code!=200、响应体不合法等)。"""


def parse_cn_labels(data: dict, none_label: str = "无") -> list[str]:
    """从服务响应的 data.labels.cn 提取非「无」标签,按类目/列表顺序扁平合并、去重保序。

    cn 是「类目 -> 标签列表」;值为 [none_label] 表示该类目无内容。例:
        {"自然风光": ["日落","山脉","森林"], "车辆识别": ["汽车"], "天气识别": ["无"], ...}
      -> ["日落", "山脉", "森林", "汽车"]
    """
    cn = ((data or {}).get("labels") or {}).get("cn") or {}
    out: list[str] = []
    seen: set[str] = set()
    for values in cn.values():
        if not isinstance(values, list):
            continue
        for v in values:
            s = str(v).strip()
            if not s or s == none_label or s in seen:
                continue
            seen.add(s)
            out.append(s)
    return out


def _endpoint(le: LabelExtractConfig) -> str:
    """拼出完整请求地址(容忍 base_url 末尾斜杠 / path 开头斜杠的组合)。"""
    return le.base_url.rstrip("/") + "/" + le.path.lstrip("/")


def _post_data(text: str, le: LabelExtractConfig) -> dict:
    """POST {"text": text},校验 code==200 后返回响应体的 data 段;失败抛 LabelExtractError。"""
    body = json.dumps({"text": text}, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(_endpoint(le), data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", le.auth_token)
    try:
        with urllib.request.urlopen(req, timeout=le.request_timeout) as resp:
            raw = resp.read().decode("utf-8")
    except Exception as e:  # noqa: BLE001 - URLError/HTTPError/超时统一转本模块异常
        raise LabelExtractError(f"{type(e).__name__}: {e}") from e
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as e:
        raise LabelExtractError(f"响应非合法 JSON: {raw[:200]}") from e
    if obj.get("code") != 200:
        raise LabelExtractError(f"服务返回 code={obj.get('code')}, msg={obj.get('msg')}")
    data = obj.get("data")
    if not isinstance(data, dict):
        raise LabelExtractError(f"响应缺少 data 段: {raw[:200]}")
    return data


def extract_one(text: str, le: LabelExtractConfig) -> list[str]:
    """对单条描述抽取 cn 标签列表,带指数退避重试;彻底失败抛 LabelExtractError。"""
    last: Optional[Exception] = None
    for attempt in range(le.max_retries + 1):
        try:
            return parse_cn_labels(_post_data(text, le), le.none_label)
        except LabelExtractError as e:
            last = e
            if attempt < le.max_retries:
                time.sleep(min(2 ** attempt, 10))
    raise LabelExtractError(str(last))


def _iter_descriptions(pred_path: Path) -> Iterator[tuple[str, str]]:
    """从 predictions.jsonl 读 (image, description)。

    predictions.jsonl 每行是一条成功描述(LlamaFactory 格式):id=图名,末条 assistant
    轮的 content 即模型生成的描述。忽略含 error / 无 id 的行(理论上不会有,防御性)。
    """
    if not pred_path.exists():
        return
    with pred_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("error") is not None or "id" not in rec:
                continue
            desc = ""
            for m in reversed(rec.get("messages") or []):
                if m.get("role") == "assistant":
                    desc = str(m.get("content", ""))
                    break
            yield str(rec["id"]), desc


def _done_labels(path: Path) -> set[str]:
    """读取 label.jsonl 中已成功抽取的图片名(断点续跑跳过;有 error 的不算)。"""
    done: set[str] = set()
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("error") is None and "image" in obj:
                done.add(str(obj["image"]))
    return done


def run_label_extract(cfg: Config, *, overwrite: bool = False) -> dict:
    """读 predictions.jsonl -> 逐条调服务抽取标签 -> 落 label.jsonl。返回统计。

    overwrite=True 时忽略已有 label.jsonl 整份重跑(截断重写);默认断点续跑
    (已成功的图片跳过,追加写)。抽取失败的条目写 label_failures.jsonl(记 error),
    不中断整批,可重跑补齐。
    """
    le = cfg.label_extract
    pred_path = cfg.predictions_path
    if not pred_path.exists():
        raise FileNotFoundError(
            f"未找到预测结果,无法抽取标签: {pred_path}(请先运行 pred 生成描述)"
        )

    items = list(_iter_descriptions(pred_path))
    out_dir = cfg.run_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    labels_path = cfg.labels_path
    fail_path = cfg.label_failures_path

    done = set() if overwrite else _done_labels(labels_path)
    todo = [(img, desc) for img, desc in items if img not in done]

    n_ok = 0
    n_err = 0
    interrupted = False
    started = datetime.now(timezone.utc).isoformat()

    def _append(fh, obj: dict) -> None:
        """写一条 -> flush -> fsync:进程中途崩溃/被杀,已写结果也已落盘,可断点续跑。"""
        fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
        fh.flush()
        try:
            os.fsync(fh.fileno())
        except OSError:
            pass

    if not todo:
        print(f"[label-extract] 全部 {len(items)} 条描述已抽取标签,无需请求(断点续跑)。")
    else:
        print(f"[label-extract] 待抽取 {len(todo)} 条(已完成跳过 {len(items) - len(todo)} 条),"
              f"服务 -> {_endpoint(le)}", flush=True)
        max_workers = max(1, le.max_concurrency)
        with labels_path.open("w" if overwrite else "a", encoding="utf-8") as ok_fh, \
                fail_path.open("w", encoding="utf-8") as fail_fh:

            def _record(img: str, result) -> None:
                nonlocal n_ok, n_err
                if isinstance(result, Exception):
                    n_err += 1
                    _append(fail_fh, {"image": img,
                                      "error": f"{type(result).__name__}: {result}"})
                else:
                    n_ok += 1
                    _append(ok_fh, {"image": img, "labels": result})

            try:
                if max_workers == 1:
                    for img, desc in tqdm(todo, desc="label-extract", unit="img"):
                        try:
                            _record(img, extract_one(desc, le))
                        except Exception as e:  # noqa: BLE001 - 记录而非中断整批
                            _record(img, e)
                else:
                    with ThreadPoolExecutor(max_workers=max_workers) as pool:
                        futures = {pool.submit(extract_one, desc, le): img
                                   for img, desc in todo}
                        try:
                            for fut in tqdm(as_completed(futures), total=len(futures),
                                            desc="label-extract", unit="img"):
                                img = futures[fut]
                                try:
                                    _record(img, fut.result())
                                except Exception as e:  # noqa: BLE001
                                    _record(img, e)
                        except KeyboardInterrupt:
                            for fut in futures:
                                fut.cancel()
                            raise
            except KeyboardInterrupt:
                interrupted = True
                print(f"\n[label-extract] 已中断:本轮成功 {n_ok} 条已落盘 -> {labels_path};"
                      f"重跑同一命令即可断点续跑补齐。", flush=True)

    stats = {
        "endpoint": _endpoint(le),
        "num_predictions": len(items),
        "newly_completed": n_ok,
        "errors": n_err,
        "skipped_already_done": len(items) - len(todo),
        "interrupted": interrupted,
        "labels_path": str(labels_path),
        "started_at": started,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    store.write_json(out_dir / "label_extract_meta.json", stats)
    return stats
