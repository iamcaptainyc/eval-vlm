"""MNN(C++ 原生库)**批量**推理后端。

与 ``mnn`` 后端(pymnn,串行)功能一致——同样是本机加载转换后的 mnn 模型、单图
多模态、同一套 value-gated 采样/重复抑制旋钮、超大图缩放、退化输出检测——区别只在
**执行方式**:

  - ``mnn``  : 走 pymnn。pymnn 原生推理基本不释放 GIL,且单个 ``Llm`` 对象有状态
               (KV cache),故只能**串行**;编排层把并发降为 1。
  - ``cmnn`` : 走一个 C++ 原生扩展 ``cmnn_native``,内部起 ``num_workers`` 个独立
               ``MNN::Transformer::Llm`` 实例 + 线程池,一次吃下整批请求、在 C++ 层
               **真并行**处理(无 GIL)。这是 pymnn 拿不到的吞吐。

分工(Python 侧薄、C++ 侧重):
  - Python(本文件):组多模态 prompt 文本(``<image>`` -> ``<img>image_0</img>``)、
    把图片路径**净化**成 C++ imread 能安全读取的形式(.webp 等转码 / 超大图缩放,复用
    与 mnn 相同的防 native-segfault 策略,但只依赖 Pillow、不依赖 pymnn)、翻译采样
    配置、检测退化输出。Python 层**不依赖 pymnn**,只需 Pillow + cmnn_native 扩展。
  - C++(``cmnn_native``):加载 N 个 Llm 实例、imread 图片、跑推理、回收统计。

批量契约(见 base.InferenceBackend.complete_batch):``complete_batch`` 收 N 条
``BatchItem``、返回**等长同序**的 N 条 ``Prediction``;单条失败写进该条 error,不中断整批。
``complete``(单条)退化为 ``complete_batch([item])[0]``,以便 runner 等逐条调用方也能用。

约束(同 mnn):面向 ``pred`` 的无标注单图描述——每样本恰好 1 张图、单个 user 提问。
"""
from __future__ import annotations

import json
import os
import tempfile
import warnings
from pathlib import Path
from typing import Optional

from ..config import Config
from ..data.loader import resolve_image_path
from ..data.schema import Prediction, Turn
from .base import BatchItem, InferenceBackend
from .mnn_backend import (
    _IMREAD_NATIVE_OK,
    _INTERNAL_PLACEHOLDER,
    MNNBackend,
    translate_sampler_config,
)

# 从 C++ 原生结果里往 Prediction.raw 收集的统计键(与 mnn 对齐)。
_STAT_KEYS = ("prompt_len", "gen_seq_len", "vision_us", "prefill_us", "decode_us", "pixels_mp")


class CMNNBackend(InferenceBackend):
    # 并发在 C++ 库内部完成(num_workers 个实例),对 Python 呈现为「一次批量调用」;
    # 故声明支持批量,编排层走 complete_batch 批量路径而非线程池。
    supports_batch = True
    # 不声明 Python 层线程安全:并行都在原生库内;Python 侧不应再对同一 engine 并发调用。
    thread_safe = False

    def __init__(self, cfg: Config) -> None:
        super().__init__(cfg)
        try:
            import cmnn_native
        except ImportError as e:  # pragma: no cover - 取决于是否已编译安装原生扩展
            raise ImportError(
                "backend=cmnn 需要已编译的 C++ 原生扩展 cmnn_native(链接 MNN 的 "
                "libllm/libMNN)。请按 native/cmnn/README.md 在目标机上编译安装,"
                "或改用 backend=mnn(pymnn,串行)。"
            ) from e

        mc = cfg.inference.cmnn
        config_path = mc.config_path
        if not config_path:
            raise ValueError(
                "backend=cmnn 需要 inference.cmnn.config_path(转换产物目录里的 "
                "config.json 路径);请在数据集 config.yaml 设置,或用 "
                "`eval-vlm pred --cmnn-config <config.json>` 临时指定。"
            )
        if mc.num_workers < 1:
            raise ValueError(f"inference.cmnn.num_workers 必须 >= 1,当前 {mc.num_workers}")

        self._native = cmnn_native
        # 生成长度超过 max_tokens 的一次性告警开关(见 complete_batch)。
        self._warned_over_limit = False
        # 起 num_workers 个 Llm 实例 + 线程池;加载权重可能较慢。
        self.engine = cmnn_native.Engine(str(config_path), mc.num_workers)
        # 采样/重复抑制 + 生成上限一次性下发到全部实例(C++ 侧对每个 Llm set_config)。
        self._apply_config()

    # ------------------------------------------------------------------
    def _apply_config(self) -> None:
        """把采样配置 + max_new_tokens 合成一份下发给原生 engine(应用到所有实例)。

        采样翻译与 mnn 共用 translate_sampler_config(cmnn 继承同一套旋钮)。
        best-effort:engine 无 set_config 或下发失败都不致命(退化为模型 config.json 默认)。
        """
        mc = self.cfg.inference.cmnn
        combined: dict = {}
        sampler = translate_sampler_config(mc)
        if sampler:
            combined.update(sampler)
        if mc.max_tokens and mc.max_tokens > 0:
            combined["max_new_tokens"] = int(mc.max_tokens)
        if not combined:
            return
        fn = getattr(self.engine, "set_config", None)
        if not callable(fn):
            return
        try:
            fn(json.dumps(combined))
        except Exception:  # noqa: BLE001 - 下发失败退化为默认,不中断
            pass

    def _prepare_image(self, img_path: Path) -> tuple[str, bool]:
        """把图片净化成 C++ imread 能安全读取的路径,返回 (路径, 是否临时文件需清理)。

        与 mnn._imread 相同的防 native-segfault 策略,但产出**路径**(由 C++ imread)、
        且只用 Pillow(不依赖 pymnn):
          1. 原生 imread 只认 _IMREAD_NATIVE_OK;.webp/.gif 等打不开会返回非法 Var ->
             读 shape 时 native segfault。故这类先 Pillow 转码成临时 PNG。
          2. 超大图(几千×几千)会撑爆视觉编码器 -> native OOM/segfault。超过 image_max_side
             的先 Pillow 等比缩放。native-OK 且尺寸达标的原样透传(让 C++ 原生高效解码)。
        """
        from PIL import Image

        max_side = self.cfg.inference.cmnn.image_max_side
        suffix = img_path.suffix.lower()

        # native-OK 且(不限尺寸 或 尺寸达标):原样透传,不经 Pillow 全解码。
        if suffix in _IMREAD_NATIVE_OK:
            if max_side <= 0:
                return str(img_path), False
            with Image.open(img_path) as im:      # 只读 header 拿尺寸,不 load 像素
                w, h = im.size
            if max(w, h) <= max_side:
                return str(img_path), False

        # 需要转码(非原生格式)或缩放(超限):Pillow 解码 -> 可选缩放 -> 临时 PNG。
        fd, tmp = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        try:
            with Image.open(img_path) as im:
                if max_side > 0:
                    w, h = im.size
                    if max(w, h) > max_side:
                        ratio = max_side / max(w, h)
                        im = im.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
                im.convert("RGB").save(tmp, format="PNG")
        except Exception:
            # 转码失败:删掉半成品临时文件,交由上层记为该条 build error。
            try:
                os.remove(tmp)
            except OSError:
                pass
            raise
        return tmp, True

    @staticmethod
    def _build_query_text(context: list[Turn], sample_id: str) -> str:
        """取含 <image> 的 user 轮,把唯一占位符替换为 <img>image_0</img>(与 mnn 一致)。"""
        img_turns = [t for t in context if _INTERNAL_PLACEHOLDER in t.content]
        if len(img_turns) != 1:
            raise ValueError(
                f"样本 {sample_id}:cmnn 后端要求对话恰好含 1 个 {_INTERNAL_PLACEHOLDER} "
                f"占位符,当前 {len(img_turns)} 个。"
            )
        return img_turns[0].content.replace(_INTERNAL_PLACEHOLDER, "<img>image_0</img>", 1)

    # ------------------------------------------------------------------
    def complete(
        self,
        context: list[Turn],
        images: list[str],
        sample_id: str,
        expected: Optional[str] = None,
    ) -> Prediction:
        """单条推理:退化为单元素批量(逐条调用方如 runner 也能用 cmnn,只是无并行收益)。"""
        return self.complete_batch(
            [BatchItem(context=context, images=images, sample_id=sample_id, expected=expected)]
        )[0]

    def complete_batch(self, items: list[BatchItem]) -> list[Prediction]:
        """批量推理:一次把整批送进原生库并行处理,返回等长同序的 Prediction 列表。

        流程:① 逐条构造(组 prompt 文本 + 净化图片路径),构造失败的当场记为 error;
        ② 把构造成功的整批交给 engine.generate_batch 并行跑;③ 结果按原下标回填,
        对每条做退化检测/超限告警/统计收集;④ 清理临时图片文件。
        """
        results: list[Optional[Prediction]] = [None] * len(items)
        temps: list[str] = []                     # 需清理的临时图片
        requests: list[dict] = []                 # 送 C++ 的请求(仅构造成功项)
        req_index: list[int] = []                 # requests[k] 对应 items 的原下标
        mc = self.cfg.inference.cmnn

        # ---- 阶段一:逐条构造(纯 Python,失败不影响其它条)----
        for i, it in enumerate(items):
            try:
                if len(it.images) != 1:
                    raise ValueError(
                        f"样本 {it.sample_id}:cmnn 后端仅支持单图推理,当前 {len(it.images)} 张。"
                    )
                text = self._build_query_text(it.context, it.sample_id)
                img_path = resolve_image_path(it.images[0], self.cfg)
                if not img_path.exists():
                    raise FileNotFoundError(f"图片不存在: {img_path}(原始引用: {it.images[0]})")
                native_path, is_temp = self._prepare_image(img_path)
                if is_temp:
                    temps.append(native_path)
                requests.append({
                    "text": text,
                    "image_path": native_path,
                    "max_new_tokens": int(mc.max_tokens),
                })
                req_index.append(i)
            except Exception as e:  # noqa: BLE001 - 构造失败记为该条 error
                results[i] = Prediction(id=it.sample_id, error=f"build_prompt: {e}")

        # ---- 阶段二:整批送 C++ 并行推理 ----
        try:
            outputs = self.engine.generate_batch(requests) if requests else []
        except Exception as e:  # noqa: BLE001 - 整批原生调用失败:全部构造成功项记为 error
            for k, i in enumerate(req_index):
                results[i] = Prediction(
                    id=items[i].sample_id, error=f"cmnn_native: {type(e).__name__}: {e}"
                )
            outputs = []
        finally:
            for tmp in temps:                     # 无论成败都清理临时图片
                try:
                    os.remove(tmp)
                except OSError:
                    pass

        if outputs and len(outputs) != len(requests):
            # 原生库违反等长契约:整批判失败(避免错位回填污染结果)。
            for k, i in enumerate(req_index):
                results[i] = Prediction(
                    id=items[i].sample_id,
                    error=f"cmnn_native: 返回条数 {len(outputs)} != 请求条数 {len(requests)}",
                )
            outputs = []

        # ---- 阶段三:结果回填 + 退化检测 + 统计 ----
        for k, out in enumerate(outputs):
            i = req_index[k]
            results[i] = self._finalize(items[i].sample_id, out, mc.max_tokens)

        # 兜底:任何仍为 None 的槽位(理论上不该发生)记为 error,保证等长同序无 None。
        for i, r in enumerate(results):
            if r is None:
                results[i] = Prediction(id=items[i].sample_id, error="cmnn_native: 无返回结果")
        return results  # type: ignore[return-value]

    def _finalize(self, sample_id: str, out: dict, max_tokens: int) -> Prediction:
        """把一条原生结果 dict 收敛成 Prediction(退化检测/超限告警/统计,与 mnn 对齐)。"""
        # 原生层单条失败:直接透传 error。
        err = out.get("error")
        if err:
            return Prediction(id=sample_id, error=f"cmnn_native: {err}")

        text_out = out.get("text") or ""
        raw: dict = {"backend": "cmnn"}
        for key in _STAT_KEYS:
            if key in out and out[key] is not None:
                raw[key] = out[key]
        latency = out.get("latency")
        gen_len = raw.get("gen_seq_len")

        # 退化输出(几乎全空白/换行刷屏):判为失败记入 failures,而非污染可复用数据集。
        # 复用 mnn 的判据(纯静态,无状态)。先于超限告警判断——退化是另一类问题
        # (模型不发 EOS),不应误报为「构建忽略 max_new_tokens」。
        degenerate, why = MNNBackend._is_degenerate(text_out)
        if degenerate:
            tok = f",共 {gen_len} token" if isinstance(gen_len, int) else ""
            return Prediction(
                id=sample_id,
                latency=latency,
                error=f"degenerate_output: {why}{tok}(疑似模型未发 EOS)",
                raw=raw,
            )

        # 正常文本但长度超上限:一次性告警(说明该 MNN 构建未采纳 max_new_tokens)。
        if (isinstance(gen_len, int) and gen_len > max_tokens
                and not self._warned_over_limit):
            self._warned_over_limit = True
            warnings.warn(
                f"[cmnn] 本次生成 {gen_len} token 超过 max_tokens={max_tokens},"
                f"说明当前 MNN 构建未采纳 max_new_tokens 上限。请在模型转换产物的 "
                f"config.json 里显式设置 \"max_new_tokens\": {max_tokens}(加载时读入),"
                f"或升级到支持该项的 MNN 版本。",
                stacklevel=1,
            )

        return Prediction(id=sample_id, prediction=text_out, latency=latency, raw=raw)

    def close(self) -> None:
        fn = getattr(self.engine, "close", None)
        if callable(fn):
            try:
                fn()
            except Exception:  # noqa: BLE001
                pass
