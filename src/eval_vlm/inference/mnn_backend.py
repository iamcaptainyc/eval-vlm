"""MNN(pymnn)本地推理后端。

训练完的 VLM 转成 MNN 格式后(llm.mnn / llm.mnn.weight / tokenizer.mtok /
config.json 等放同一目录),用 pymnn 的 ``MNN.llm`` 在本机直接推理,**无需起
HTTP 服务**——与 openai/vllm 后端(调远端 OpenAI 兼容 API)互补。

关键事实(据 pymnn C++ 绑定 pymnn/src/llm.h):
  - ``llm.create(config_path)`` + ``model.load()`` 加载模型(config_path 指向
    转换产物目录里的 config.json)。
  - ``model.response(content, stream, max_new_tokens)``:
      * content 可为 str 或**多模态 dict**;
      * stream=0(False)时,生成内容写入内部字符串并**作为返回值**返回;
        stream=1 时只往 stdout 流式打印、返回空串。故本后端用 **stream=False**
        来拿到完整文本,无需手写 forward/argmax 解码循环。
  - 多模态 dict 形如 ``{'text': '<img>image_0</img>...', 'images': [{'data': Var,
    'width': W, 'height': H}]}``;images[i] 对应文本里的 ``image_{i}``。data 需是
    MNN Var,用 ``MNN.cv.imread(path)`` 高效原生读取(返回 Var,免 numpy 往返)。

约束:
  - pymnn 的 LLM 无批量接口(response/generate/forward 均单条),且单个 Llm 对象
    有状态(KV cache / history),**不可并发**。故 ``thread_safe=False``,编排层会
    把并发降为 1;本类再加一把锁兜底。
  - 面向 ``pred`` 的无标注单图描述:每个样本恰好 1 张图、单个 user 提问。
"""
from __future__ import annotations

import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional

from ..config import Config
from ..data.loader import resolve_image_path
from ..data.schema import Prediction, Turn
from .base import InferenceBackend

# 文本里的图片占位:本工具内部统一用 <image>;MNN 多模态用 <img>image_N</img>。
_INTERNAL_PLACEHOLDER = "<image>"

# MNN.cv.imread 底层用 stb_image 解码,只稳定支持这几种扩展名;.webp/.gif/.tiff
# 等会"打不开"并返回**非法 Var**,后续读 .shape 会在 native 层 Segfault。
_IMREAD_NATIVE_OK = {".jpg", ".jpeg", ".png", ".bmp", ".ppm", ".pgm"}


class MNNBackend(InferenceBackend):
    # 单个有状态 Llm 对象,绝不能并发调用。
    thread_safe = False

    def __init__(self, cfg: Config) -> None:
        super().__init__(cfg)
        try:
            import MNN.llm as llm
            import MNN.cv as cv
        except ImportError as e:  # pragma: no cover - 取决于运行环境是否装了 pymnn
            raise ImportError(
                "backend=mnn 需要安装带 LLM 支持的 pymnn(MNN.llm / MNN.cv)。"
                "请参考 MNN 文档编译安装(-DMNN_BUILD_LLM=ON,多模态再加 "
                "-DMNN_BUILD_LLM_OMNI=ON),或 `pip install MNN`。"
            ) from e

        config_path = cfg.inference.mnn_config_path
        if not config_path:
            raise ValueError(
                "backend=mnn 需要 inference.mnn_config_path(转换产物目录里的 "
                "config.json 路径);请在数据集 config.yaml 设置,或用 "
                "`eval-vlm pred --mnn-config <config.json>` 临时指定。"
            )

        self._cv = cv
        self._lock = threading.Lock()
        self.model = llm.create(str(config_path))
        self.model.load()

    # ------------------------------------------------------------------
    def _imread(self, img_path: Path):
        """安全地把图片读成 MNN Var(HWC, uint8, BGR),兼容 .webp 等格式。

        不直接对任意格式调 MNN.cv.imread:其底层解码器(stb_image)不认 .webp、
        .gif 等;打不开时返回**非法 Var**,后续读 .shape 会在 native 层 Segfault、
        直接 core dump 整个进程(而非被 complete 捕获成 Prediction.error)。

        策略:原生解码器认得的扩展名走 imread 快路;其余先用 Pillow 解码并转码成
        临时 PNG,再交给 imread —— 这样最终 Var 仍由 imread 产生,dtype/通道序/
        排布与可用格式完全一致,只是多一步格式转换,不改变送入模型的像素语义。
        """
        if img_path.suffix.lower() in _IMREAD_NATIVE_OK:
            return self._cv.imread(str(img_path))

        from PIL import Image  # Pillow 已是项目依赖,支持 webp/gif/tiff 等

        fd, tmp = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        try:
            with Image.open(img_path) as im:
                im.convert("RGB").save(tmp, format="PNG")
            return self._cv.imread(tmp)
        finally:
            try:
                os.remove(tmp)
            except OSError:
                pass

    def _build_query_text(self, context: list[Turn], sample_id: str) -> str:
        """从对话上下文取出含 <image> 的 user 轮,转成 MNN 的单条多模态查询文本。

        把唯一的 <image> 占位符替换为 <img>image_0</img>(单图)。pred 的对话已由
        build_context 校验:<image> 恰好 1 个且在 user 轮、最后一轮为 user。
        MNN response() 只接受**单条**查询,故若上下文还有其它轮(如 few-shot),
        它们不会被送入模型——这里只取图片所在轮的提问文本。
        """
        img_turns = [t for t in context if _INTERNAL_PLACEHOLDER in t.content]
        if len(img_turns) != 1:
            raise ValueError(
                f"样本 {sample_id}:MNN 后端要求对话恰好含 1 个 {_INTERNAL_PLACEHOLDER} "
                f"占位符,当前 {len(img_turns)} 个。"
            )
        text = img_turns[0].content.replace(_INTERNAL_PLACEHOLDER, "<img>image_0</img>", 1)
        return text

    def complete(
        self,
        context: list[Turn],
        images: list[str],
        sample_id: str,
        expected: Optional[str] = None,
    ) -> Prediction:
        ic = self.cfg.inference
        start = time.time()
        try:
            if len(images) != 1:
                raise ValueError(
                    f"样本 {sample_id}:MNN 后端仅支持单图推理,当前 {len(images)} 张。"
                )
            text = self._build_query_text(context, sample_id)

            img_path = resolve_image_path(images[0], self.cfg)
            if not img_path.exists():
                raise FileNotFoundError(f"图片不存在: {img_path}(原始引用: {images[0]})")

            img = self._imread(img_path)   # MNN Var,HWC(.webp 等经 Pillow 转码)
            shape = list(img.shape)
            if len(shape) < 2:
                raise ValueError(f"无法识别图片尺寸(shape={shape}): {img_path}")
            height, width = int(shape[0]), int(shape[1])

            prompt = {
                "text": text,
                "images": [{"data": img, "height": height, "width": width}],
            }
        except Exception as e:  # 构造阶段失败(缺图/尺寸异常等)
            return Prediction(id=sample_id, error=f"build_prompt: {e}")

        with self._lock:
            try:
                # 每张图是独立单轮对话:先清掉上一次的 history/KV,避免串话。
                try:
                    self.model.reset()
                except Exception:  # noqa: BLE001 - reset 不可用则依赖 reuse_kv=false 默认
                    pass
                # stream=False -> 返回完整生成文本(见模块 docstring)。
                text_out = self.model.response(prompt, False, ic.max_tokens)
                raw = {"backend": "mnn"}
                try:
                    ctx = self.model.get_context()
                    if isinstance(ctx, dict):
                        for k in ("prompt_len", "gen_seq_len", "vision_us",
                                  "prefill_us", "decode_us", "pixels_mp"):
                            if k in ctx:
                                raw[k] = ctx[k]
                except Exception:  # noqa: BLE001 - 统计信息可选,拿不到不影响结果
                    pass
                return Prediction(
                    id=sample_id,
                    prediction=text_out or "",
                    latency=round(time.time() - start, 3),
                    raw=raw,
                )
            except Exception as e:  # noqa: BLE001 - 捕获以记录而非中断整批
                return Prediction(
                    id=sample_id,
                    latency=round(time.time() - start, 3),
                    error=f"{type(e).__name__}: {e}",
                )
