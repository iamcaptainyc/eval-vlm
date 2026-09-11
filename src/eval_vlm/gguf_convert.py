"""HuggingFace 转 GGUF 转换模块。

支持调用 llama.cpp 官方转换脚本 convert_hf_to_gguf.py 以及 llama-quantize。
针对多模态模型执行两次转换:
  1. 主语言骨干模型: convert_hf_to_gguf.py <hf_path> --outfile <out_dir>/<A>_<outtype>.gguf --outtype <outtype>
  2. 多模态投影器:   convert_hf_to_gguf.py <hf_path> --mmproj --outfile <out_dir>/<A>_<mmproj_outtype>_mmproj.gguf --outtype <mmproj_outtype>
  3. 可选量化:       llama-quantize <out_dir>/<A>_<outtype>.gguf <out_dir>/<A>_<quant>.gguf <quant>
     (注意: mmproj 保持不量化)

模型产物统一下发到用户配置的 <llamacpp_models_dir>/<A>/ 目录下。
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

from .workspace import load_global_config, resolve_workspace


def find_llama_cpp_root(custom_dir: Optional[str | Path] = None) -> Optional[Path]:
    """尝试定位 llama.cpp 根目录或构建目录。"""
    candidates: list[Path] = []
    if custom_dir:
        candidates.append(Path(custom_dir).expanduser().resolve())

    env_dir = os.environ.get("LLAMA_CPP_DIR")
    if env_dir:
        candidates.append(Path(env_dir).expanduser().resolve())

    # 常见相对路径探测
    candidates.extend([
        Path.cwd() / "llama.cpp",
        Path.cwd().parent / "llama.cpp",
        Path("/root/autodl-tmp/capt/llama.cpp"),
        Path("/workspace/llama.cpp"),
    ])

    for p in candidates:
        if p.exists() and (p / "convert_hf_to_gguf.py").exists():
            return p
    return None


def find_convert_script(llama_cpp_root: Optional[Path] = None) -> Optional[Path]:
    """查找 convert_hf_to_gguf.py 脚本路径。"""
    if llama_cpp_root and (llama_cpp_root / "convert_hf_to_gguf.py").exists():
        return llama_cpp_root / "convert_hf_to_gguf.py"
    # PATH 探测
    which_script = shutil.which("convert_hf_to_gguf.py")
    if which_script:
        return Path(which_script).resolve()
    return None


def find_quantize_binary(llama_cpp_root: Optional[Path] = None) -> Optional[str]:
    """查找 llama-quantize 可执行文件。"""
    which_bin = shutil.which("llama-quantize")
    if which_bin:
        return which_bin
    if llama_cpp_root:
        for sub in ("build/bin", "bin", "build"):
            candidate = llama_cpp_root / sub / ("llama-quantize.exe" if os.name == "nt" else "llama-quantize")
            if candidate.exists():
                return str(candidate.resolve())
    return None


def is_multimodal_hf(hf_path: Path) -> bool:
    """粗略检测 HF 目录是否为多模态 VLM (含 preprocessor_config 或 visual 模块)。"""
    if not hf_path.is_dir():
        return False
    if (hf_path / "preprocessor_config.json").exists():
        return True
    cfg_file = hf_path / "config.json"
    if cfg_file.exists():
        try:
            content = cfg_file.read_text(encoding="utf-8", errors="ignore").lower()
            if any(k in content for k in ("visual", "vision_config", "qwen2_vl", "minicpmv", "llava")):
                return True
        except Exception:
            pass
    return False


def run_gguf_conversion(
    hf_path: str | Path,
    name: Optional[str] = None,
    out_dir: Optional[str | Path] = None,
    outtype: str = "bf16",
    is_multimodal: Optional[bool] = None,
    mmproj_outtype: str = "f16",
    mmproj_type: Optional[str] = None,
    quantize: Optional[str] = None,
    clean_intermediate: bool = False,
    llama_cpp_dir: Optional[str | Path] = None,
) -> dict[str, Any]:
    """执行完整的 HF -> GGUF 转换流。

    返回:
      {
        "model_name": str,
        "out_dir": str,
        "main_gguf": str,
        "mmproj_gguf": Optional[str],
        "quantized_gguf": Optional[str],
        "commands": list[list[str]]
      }
    """
    hf_p = Path(hf_path).expanduser().resolve()
    if not hf_p.exists():
        raise FileNotFoundError(f"HuggingFace 权重目录不存在: {hf_p}")

    # 1. 确定模型名 A
    model_name = (name or hf_p.name).strip()
    if not model_name:
        model_name = "model"

    # 2. 确定落地输出目录
    if out_dir:
        target_dir = Path(out_dir).expanduser().resolve()
    else:
        global_cfg = load_global_config()
        base_dir = global_cfg.get("llamacpp_models_dir")
        if base_dir:
            # 若配置了多个目录，取第一个
            if isinstance(base_dir, list) and base_dir:
                base_dir = base_dir[0]
            elif isinstance(base_dir, str) and ("\n" in base_dir or ";" in base_dir):
                import re
                parts = [p.strip() for p in re.split(r"[;\n\r]+", base_dir) if p.strip()]
                base_dir = parts[0] if parts else None
        if base_dir:
            target_dir = Path(base_dir).expanduser().resolve() / model_name
        else:
            # 兜底到工作区下的 _models/llamacpp/<A>
            ws = resolve_workspace(None, global_cfg)
            target_dir = ws / "_models" / "llamacpp" / model_name

    target_dir.mkdir(parents=True, exist_ok=True)

    # 3. 定位 llama.cpp 工具
    llama_root = find_llama_cpp_root(llama_cpp_dir)
    convert_script = find_convert_script(llama_root)
    if not convert_script or not convert_script.exists():
        raise FileNotFoundError(
            "未找到 convert_hf_to_gguf.py 脚本！请安装或通过 --llama-cpp-dir 明确指定 llama.cpp 源码根目录。"
        )

    # 4. 判断是否需要多模态转换
    if is_multimodal is None:
        is_multimodal = is_multimodal_hf(hf_p)

    outtype_clean = outtype.lower().strip()
    mmproj_outtype_clean = mmproj_outtype.lower().strip()

    # 目标文件名规范: <A>_<outtype>.gguf 与 <A>_<mmproj_outtype>_mmproj.gguf
    base_main_gguf = target_dir / f"{model_name}_{outtype_clean}.gguf"
    base_mmproj_gguf = target_dir / f"{model_name}_{mmproj_outtype_clean}_mmproj.gguf" if is_multimodal else None

    history_cmds: list[list[str]] = []

    print(f"\n========================================================", flush=True)
    print(f"🚀 开始执行 HuggingFace -> GGUF 转换管线", flush=True)
    print(f"  - 输入源目录:   {hf_p}", flush=True)
    print(f"  - 目标模型名:   {model_name}", flush=True)
    print(f"  - 最终落地目录: {target_dir}", flush=True)
    print(f"  - 骨干导出精度: {outtype_clean}", flush=True)
    print(f"  - 多模态转换:   {'是 (将两阶段导出主模型与 mmproj)' if is_multimodal else '否 (纯语言模型)'}", flush=True)
    if is_multimodal:
        print(f"  - 投影器精度:   {mmproj_outtype_clean}", flush=True)
        if mmproj_type and mmproj_type.strip() and mmproj_type.strip().lower() != "auto":
            print(f"  - 投影器类型:   {mmproj_type.strip()}", flush=True)
    if quantize:
        print(f"  - 目标量化格式: {quantize.upper()}", flush=True)
    print(f"========================================================\n", flush=True)

    # 阶段 1: 转换主语言模型
    cmd_main = [
        sys.executable,
        str(convert_script),
        str(hf_p),
        "--outfile", str(base_main_gguf),
        "--outtype", outtype_clean,
    ]
    history_cmds.append(cmd_main)
    print(f"[阶段 1/3] 导出主语言骨干模型 -> {base_main_gguf.name}...", flush=True)
    print(f"  执行指令: {' '.join(cmd_main)}\n", flush=True)

    res1 = subprocess.run(cmd_main, check=False)
    if res1.returncode != 0:
        raise RuntimeError(f"主模型 GGUF 转换失败 (退出码 {res1.returncode})")
    if not base_main_gguf.exists():
        raise FileNotFoundError(f"未产出主模型文件: {base_main_gguf}")

    # 阶段 2: 转换多模态投影器 (若开启)
    if is_multimodal and base_mmproj_gguf is not None:
        cmd_mmproj = [
            sys.executable,
            str(convert_script),
            str(hf_p),
            "--mmproj",
            "--outfile", str(base_mmproj_gguf),
            "--outtype", mmproj_outtype_clean,
        ]
        if mmproj_type and mmproj_type.strip() and mmproj_type.strip().lower() != "auto":
            # 某些 llama.cpp 分支支持 --model-name 或特定的投影器模式覆盖
            cmd_mmproj.extend(["--model-name", mmproj_type.strip()])
        history_cmds.append(cmd_mmproj)
        print(f"\n[阶段 2/3] 导出多模态视觉投影器 (--mmproj) -> {base_mmproj_gguf.name}...", flush=True)
        print(f"  执行指令: {' '.join(cmd_mmproj)}\n", flush=True)

        res2 = subprocess.run(cmd_mmproj, check=False)
        if res2.returncode != 0:
            raise RuntimeError(f"多模态投影器 mmproj GGUF 转换失败 (退出码 {res2.returncode})")
        if not base_mmproj_gguf.exists():
            raise FileNotFoundError(f"未产出 mmproj 文件: {base_mmproj_gguf}")
    else:
        print(f"\n[阶段 2/3] 跳过多模态投影器转换 (纯语言模型或已关闭 mmproj)", flush=True)

    # 阶段 3: llama-quantize 量化 (可选)
    final_main_gguf = base_main_gguf
    quant_file_out = None
    if quantize and quantize.strip():
        quant_upper = quantize.strip().upper()
        quant_bin = find_quantize_binary(llama_root)
        if not quant_bin:
            raise FileNotFoundError(
                f"未在系统 PATH 或 llama.cpp 编译目录中找到 llama-quantize 可执行文件！无法执行 {quant_upper} 量化。"
            )
        quant_file = target_dir / f"{model_name}_{quant_upper}.gguf"
        cmd_quant = [
            quant_bin,
            str(base_main_gguf),
            str(quant_file),
            quant_upper,
        ]
        history_cmds.append(cmd_quant)
        print(f"\n[阶段 3/3] 执行 llama-quantize 量化 ({quant_upper}) -> {quant_file.name}...", flush=True)
        print(f"  执行指令: {' '.join(cmd_quant)}\n", flush=True)

        res3 = subprocess.run(cmd_quant, check=False)
        if res3.returncode != 0:
            raise RuntimeError(f"llama-quantize 量化失败 (退出码 {res3.returncode})")
        if not quant_file.exists():
            raise FileNotFoundError(f"未产出量化模型文件: {quant_file}")

        final_main_gguf = quant_file
        quant_file_out = str(quant_file.resolve())

        # 清理未量化中间文件
        if clean_intermediate and base_main_gguf.exists() and base_main_gguf != quant_file:
            print(f"  [清理] 删除未量化基座文件: {base_main_gguf.name}", flush=True)
            try:
                base_main_gguf.unlink()
            except Exception as e:
                print(f"  [警告] 无法删除临时文件: {e}", flush=True)

    print(f"\n🎉 GGUF 转换圆满完成！", flush=True)
    print(f"  - 语言主干模型: {final_main_gguf}", flush=True)
    if is_multimodal and base_mmproj_gguf:
        print(f"  - 多模态投影器: {base_mmproj_gguf}", flush=True)
    print(f"  - 模型所在文件夹: {target_dir}", flush=True)
    print(f"  - 对应模型名标识: {model_name}\n", flush=True)

    return {
        "model_name": model_name,
        "out_dir": str(target_dir.resolve()),
        "main_gguf": str(final_main_gguf.resolve()),
        "mmproj_gguf": str(base_mmproj_gguf.resolve()) if (is_multimodal and base_mmproj_gguf) else None,
        "quantized_gguf": quant_file_out,
        "commands": history_cmds,
    }
