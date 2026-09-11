"""llama.cpp GGUF 转换、目录扫描与多模态配对测试。"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch
import pytest

from eval_vlm.gguf_convert import run_gguf_conversion
from eval_vlm.workspace import scan_local_models, load_global_config
from eval_vlm.config import Config


def test_gguf_convert_command_generation_two_stage(tmp_path):
    """测试多模态两阶段导出 + 量化命令生成与中间文件清理。"""
    hf_path = tmp_path / "MyModel"
    hf_path.mkdir()
    (hf_path / "config.json").write_text("{}", encoding="utf-8")

    out_dir = tmp_path / "output_models" / "MyModel"

    executed_cmds = []

    def fake_run(cmd, check=False, **kwargs):
        executed_cmds.append(cmd)
        import subprocess
        # 模拟生成对应文件
        if len(cmd) > 1 and "convert_hf_to_gguf.py" in str(cmd[1]) and "--mmproj" in cmd:
            (out_dir / "MyModel_f16_mmproj.gguf").write_bytes(b"dummy mmproj")
        elif len(cmd) > 1 and "convert_hf_to_gguf.py" in str(cmd[1]):
            (out_dir / "MyModel_bf16.gguf").write_bytes(b"dummy base gguf")
        elif "llama-quantize" in str(cmd[0]):
            (out_dir / "MyModel_Q4_K_M.gguf").write_bytes(b"dummy quant gguf")
        return subprocess.CompletedProcess(cmd, returncode=0)

    with patch("subprocess.run", side_effect=fake_run), \
         patch("eval_vlm.gguf_convert.find_quantize_binary", return_value="llama-quantize"):
        res = run_gguf_conversion(
            hf_path=hf_path,
            name="MyModel",
            out_dir=out_dir,
            outtype="bf16",
            is_multimodal=True,
            mmproj_outtype="f16",
            quantize="Q4_K_M",
            clean_intermediate=True,
        )

    assert res["model_name"] == "MyModel"
    assert res["quantized_gguf"] is not None and Path(res["quantized_gguf"]).exists()
    assert res["mmproj_gguf"] is not None and Path(res["mmproj_gguf"]).exists()
    # 验证中间文件被成功清理
    assert not (out_dir / "MyModel_bf16.gguf").exists()

    # 验证执行了 3 步命令
    assert len(executed_cmds) == 3
    # 第 1 步: 主模型导出
    assert "--outtype" in executed_cmds[0]
    assert "bf16" in executed_cmds[0]
    assert "--mmproj" not in executed_cmds[0]
    # 第 2 步: 投影器导出
    assert "--mmproj" in executed_cmds[1]
    assert "--outtype" in executed_cmds[1]
    assert "f16" in executed_cmds[1]
    # 第 3 步: 量化
    assert "Q4_K_M" in executed_cmds[2]


def test_scan_local_models_llamacpp_pairing(tmp_path):
    """测试 scan_local_models 中对 llamacpp 子目录的扫描以及主模型与 mmproj 的成对识别。"""
    llamacpp_root = tmp_path / "model_gguf"
    llamacpp_root.mkdir()

    # 模型 A: 含主模型与 mmproj
    model_a_dir = llamacpp_root / "ModelA"
    model_a_dir.mkdir()
    main_a = model_a_dir / "ModelA_Q4_K_M.gguf"
    main_a.write_bytes(b"gguf main")
    mmproj_a = model_a_dir / "ModelA_bf16_mmproj.gguf"
    mmproj_a.write_bytes(b"gguf mmproj")

    # 模型 B: 纯语言主模型 (无 mmproj)
    model_b_dir = llamacpp_root / "ModelB"
    model_b_dir.mkdir()
    main_b = model_b_dir / "ModelB_bf16.gguf"
    main_b.write_bytes(b"gguf main b")

    res = scan_local_models(llamacpp_dir=str(llamacpp_root))
    lcpp_models = res["llamacpp_models"]
    assert len(lcpp_models) == 2

    a_entry = next((m for m in lcpp_models if m["model_name"] == "ModelA"), None)
    assert a_entry is not None
    assert a_entry["name"] == "ModelA (ModelA_Q4_K_M.gguf + ModelA_bf16_mmproj.gguf)"
    assert a_entry["path"] == str(main_a)
    assert a_entry["mmproj_path"] == str(mmproj_a)
    assert a_entry["type"] == "llamacpp"

    b_entry = next((m for m in lcpp_models if m["model_name"] == "ModelB"), None)
    assert b_entry is not None
    assert b_entry["path"] == str(main_b)
    assert b_entry["mmproj_path"] is None


def test_config_result_name_llamacpp_parent_dir(tmp_path):
    """测试 Config.inference.result_name 在 llamacpp 后端且 model 为默认时，自动提取父目录名 A。"""
    from eval_vlm.config import _build

    model_a_path = tmp_path / "model_gguf" / "A" / "A_Q4_K_M.gguf"
    model_a_path.parent.mkdir(parents=True)
    model_a_path.write_bytes(b"gguf")

    cfg_dict = {
        "dataset_dir": str(tmp_path),
        "inference": {
            "backend": "llamacpp",
            "llamacpp": {
                "model": "default",
                "model_path": str(model_a_path),
            },
        },
    }
    cfg = _build(Config, cfg_dict)
    assert cfg.inference.result_name == "A"
