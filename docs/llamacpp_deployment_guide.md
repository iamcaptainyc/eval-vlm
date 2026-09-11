# llama.cpp 多模态服务端编译与远程实机部署测试指南

本文档为远程服务器编译、部署 `llama.cpp` 多模态服务 (`llama-server`) 以及联动 `eval_vlm` 评测工具的操作指南。

---

## 一、服务器环境要求与编译

### 1. 克隆代码与依赖安装
确保远程机器安装了 `cmake` (>= 3.14) 和对应加速平台的编译器驱动（如 CUDA Toolkit 或 ROCm）。

### 2. CMake 编译选项

根据服务器硬件平台选择对应的编译参数：

#### (1) NVIDIA GPU (CUDA 架构)
```bash
cd /path/to/llama.cpp
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j $(nproc)
```

#### (2) AMD GPU (ROCm / HIP 架构)
```bash
cd /path/to/llama.cpp
cmake -B build -DGGML_HIPBLAS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j $(nproc)
```

#### (3) 仅 CPU 或 Apple Silicon
- CPU: `cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --config Release -j $(nproc)`
- macOS Metal: 默认开启 Metal 加速。

编译完成后，核心产物位于：
- `build/bin/llama-server`：HTTP 服务端（推荐）
- `build/bin/mtmd-cli`：离线命令行工具

---

## 二、多模态权重准备 (GGUF 与 mmproj)

llama.cpp 最新多模态架构采用两文件结构：
1. **主语言模型文件**：如 `qwen2.5-vl-7b-instruct-q4_k_m.gguf`
2. **多模态投影器文件 (`mmproj`)**：如 `mmproj-qwen2.5-vl-7b-instruct-f16.gguf`

### 获取方式：
- **方式一（直接下载社区已量化权重）**：
  从 Hugging Face（如 `Qwen` 或 `ggml-org`）直接下载对应的 GGUF 模型与 `mmproj-*.gguf` 文件。
- **方式二（从 Hugging Face 权重转换）**：
  ```bash
  # 转换主模型
  python convert_hf_to_gguf.py /path/to/Qwen2.5-VL-7B-Instruct --outfile /models/qwen2.5-vl-7b-f16.gguf
  # 转换并提取 mmproj
  python convert_hf_to_gguf.py /path/to/Qwen2.5-VL-7B-Instruct --mmproj --outfile /models/mmproj-f16.gguf
  ```

---

## 三、启动 llama-server 多模态服务

使用编译好的 `llama-server` 启动 HTTP 服务，挂载语言模型与视觉投影器：

```bash
./build/bin/llama-server \
  -m /models/qwen2.5-vl-7b-instruct-q4_k_m.gguf \
  --mmproj /models/mmproj-qwen2.5-vl-7b-instruct-f16.gguf \
  --port 8080 \
  --host 0.0.0.0 \
  -ngl 99 \
  -c 4096 \
  -np 4
```

### 关键参数说明：
- `--mmproj <path>`：**关键项**。必须指定多模态投影器路径，否则服务端无法处理图片输入。
- `-ngl 99` (`--n-gpu-layers`)：将模型与 mmproj 的层尽可能多地卸载到 GPU 显存。
- `-c 4096` (`--ctx-size`)：上下文窗口大小。
- `-np 4` (`--parallel`)：开启 4 个并发槽位（搭配 continuous batching 加速评测）。

---

## 四、使用 eval_vlm 进行多模态评测

`eval_vlm` 现在原生支持 `llamacpp` 后端，支持多轮对话、多图多轮输入以及 `field-eval` 闭环。

### 1. 一键预测 (pred)
```bash
# 对无标注图片文件夹生成描述 (产物自动按模型归档到 outputs/<dataset>/<model>/llamacpp/)
eval-vlm pred --datadir /path/to/test_images \
  --backend llamacpp \
  --llamacpp-base-url http://127.0.0.1:8080/v1 \
  --llamacpp-model qwen2.5-vl-7b
```

### 2. 标准评测与多轮打分 (eval = pred + score)
```bash
# 评测带标注数据集（支持多轮对话 rollout）
eval-vlm eval --dataset my_test_dataset \
  --backend llamacpp \
  --llamacpp-base-url http://127.0.0.1:8080/v1 \
  --llamacpp-model qwen2.5-vl-7b \
  --targets all
```

### 3. 字段级精准评估 (field-eval)
```bash
# 若本地尚无当前后端的预测结果，系统将自动使用 llamacpp 现跑预测，并调服务抽取主辅路等字段精准比对
eval-vlm field-eval --dataset my_test_dataset \
  --backend llamacpp \
  --llamacpp-base-url http://127.0.0.1:8080/v1 \
  --llamacpp-model qwen2.5-vl-7b \
  --targets 1
```

### 4. 离线 CLI 模式 (备用，无需启动 HTTP 服务)
```bash
eval-vlm pred --dataset my_test_dataset \
  --backend llamacpp \
  --llamacpp-mode cli \
  --llamacpp-model-path /models/qwen2.5-vl-7b-instruct-q4_k_m.gguf \
  --llamacpp-mmproj /models/mmproj-qwen2.5-vl-7b-instruct-f16.gguf
```
