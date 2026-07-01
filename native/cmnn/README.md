# cmnn 原生后端(C++ 批量 MNN 推理)

`cmnn` 后端在 C++ 层起 **N 个 `MNN::Transformer::Llm` 实例 + 线程池**,一次并行处理
整批 (图, prompt),绕开 pymnn「原生推理不释放 GIL + 单实例有状态只能串行」的瓶颈。
功能与 `mnn`(pymnn)后端一致(单图多模态、采样/重复惩罚、超大图缩放、退化检测)。

- Python 层:`src/eval_vlm/inference/cmnn_backend.py`(组 prompt、净化图片路径、下发配置、退化检测)。
- C++ 扩展:本目录 `engine.cpp` → 编译出 `cmnn_native` 模块。
- 契约见 `engine.cpp` 顶部注释与 `cmnn_backend.py`。

> **目标平台:x86 Linux 服务器。** 本机 Windows 无法编译/验证原生库;以下步骤在 Linux 上执行。

---

## 0. 先跑可行性 spike(强烈建议)

多实例并发是本方案的核心假设。正式用前,先用 `spike/concurrency_spike.cpp` 验证
「同进程 N 个 Llm 实例并发」稳定、内存可控、确有加速(见该文件顶部编译命令)。

```bash
./concurrency_spike /path/to/qwen2-vl-mnn/config.json /path/to/test.jpg 4 8
```

关注:进程是否崩溃、加速比是否明显、加载 N 实例后的内存/显存(`ps`/`nvidia-smi`)。
不通过则回头讨论方案,不要硬上原生库。

---

## 1. 编译带 LLM 的 MNN

```bash
git clone https://github.com/alibaba/MNN.git
cd MNN && mkdir build && cd build
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DMNN_BUILD_LLM=ON \
  -DMNN_BUILD_LLM_OMNI=ON \
  -DMNN_LOW_MEMORY=ON \
  -DMNN_BUILD_OPENCV=ON -DMNN_IMGCODECS=ON   # 提供 MNN::CV::imread
  # GPU 可选:-DMNN_OPENCL=ON 或 -DMNN_CUDA=ON
make -j$(nproc)
# 记下:MNN_ROOT=<MNN 源码根>  MNN_BUILD_DIR=<该 build 目录>
```

## 2. 编译 cmnn_native 扩展

```bash
pip install pybind11

cmake -S native/cmnn -B build/cmnn \
  -DMNN_ROOT=$MNN_ROOT \
  -DMNN_BUILD_DIR=$MNN_BUILD_DIR \
  -DPython3_EXECUTABLE=$(which python) \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build/cmnn -j
```

产物:`build/cmnn/cmnn_native.cpython-*.so`。让 Python 能 `import cmnn_native`:

```bash
export PYTHONPATH=$PWD/build/cmnn:$PYTHONPATH
# 或复制到 site-packages / 项目根
python -c "import cmnn_native; print('cmnn_native OK')"
```

## 3. 使用

```bash
eval-vlm pred --datadir /path/to/images \
  --backend cmnn \
  --cmnn-config /path/to/qwen2-vl-mnn/config.json \
  --cmnn-num-workers 4 \
  --cmnn-batch-size 16
```

或在数据集 `config.yaml` 的 `inference` 段设 `backend: cmnn` 并填 `cmnn.config_path`
(参数含义见模板注释)。`--backend/--cmnn-*` 会永久写回该 `config.yaml`。

---

## 版本适配备忘(不同 MNN 版本可能微调)

`engine.cpp` / `CMakeLists.txt` 里以下点若编译报错,按你的 MNN 版本调整:

- **头文件路径**:`<llm/llm.hpp>`、`<cv/cv.hpp>`。有的版本 cv 头在 `include/` 下或
  命名空间为 `MNN::CV` vs `CV`。以 `transformers/llm/engine/include/llm/llm.hpp` 为准
  (本仓库 `engine.cpp` 已对齐该头的公开 API)。
- **`libllm`**:可能是独立 `.so`/`.a`,也可能已并入 `libMNN`;CMake 里 `MNN_LLM_LIB`
  找不到时不致命,若链接报未定义符号,手动把 `libllm` 路径加进 `MNN_BUILD_DIR` 搜索。
- **多模态 map key**:`MultimodalPrompt.images` 的 key 需与 prompt 文本里
  `<img>KEY</img>` 一致。本工具固定单图,Python 侧生成 `<img>image_0</img>`,故 key
  用 `"image_0"`。若你的 MNN 版本要求别的引用方式(如按顺序、或不同标签),在
  `engine.cpp` 的 `kImageKey` 与 `cmnn_backend._build_query_text` 两处同步改。
- **`MNN::CV::imread` 空图判定**:本实现读 `getInfo()` 取 H/W;若你的版本 imread 失败
  不返回 null 而是空 Var,`process_one` 里加更强的校验(参考 pymnn 后端对非法 Var 的处理)。

## 与 pymnn 后端的一致性验证

编译通过后,建议对同一批图分别用 `--backend mnn` 与 `--backend cmnn` 各跑一遍,
对拍 `predictions.jsonl`(同一图的描述应高度一致或语义等价)+ 比较吞吐,确认功能对齐。
