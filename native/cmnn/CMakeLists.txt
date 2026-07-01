# cmnn_native —— eval_vlm 的 C++ 批量 MNN 推理扩展构建脚本。
#
# 依赖:① 已编译好的 MNN(开启 LLM:-DMNN_BUILD_LLM=ON,多模态再加 -DMNN_BUILD_LLM_OMNI=ON);
#      ② pybind11(pip install pybind11,或系统包)。
#
# 典型用法(Linux,见 README.md 完整步骤):
#   cmake -S native/cmnn -B build/cmnn \
#         -DMNN_ROOT=/path/to/MNN \
#         -DMNN_BUILD_DIR=/path/to/MNN/build \
#         -DPython3_EXECUTABLE=$(which python) \
#         -DCMAKE_BUILD_TYPE=Release
#   cmake --build build/cmnn -j
# 产物:build/cmnn/cmnn_native*.so —— 放到 PYTHONPATH 或 site-packages 即可被 import。

cmake_minimum_required(VERSION 3.18)
project(cmnn_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# ---- 路径入参 ----------------------------------------------------------------
# MNN_ROOT       : MNN 源码根(含 include/ 与 transformers/llm/engine/include/)
# MNN_BUILD_DIR  : MNN 的构建输出目录(含 libMNN.so / libllm.so 或 .a)
set(MNN_ROOT "" CACHE PATH "MNN 源码根目录")
set(MNN_BUILD_DIR "" CACHE PATH "MNN 构建输出目录(含 libMNN / libllm)")

if(NOT MNN_ROOT)
  message(FATAL_ERROR "请传入 -DMNN_ROOT=<MNN 源码根>")
endif()
if(NOT MNN_BUILD_DIR)
  message(FATAL_ERROR "请传入 -DMNN_BUILD_DIR=<MNN 构建输出目录>")
endif()

# ---- pybind11 ----------------------------------------------------------------
# 优先用 pip 装的 pybind11(其 CMake 配置随包分发);找不到再退回 find_package。
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
execute_process(
  COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
  OUTPUT_VARIABLE _pybind11_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE _pybind11_rc)
if(_pybind11_rc EQUAL 0)
  list(APPEND CMAKE_PREFIX_PATH "${_pybind11_dir}")
endif()
find_package(pybind11 CONFIG REQUIRED)

# ---- MNN 头文件 / 库 ----------------------------------------------------------
set(MNN_INCLUDE_DIRS
  "${MNN_ROOT}/include"
  "${MNN_ROOT}/transformers/llm/engine/include"   # <llm/llm.hpp>
  "${MNN_ROOT}/tools/cv/include"                  # <cv/cv.hpp>(MNN::CV);版本不同可能在 include/ 下
)

find_library(MNN_LIB NAMES MNN libMNN PATHS "${MNN_BUILD_DIR}" REQUIRED)
# libllm 可能是独立库(共享或静态),也可能已并入 libMNN;找不到不致命。
find_library(MNN_LLM_LIB NAMES llm libllm PATHS "${MNN_BUILD_DIR}"
             "${MNN_BUILD_DIR}/transformers/llm/engine")

# ---- 目标 --------------------------------------------------------------------
pybind11_add_module(cmnn_native engine.cpp)
target_include_directories(cmnn_native PRIVATE ${MNN_INCLUDE_DIRS})
target_link_libraries(cmnn_native PRIVATE ${MNN_LIB})
if(MNN_LLM_LIB)
  target_link_libraries(cmnn_native PRIVATE ${MNN_LLM_LIB})
endif()
find_package(Threads REQUIRED)
target_link_libraries(cmnn_native PRIVATE Threads::Threads)

# 运行期能找到 libMNN/libllm(避免每次都要 LD_LIBRARY_PATH)。
set_target_properties(cmnn_native PROPERTIES
  BUILD_RPATH "${MNN_BUILD_DIR}"
  INSTALL_RPATH "${MNN_BUILD_DIR}")
