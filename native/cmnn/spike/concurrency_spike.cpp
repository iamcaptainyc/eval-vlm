// 并发可行性 spike —— 路径 B 的门槛验证(在 Linux 目标机上跑)。
//
// 目的:在**正式接线前**验证「同进程内起 N 个 MNN::Transformer::Llm 实例、N 条线程
// 并发跑」是否 ① 稳定不崩 ② 内存可控 ③ 相对单实例串行确有并行加速。这是 cmnn 后端
// (多实例并发方案)成败的关键假设;不通过则应回头重新讨论方案,而非硬写原生库。
//
// 用法:
//   concurrency_spike <config.json> <image_path> [num_workers=4] [iters_per_worker=8]
// 它对同一 (prompt, 图) 重复推理 num_workers*iters_per_worker 次,分别用
// 「1 实例串行」与「N 实例并发」各跑一遍,打印耗时与加速比;进程不崩即通过稳定性检验。
//
// 编译(示例,路径按你的 MNN 构建调整):
//   g++ -std=c++17 -O2 concurrency_spike.cpp \
//       -I$MNN_ROOT/include -I$MNN_ROOT/transformers/llm/engine/include \
//       -I$MNN_ROOT/tools/cv/include \
//       -L$MNN_BUILD_DIR -lMNN -lllm -lpthread \
//       -Wl,-rpath,$MNN_BUILD_DIR -o concurrency_spike

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <MNN/expr/Expr.hpp>
#include <cv/cv.hpp>
#include <llm/llm.hpp>

using MNN::Transformer::Llm;
using MNN::Transformer::MultimodalPrompt;
using MNN::Transformer::PromptImagePart;

namespace {

constexpr const char* kImageKey = "image_0";

// 跑一条推理(每次前 reset);出错抛异常,便于 spike 直接暴露问题。
std::string run_once(Llm* llm, const std::string& text, const std::string& image_path) {
    llm->reset();
    auto image = MNN::CV::imread(image_path);
    if (image == nullptr) {
        throw std::runtime_error("imread failed: " + image_path);
    }
    auto info = image->getInfo();
    int h = 0, w = 0;
    if (info != nullptr && info->dim.size() >= 2) {
        h = info->dim[0];
        w = info->dim[1];
    }
    MultimodalPrompt mm;
    mm.prompt_template = text;
    PromptImagePart part;
    part.image_data = image;
    part.width = w;
    part.height = h;
    mm.images[kImageKey] = part;

    std::ostringstream oss;
    llm->response(mm, &oss, nullptr, 64);
    return oss.str();
}

double seconds_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr,
                     "用法: %s <config.json> <image_path> [num_workers=4] [iters_per_worker=8]\n",
                     argv[0]);
        return 2;
    }
    const std::string config_path = argv[1];
    const std::string image_path = argv[2];
    const int num_workers = argc > 3 ? std::atoi(argv[3]) : 4;
    const int iters = argc > 4 ? std::atoi(argv[4]) : 8;
    const std::string text = "<img>image_0</img>请描述图片";
    const int total = num_workers * iters;

    std::printf("[spike] config=%s image=%s workers=%d total=%d\n",
                config_path.c_str(), image_path.c_str(), num_workers, total);

    // ---- 加载 N 个实例(观察内存:此时 RSS 应 ≈ N × 单实例) ----
    std::printf("[spike] 加载 %d 个 Llm 实例...\n", num_workers);
    std::vector<Llm*> instances;
    for (int i = 0; i < num_workers; ++i) {
        Llm* llm = Llm::createLLM(config_path);
        if (llm == nullptr || !llm->load()) {
            std::fprintf(stderr, "[spike] 实例 %d 加载失败\n", i);
            return 1;
        }
        instances.push_back(llm);
    }
    std::printf("[spike] 加载完成。请用 `ps`/`nvidia-smi` 记录此刻内存/显存占用。\n");

    // ---- A) 单实例串行跑 total 次 ----
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < total; ++i) {
        (void)run_once(instances[0], text, image_path);
    }
    double serial_s = seconds_since(t0);
    std::printf("[spike] 串行(1 实例 × %d 次): %.2fs, %.3f it/s\n",
                total, serial_s, total / serial_s);

    // ---- B) N 实例并发跑 total 次(一线程绑定一实例,原子游标抢占) ----
    std::atomic<int> next{0};
    std::atomic<int> errors{0};
    auto worker = [&](Llm* llm) {
        for (;;) {
            int idx = next.fetch_add(1);
            if (idx >= total) break;
            try {
                (void)run_once(llm, text, image_path);
            } catch (const std::exception& e) {
                errors.fetch_add(1);
                std::fprintf(stderr, "[spike] 并发第 %d 次出错: %s\n", idx, e.what());
            }
        }
    };
    auto t1 = std::chrono::steady_clock::now();
    std::vector<std::thread> pool;
    for (auto* llm : instances) pool.emplace_back(worker, llm);
    for (auto& t : pool) t.join();
    double par_s = seconds_since(t1);

    std::printf("[spike] 并发(%d 实例): %.2fs, %.3f it/s, 错误 %d 次\n",
                num_workers, par_s, total / par_s, errors.load());
    std::printf("[spike] 加速比 ≈ %.2fx(理想上限 %d)。\n", serial_s / par_s, num_workers);
    std::printf("[spike] 结论:进程未崩 + 加速比明显 + 内存可控 => 路径 B 可行。\n");

    for (auto* llm : instances) Llm::destroy(llm);
    return errors.load() == 0 ? 0 : 1;
}
