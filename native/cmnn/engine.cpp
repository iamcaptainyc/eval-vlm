// cmnn_native —— eval_vlm 的 C++ 批量 MNN 推理原生扩展。
//
// 目标:绕开 pymnn 的「原生推理不释放 GIL + 单个 Llm 对象有状态只能串行」瓶颈,
// 在 C++ 层起 num_workers 个独立的 MNN::Transformer::Llm 实例 + 线程池,一次吃下
// 整批 (图, prompt),真并行处理后按原顺序返回。功能对齐 pymnn 后端(单图多模态、
// set_config 下发采样/上限、返回文本 + 统计)。
//
// 与 Python 侧(src/eval_vlm/inference/cmnn_backend.py)的契约:
//   Engine(config_path: str, num_workers: int)
//   .set_config(json: str) -> bool            # 应用到所有实例
//   .generate_batch(list[dict]) -> list[dict] # 等长同序;每条 {text, image_path, max_new_tokens}
//                                             # 返回 {text, prompt_len, gen_seq_len, vision_us,
//                                             #       prefill_us, decode_us, pixels_mp, latency, [error]}
//   .close()
//
// 关键点:
//   * 每个 Llm 有状态(KV cache/history),**绝不并发调用同一实例**。线程池按
//     「一线程绑定一实例」分发,天然满足。
//   * 图片路径已由 Python 侧净化(.webp 转码 / 超大图缩放到临时 PNG),这里直接
//     MNN::CV::imread 原生解码即可。
//   * 每条请求前 reset() 清空上一条的 history/KV,保证各图独立单轮。
//   * 重活期间释放 GIL(py::gil_scoped_release),让 Python 侧不被阻塞;解析入参与
//     构造出参在持 GIL 时完成。
//
// 注意(不同 MNN 版本可能微调,见 native/cmnn/README.md):
//   - LLM 头文件路径 <llm/llm.hpp>、cv 头文件 <cv/cv.hpp> 与 MNN::CV 命名空间;
//   - MultimodalPrompt.images 的 map key 需与 prompt 文本里的 <img>KEY</img> 一致
//     (本工具固定单图,Python 侧生成 <img>image_0</img>,故 key = "image_0")。

#include <atomic>
#include <chrono>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <cv/cv.hpp>          // MNN::CV::imread
#include <llm/llm.hpp>        // MNN::Transformer::Llm

namespace py = pybind11;
using MNN::Transformer::Llm;
using MNN::Transformer::MultimodalPrompt;
using MNN::Transformer::PromptImagePart;

namespace {

// 单条请求(从 Python dict 解析而来,持 GIL 时构造)。
struct Request {
    std::string text;         // 已含 <img>image_0</img> 的多模态 prompt
    std::string image_path;   // 已净化、MNN::CV::imread 可直接读的本地路径
    int max_new_tokens = -1;
};

// 单条结果(线程池填充,释放 GIL 时构造;之后回 Python)。
struct Result {
    std::string text;
    int prompt_len = 0;
    int gen_seq_len = 0;
    long long vision_us = 0;
    long long prefill_us = 0;
    long long decode_us = 0;
    float pixels_mp = 0.f;
    double latency = 0.0;
    bool ok = true;
    std::string error;
};

// 固定单图占位:与 Python 侧 <img>image_0</img> 对应。
constexpr const char* kImageKey = "image_0";

// 用一个 Llm 实例处理一条请求,异常收进 Result.error(绝不抛出,避免带崩整批)。
void process_one(Llm* llm, const Request& req, Result& out) {
    auto t0 = std::chrono::steady_clock::now();
    try {
        // 每条独立单轮:先清掉上一条的 history/KV,防止串话。
        llm->reset();

        // 原生解码图片(路径已净化)。imread 返回 HWC 的 VARP。
        auto image = MNN::CV::imread(req.image_path);
        if (image == nullptr) {
            out.ok = false;
            out.error = "imread failed: " + req.image_path;
            return;
        }
        auto shape = image->getInfo();
        int height = 0, width = 0;
        if (shape != nullptr && shape->dim.size() >= 2) {
            height = shape->dim[0];
            width = shape->dim[1];
        }

        // 组多模态 prompt:text 里的 <img>image_0</img> 引用 images["image_0"]。
        MultimodalPrompt mm;
        mm.prompt_template = req.text;
        PromptImagePart part;
        part.image_data = image;
        part.width = width;
        part.height = height;
        mm.images[kImageKey] = part;

        // stream 到内存,拿到完整生成文本(而非打到 stdout)。
        std::ostringstream oss;
        llm->response(mm, &oss, nullptr, req.max_new_tokens);
        out.text = oss.str();

        // 收集统计(与 pymnn 后端一致的字段)。
        const auto* ctx = llm->getContext();
        if (ctx != nullptr) {
            out.prompt_len = ctx->prompt_len;
            out.gen_seq_len = ctx->gen_seq_len;
            out.vision_us = ctx->vision_us;
            out.prefill_us = ctx->prefill_us;
            out.decode_us = ctx->decode_us;
            out.pixels_mp = ctx->pixels_mp;
        }
        out.ok = true;
    } catch (const std::exception& e) {
        out.ok = false;
        out.error = e.what();
    } catch (...) {
        out.ok = false;
        out.error = "unknown native error";
    }
    auto t1 = std::chrono::steady_clock::now();
    out.latency = std::chrono::duration<double>(t1 - t0).count();
}

class Engine {
public:
    Engine(const std::string& config_path, int num_workers) {
        if (num_workers < 1) {
            throw std::invalid_argument("num_workers must be >= 1");
        }
        config_path_ = config_path;
        // 起 num_workers 个独立 Llm 实例。第 0 个正常 createLLM;其余尽量共享其
        // 运行时/模块以省内存(不同 MNN 版本共享能力不一,失败则各自独立加载)。
        for (int i = 0; i < num_workers; ++i) {
            Llm* llm = Llm::createLLM(config_path);
            if (llm == nullptr) {
                throw std::runtime_error("Llm::createLLM returned null: " + config_path);
            }
            if (!llm->load()) {
                Llm::destroy(llm);
                throw std::runtime_error("Llm::load failed for instance " + std::to_string(i));
            }
            instances_.push_back(llm);
        }
    }

    ~Engine() { close(); }

    // 把一份 MNN 原生 JSON 配置下发给所有实例(采样/max_new_tokens 等)。
    bool set_config(const std::string& json) {
        bool all_ok = true;
        for (auto* llm : instances_) {
            all_ok = llm->set_config(json) && all_ok;
        }
        return all_ok;
    }

    // 批量推理:输入 list[dict],返回等长同序 list[dict]。
    py::list generate_batch(py::list py_requests) {
        // ---- 持 GIL:解析入参 ----
        std::vector<Request> reqs;
        reqs.reserve(py::len(py_requests));
        for (auto item : py_requests) {
            auto d = item.cast<py::dict>();
            Request r;
            r.text = d.contains("text") ? d["text"].cast<std::string>() : std::string();
            r.image_path = d.contains("image_path") ? d["image_path"].cast<std::string>()
                                                     : std::string();
            r.max_new_tokens = d.contains("max_new_tokens")
                                   ? d["max_new_tokens"].cast<int>()
                                   : -1;
            reqs.push_back(std::move(r));
        }

        std::vector<Result> results(reqs.size());

        // ---- 释放 GIL:线程池并行推理(一线程绑定一实例)----
        {
            py::gil_scoped_release release;
            run_pool(reqs, results);
        }

        // ---- 重新持 GIL:构造出参 ----
        py::list out;
        for (const auto& r : results) {
            py::dict d;
            if (!r.ok) {
                d["error"] = r.error;
            } else {
                d["text"] = r.text;
                d["prompt_len"] = r.prompt_len;
                d["gen_seq_len"] = r.gen_seq_len;
                d["vision_us"] = r.vision_us;
                d["prefill_us"] = r.prefill_us;
                d["decode_us"] = r.decode_us;
                d["pixels_mp"] = r.pixels_mp;
            }
            d["latency"] = r.latency;
            out.append(std::move(d));
        }
        return out;
    }

    void close() {
        for (auto* llm : instances_) {
            if (llm != nullptr) {
                Llm::destroy(llm);
            }
        }
        instances_.clear();
    }

private:
    // 一线程绑定一实例,用原子游标抢占请求下标 —— 保证同一实例串行、不同实例并行。
    void run_pool(const std::vector<Request>& reqs, std::vector<Result>& results) {
        std::atomic<size_t> next{0};
        const size_t n = reqs.size();
        auto worker = [&](Llm* llm) {
            for (;;) {
                size_t idx = next.fetch_add(1);
                if (idx >= n) break;
                process_one(llm, reqs[idx], results[idx]);
            }
        };
        std::vector<std::thread> pool;
        pool.reserve(instances_.size());
        for (auto* llm : instances_) {
            pool.emplace_back(worker, llm);
        }
        for (auto& t : pool) {
            t.join();
        }
    }

    std::string config_path_;
    std::vector<Llm*> instances_;
};

}  // namespace

PYBIND11_MODULE(cmnn_native, m) {
    m.doc() = "eval_vlm C++ batch MNN inference backend (multi-instance thread pool)";
    py::class_<Engine>(m, "Engine")
        .def(py::init<const std::string&, int>(), py::arg("config_path"), py::arg("num_workers"))
        .def("set_config", &Engine::set_config, py::arg("json"))
        .def("generate_batch", &Engine::generate_batch, py::arg("requests"))
        .def("close", &Engine::close);
}
