#pragma once

#ifdef USE_ONNX_RUNTIME

#include "inference_backend.hpp"
#include <onnxruntime_cxx_api.h>
#include <memory>

namespace rfdetr {
namespace backend {

/**
 * @brief ONNX Runtime implementation of InferenceBackend
 * 
 * This backend uses Microsoft's ONNX Runtime for cross-platform inference.
 * Supports CPU and GPU execution providers.
 */
class OnnxRuntimeBackend : public InferenceBackend {
public:
    OnnxRuntimeBackend();
    ~OnnxRuntimeBackend() override = default;

    std::vector<int64_t> initialize(
        const std::filesystem::path& model_path,
        const std::vector<int64_t>& input_shape
    ) override;

    std::vector<void*> run_inference(
        std::span<const float> input_data,
        const std::vector<int64_t>& input_shape
    ) override;

    size_t get_output_count() const override;

    void get_output_data(
        size_t output_index,
        float* data,
        size_t size
    ) override;

    std::vector<int64_t> get_output_shape(size_t output_index) const override;

    std::string get_backend_name() const override {
        return "ONNX Runtime";
    }

private:
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
    Ort::AllocatorWithDefaultOptions allocator_;
    Ort::MemoryInfo memory_info_;
    
    const char* input_name_ = "input";
    std::vector<std::string> output_name_strings_;
    std::vector<const char*> output_names_;
    
    // Cache for output tensors
    std::vector<Ort::Value> ort_output_tensors_;
};

} // namespace backend
} // namespace rfdetr

#endif // USE_ONNX_RUNTIME
