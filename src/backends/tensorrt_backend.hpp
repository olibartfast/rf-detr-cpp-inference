#pragma once

#ifdef USE_TENSORRT

#include "inference_backend.hpp"
#include <NvInfer.h>
#include <NvInferVersion.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>
#include <memory>

// Check TensorRT version compatibility
#if NV_TENSORRT_MAJOR < 8
    #error "TensorRT 8.0 or newer is required"
#endif

namespace rfdetr {
namespace backend {

// Custom deleter for TensorRT objects
struct TensorRTDeleter {
    template<typename T>
    void operator()(T* obj) const {
        if (obj) {
#if NV_TENSORRT_MAJOR >= 10
            // TensorRT 10+ uses proper RAII, just delete
            delete obj;
#else
            // TensorRT 8.x and 9.x use destroy()
            obj->destroy();
#endif
        }
    }
};

/**
 * @brief TensorRT implementation of InferenceBackend
 * 
 * This backend uses NVIDIA TensorRT for optimized GPU inference.
 * Provides low-latency and high-throughput inference on NVIDIA GPUs.
 */
class TensorRTBackend : public InferenceBackend {
public:
    TensorRTBackend();
    ~TensorRTBackend() override;

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
        return "TensorRT";
    }

private:
    // TensorRT logger
    class Logger : public nvinfer1::ILogger {
    public:
        void log(Severity severity, const char* msg) noexcept override;
    };

    // Build TensorRT engine from ONNX model
    bool build_engine_from_onnx(
        const std::filesystem::path& model_path,
        const std::vector<int64_t>& input_shape
    );

    // Serialize and save engine to file
    void serialize_engine(const std::filesystem::path& engine_path);

    // Deserialize engine from file
    bool deserialize_engine(const std::filesystem::path& engine_path);

    Logger logger_;
    std::unique_ptr<nvinfer1::IRuntime, TensorRTDeleter> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine, TensorRTDeleter> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext, TensorRTDeleter> context_;

    // CUDA buffers
    std::vector<void*> device_buffers_;
    std::vector<std::vector<float>> host_output_buffers_;
    
    // Tensor metadata
    std::vector<std::vector<int64_t>> output_shapes_;
    int input_binding_index_ = -1;
    std::vector<int> output_binding_indices_;
};

} // namespace backend
} // namespace rfdetr

#endif // USE_TENSORRT
