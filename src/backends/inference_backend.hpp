#pragma once
#include <vector>
#include <string>
#include <memory>
#include <span>
#include <filesystem>

namespace rfdetr {
namespace backend {

/**
 * @brief Abstract base class for inference backends (Strategy Pattern)
 * 
 * This interface allows different inference engines (ONNX Runtime, TensorRT, etc.)
 * to be used interchangeably. Each backend implements how to:
 * - Load and initialize models
 * - Run inference
 * - Extract output tensor data
 */
class InferenceBackend {
public:
    virtual ~InferenceBackend() = default;

    /**
     * @brief Initialize the backend with a model file
     * @param model_path Path to the model file
     * @param input_shape Expected input shape [batch, channels, height, width]
     * @return Actual input shape detected from the model (for auto-detection)
     */
    virtual std::vector<int64_t> initialize(
        const std::filesystem::path& model_path,
        const std::vector<int64_t>& input_shape
    ) = 0;

    /**
     * @brief Run inference on input data
     * @param input_data Preprocessed input data (flattened)
     * @param input_shape Shape of the input tensor
     * @return Vector of output tensors (backend-specific type)
     */
    virtual std::vector<void*> run_inference(
        std::span<const float> input_data,
        const std::vector<int64_t>& input_shape
    ) = 0;

    /**
     * @brief Get the number of output tensors
     * @return Number of outputs from the model
     */
    virtual size_t get_output_count() const = 0;

    /**
     * @brief Get output tensor data as float array
     * @param output_index Index of the output tensor
     * @param data Output buffer to fill with tensor data
     * @param size Expected size of the output tensor
     */
    virtual void get_output_data(
        size_t output_index,
        float* data,
        size_t size
    ) = 0;

    /**
     * @brief Get output tensor shape
     * @param output_index Index of the output tensor
     * @return Shape of the output tensor
     */
    virtual std::vector<int64_t> get_output_shape(size_t output_index) const = 0;

    /**
     * @brief Get the backend name (for logging/debugging)
     * @return String identifying the backend type
     */
    virtual std::string get_backend_name() const = 0;

protected:
    // Cache for output tensors (backend-specific)
    std::vector<void*> output_tensors_;
};

/**
 * @brief Factory function to create backend instance (compile-time selection)
 * @return Unique pointer to the created backend
 * @throws std::runtime_error if no backend is available
 * 
 * The backend is determined at compile time based on USE_ONNX_RUNTIME or USE_TENSORRT
 * preprocessor definitions. Only one backend is compiled into the binary.
 */
std::unique_ptr<InferenceBackend> create_backend();

} // namespace backend
} // namespace rfdetr
