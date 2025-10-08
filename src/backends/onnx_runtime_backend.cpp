#ifdef USE_ONNX_RUNTIME

#include "onnx_runtime_backend.hpp"
#include <stdexcept>
#include <iostream>

namespace rfdetr {
namespace backend {

OnnxRuntimeBackend::OnnxRuntimeBackend()
    : env_(std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "RFDETRInference")),
      memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {
}

std::vector<int64_t> OnnxRuntimeBackend::initialize(
    const std::filesystem::path& model_path,
    const std::vector<int64_t>& input_shape
) {
    // Validate model path
    if (!std::filesystem::exists(model_path)) {
        throw std::runtime_error("Model file does not exist: " + model_path.string());
    }

    // Initialize ONNX Runtime session
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    
    session_ = std::make_unique<Ort::Session>(*env_, model_path.c_str(), session_options);

    // Auto-detect input shape from model if resolution is 0
    std::vector<int64_t> detected_shape = input_shape;
    if (input_shape[2] == 0 || input_shape[3] == 0) {
        Ort::TypeInfo input_type_info = session_->GetInputTypeInfo(0);
        auto tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
        auto shape = tensor_info.GetShape();
        
        if (shape.size() == 4 && shape[2] == shape[3] && shape[2] > 0) {
            detected_shape = shape;
            std::cout << "[ONNX Runtime] Auto-detected input resolution: " 
                      << shape[2] << "x" << shape[3] << std::endl;
        } else {
            throw std::runtime_error("Could not auto-detect valid input resolution from model.");
        }
    }

    // Get output names from model
    const size_t num_outputs = session_->GetOutputCount();
    std::cout << "[ONNX Runtime] Model has " << num_outputs << " outputs:" << std::endl;
    
    output_name_strings_.reserve(num_outputs);
    for (size_t i = 0; i < num_outputs; ++i) {
        Ort::AllocatedStringPtr output_name_ptr = session_->GetOutputNameAllocated(i, allocator_);
        std::string output_name(output_name_ptr.get());
        std::cout << "  Output " << i << ": " << output_name << std::endl;
        output_name_strings_.push_back(output_name);
    }
    
    // Get pointers after all strings are stored
    for (const auto& name : output_name_strings_) {
        output_names_.push_back(name.c_str());
    }

    return detected_shape;
}

std::vector<void*> OnnxRuntimeBackend::run_inference(
    std::span<const float> input_data,
    const std::vector<int64_t>& input_shape
) {
    // Create input tensor
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info_,
        const_cast<float*>(input_data.data()),
        input_data.size(),
        input_shape.data(),
        input_shape.size()
    );

    // Run inference
    ort_output_tensors_ = session_->Run(
        Ort::RunOptions{nullptr},
        &input_name_,
        &input_tensor,
        1,
        output_names_.data(),
        output_names_.size()
    );

    // Return void pointers (for interface compatibility)
    std::vector<void*> output_ptrs;
    output_ptrs.reserve(ort_output_tensors_.size());
    for (auto& tensor : ort_output_tensors_) {
        output_ptrs.push_back(&tensor);
    }
    
    return output_ptrs;
}

size_t OnnxRuntimeBackend::get_output_count() const {
    return ort_output_tensors_.size();
}

void OnnxRuntimeBackend::get_output_data(
    size_t output_index,
    float* data,
    size_t size
) {
    if (output_index >= ort_output_tensors_.size()) {
        throw std::out_of_range("Output index out of range");
    }

    const float* tensor_data = ort_output_tensors_[output_index].GetTensorData<float>();
    auto shape = ort_output_tensors_[output_index].GetTensorTypeAndShapeInfo().GetShape();
    
    size_t tensor_size = 1;
    for (auto dim : shape) {
        tensor_size *= dim;
    }
    
    if (tensor_size != size) {
        throw std::runtime_error("Output tensor size mismatch. Expected: " + 
                                 std::to_string(size) + ", Got: " + std::to_string(tensor_size));
    }
    
    std::copy(tensor_data, tensor_data + size, data);
}

std::vector<int64_t> OnnxRuntimeBackend::get_output_shape(size_t output_index) const {
    if (output_index >= ort_output_tensors_.size()) {
        throw std::out_of_range("Output index out of range");
    }
    
    return ort_output_tensors_[output_index].GetTensorTypeAndShapeInfo().GetShape();
}

} // namespace backend
} // namespace rfdetr

#endif // USE_ONNX_RUNTIME
