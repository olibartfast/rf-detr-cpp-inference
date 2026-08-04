#ifdef USE_EXECUTORCH

#include "executorch_backend.hpp"

#include <algorithm>
#include <iostream>
#include <numeric>
#include <stdexcept>

namespace rfdetr::backend {

namespace {

using executorch::aten::ScalarType;

/// method_meta reports sizes as int32; the InferenceBackend contract speaks int64.
std::vector<int64_t> to_int64_dims(executorch::runtime::Span<const int32_t> dims) {
    std::vector<int64_t> result;
    result.reserve(dims.size());
    std::transform(dims.begin(), dims.end(), std::back_inserter(result),
                   [](int32_t dim) { return static_cast<int64_t>(dim); });
    return result;
}

std::string shape_to_string(const std::vector<int64_t> &shape) {
    std::string out = "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        out += (i == 0 ? "" : ",") + std::to_string(shape[i]);
    }
    return out + "]";
}

/// Read output `index`'s shape from the loaded program, before any inference has run.
std::vector<int64_t> meta_output_shape(const executorch::runtime::MethodMeta &meta, size_t index) {
    const auto info = meta.output_tensor_meta(index);
    if (!info.ok()) {
        return {};
    }
    return to_int64_dims(info->sizes());
}

} // anonymous namespace

void ExecuTorchBackend::validate_output_order(const executorch::runtime::MethodMeta &meta) const {
    // RFDETRInference::postprocess_outputs() addresses outputs positionally: index 0 is the boxes
    // tensor (dets, [B,N,4]) and index 1 the class logits (labels, [B,N,num_classes]). ONNX Runtime
    // preserves that order through named outputs, but an ExecuTorch program returns an unnamed
    // tuple, so nothing downstream can detect a swap. Check it here, where method_meta still knows
    // the shapes, rather than silently decoding logits as coordinates.
    if (meta.num_outputs() < 2) {
        return; // RFDETRInference reports the too-few-outputs case with model-type context.
    }

    const auto boxes_shape = meta_output_shape(meta, 0);
    const auto logits_shape = meta_output_shape(meta, 1);
    if (boxes_shape.empty() || logits_shape.empty()) {
        std::cout << "[ExecuTorch] Output shape metadata unavailable; skipping output-order check." << std::endl;
        return;
    }

    const int64_t boxes_last = boxes_shape.back();
    const int64_t logits_last = logits_shape.back();

    // Only output 0 is checked. A swap puts logits [B,N,num_classes] at index 0, which this catches
    // for every num_classes except 4. Requiring output 1's trailing dimension to differ from 4 would
    // additionally reject a legitimate 4-class model, and would still not detect a swap in that case
    // — the two layouts are genuinely indistinguishable from shape alone.
    //
    // Deliberately not auto-swapping: a loud failure at load time is far cheaper to diagnose than
    // plausible-looking but wrong detections.
    if (boxes_last != 4) {
        throw std::runtime_error("Unexpected ExecuTorch output order. Expected output 0 = boxes [B,N,4] and output 1 = "
                                 "logits [B,N,num_classes], but got output 0 " +
                                 shape_to_string(boxes_shape) + " and output 1 " + shape_to_string(logits_shape) +
                                 ". Re-export the model with rfdetr >= 1.9.0 (dets first, then labels).");
    }

    if (logits_last == 4) {
        std::cout << "[ExecuTorch] Note: output 1 " << shape_to_string(logits_shape)
                  << " also ends in 4, so a 4-class model and swapped outputs look identical here. "
                     "Treating output 0 as boxes; verify detections if results look wrong."
                  << std::endl;
    }
}

std::vector<int64_t> ExecuTorchBackend::initialize(const std::filesystem::path &model_path,
                                                   const std::vector<int64_t> &input_shape) {
    if (!std::filesystem::exists(model_path)) {
        throw std::runtime_error("Model file does not exist: " + model_path.string());
    }

    module_ = std::make_unique<executorch::extension::Module>(model_path.string());

    const auto meta = module_->method_meta("forward");
    if (!meta.ok()) {
        throw std::runtime_error("ExecuTorch could not read forward() metadata from " + model_path.string() +
                                 ". Is this a valid .pte program?");
    }

    // Auto-detect input resolution when the caller passed 0 (mirrors OnnxRuntimeBackend).
    std::vector<int64_t> detected_shape = input_shape;
    if (input_shape.size() == 4 && (input_shape[2] == 0 || input_shape[3] == 0)) {
        if (meta->num_inputs() == 0) {
            throw std::runtime_error("ExecuTorch program declares no inputs; cannot auto-detect resolution.");
        }
        const auto input_meta = meta->input_tensor_meta(0);
        if (!input_meta.ok()) {
            throw std::runtime_error("ExecuTorch input metadata unavailable; cannot auto-detect resolution.");
        }
        const auto shape = to_int64_dims(input_meta->sizes());
        if (shape.size() == 4 && shape[2] == shape[3] && shape[2] > 0) {
            detected_shape = shape;
            std::cout << "[ExecuTorch] Auto-detected input resolution: " << shape[2] << "x" << shape[3] << std::endl;
        } else {
            throw std::runtime_error("Could not auto-detect valid input resolution from model. Input shape: " +
                                     shape_to_string(shape));
        }
    }

    output_count_ = meta->num_outputs();
    std::cout << "[ExecuTorch] Model has " << output_count_ << " outputs:" << std::endl;
    for (size_t i = 0; i < output_count_; ++i) {
        std::cout << "  Output " << i << ": " << shape_to_string(meta_output_shape(*meta, i)) << std::endl;
    }

    validate_output_order(*meta);

    return detected_shape;
}

std::vector<void *> ExecuTorchBackend::run_inference(std::span<const float> input_data,
                                                     const std::vector<int64_t> &input_shape) {
    if (!module_) {
        throw std::runtime_error("ExecuTorch backend used before initialize()");
    }

    std::vector<executorch::aten::SizesType> sizes;
    sizes.reserve(input_shape.size());
    std::transform(input_shape.begin(), input_shape.end(), std::back_inserter(sizes),
                   [](int64_t dim) { return static_cast<executorch::aten::SizesType>(dim); });

    const size_t expected = std::accumulate(input_shape.begin(), input_shape.end(), size_t{1},
                                            [](size_t acc, int64_t dim) { return acc * static_cast<size_t>(dim); });
    if (input_data.size() != expected) {
        throw std::runtime_error("Input tensor size mismatch. Expected: " + std::to_string(expected) +
                                 ", Got: " + std::to_string(input_data.size()));
    }

    // Non-owning view (no deleter): input_data outlives the forward() call below, so the per-frame
    // copy a value-owning tensor would cost is avoidable. const_cast is safe because ExecuTorch
    // only reads program inputs.
    auto input_tensor = executorch::extension::make_tensor_ptr(std::move(sizes), const_cast<float *>(input_data.data()),
                                                               ScalarType::Float);

    std::vector<executorch::runtime::EValue> inputs;
    inputs.emplace_back(*input_tensor);

    auto result = module_->forward(inputs);
    if (!result.ok()) {
        throw std::runtime_error("ExecuTorch forward() failed with error code " +
                                 std::to_string(static_cast<int>(result.error())));
    }
    output_values_ = std::move(*result);

    std::vector<void *> output_ptrs;
    output_ptrs.reserve(output_values_.size());
    std::transform(output_values_.begin(), output_values_.end(), std::back_inserter(output_ptrs),
                   [](auto &value) { return static_cast<void *>(&value); });

    return output_ptrs;
}

size_t ExecuTorchBackend::get_output_count() const { return output_count_; }

executorch::aten::Tensor ExecuTorchBackend::output_tensor(size_t output_index) const {
    if (output_index >= output_values_.size()) {
        throw std::out_of_range("Output index out of range");
    }
    if (!output_values_[output_index].isTensor()) {
        throw std::runtime_error("ExecuTorch output " + std::to_string(output_index) + " is not a tensor");
    }
    return output_values_[output_index].toTensor();
}

void ExecuTorchBackend::get_output_data(size_t output_index, float *data, size_t size) {
    const auto tensor = output_tensor(output_index);

    if (tensor.scalar_type() != ScalarType::Float) {
        throw std::runtime_error("ExecuTorch output " + std::to_string(output_index) +
                                 " is not float32; RF-DETR postprocessing requires float outputs.");
    }

    const auto tensor_size = static_cast<size_t>(tensor.numel());
    if (tensor_size != size) {
        throw std::runtime_error("Output tensor size mismatch. Expected: " + std::to_string(size) +
                                 ", Got: " + std::to_string(tensor_size));
    }

    const float *tensor_data = tensor.const_data_ptr<float>();
    std::copy(tensor_data, tensor_data + size, data);
}

std::vector<int64_t> ExecuTorchBackend::get_output_shape(size_t output_index) const {
    const auto tensor = output_tensor(output_index);
    std::vector<int64_t> shape;
    shape.reserve(static_cast<size_t>(tensor.dim()));
    for (ssize_t i = 0; i < tensor.dim(); ++i) {
        shape.push_back(static_cast<int64_t>(tensor.size(i)));
    }
    return shape;
}

} // namespace rfdetr::backend

#endif // USE_EXECUTORCH
