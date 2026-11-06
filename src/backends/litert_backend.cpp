#ifdef USE_LITERT

#include "litert_backend.hpp"

#include "litert_contract.hpp"

#include <algorithm>
#include <cstring>
#include <litert/c/litert_compiled_model.h>
#include <litert/c/litert_environment.h>
#include <litert/c/litert_model.h>
#include <litert/c/litert_options.h>
#include <litert/c/litert_tensor_buffer.h>
#include <stdexcept>

namespace rfdetr::backend {
namespace {

void check(LiteRtStatus status, const char *operation) {
    if (status != kLiteRtStatusOk) {
        throw std::runtime_error(std::string("LiteRT: ") + operation + " failed (status " +
                                 std::to_string(static_cast<int>(status)) + ")");
    }
}

LiteRtRankedTensorType tensor_type(LiteRtTensor tensor) {
    LiteRtRankedTensorType type{};
    check(LiteRtGetRankedTensorType(tensor, &type), "read ranked tensor type");
    if (type.element_type != kLiteRtElementTypeFloat32 || type.layout.has_strides != 0 || type.layout.rank == 0 ||
        type.layout.rank > LITERT_TENSOR_MAX_RANK) {
        throw std::runtime_error("LiteRT: only contiguous ranked float32 I/O is supported");
    }
    return type;
}

std::vector<int64_t> tensor_shape(const LiteRtRankedTensorType &type) {
    std::vector<int64_t> shape;
    shape.reserve(type.layout.rank);
    for (unsigned int i = 0; i < type.layout.rank; ++i) {
        shape.push_back(type.layout.dimensions[i]);
    }
    return shape;
}

void copy_buffer(LiteRtTensorBuffer buffer, void *data, size_t bytes, bool write) {
    void *memory = nullptr;
    check(LiteRtLockTensorBuffer(buffer, &memory,
                                 write ? kLiteRtTensorBufferLockModeWrite : kLiteRtTensorBufferLockModeRead),
          "lock tensor buffer");
    // memcpy cannot throw; always release the lock before checking unlock status.
    if (write) {
        std::memcpy(memory, data, bytes);
    } else {
        std::memcpy(data, memory, bytes);
    }
    check(LiteRtUnlockTensorBuffer(buffer), "unlock tensor buffer");
}

} // namespace

struct LiteRTBackend::State {
    LiteRtEnvironment environment = nullptr;
    LiteRtModel model = nullptr;
    LiteRtOptions options = nullptr;
    LiteRtCompiledModel compiled = nullptr;
    LiteRtTensorBuffer input = nullptr;
    std::vector<LiteRtTensorBuffer> outputs;
    std::vector<int64_t> input_shape;
    std::vector<std::vector<int64_t>> output_shapes;
    std::vector<std::vector<float>> output_data;
    bool has_results = false;

    State() = default;
    State(const State &) = delete;
    State &operator=(const State &) = delete;
    State(State &&) = delete;
    State &operator=(State &&) = delete;
    ~State() {
        for (auto *buffer : outputs) {
            if (buffer != nullptr) {
                LiteRtDestroyTensorBuffer(buffer);
            }
        }
        if (input != nullptr) {
            LiteRtDestroyTensorBuffer(input);
        }
        if (compiled != nullptr) {
            LiteRtDestroyCompiledModel(compiled);
        }
        if (options != nullptr) {
            LiteRtDestroyOptions(options);
        }
        if (model != nullptr) {
            LiteRtDestroyModel(model);
        }
        if (environment != nullptr) {
            LiteRtDestroyEnvironment(environment);
        }
    }
};

LiteRTBackend::LiteRTBackend() = default;
LiteRTBackend::~LiteRTBackend() = default;

std::vector<int64_t> LiteRTBackend::initialize(const std::filesystem::path &model_path,
                                               const std::vector<int64_t> &input_shape) {
    // Build locally so a failed reinitialization leaves the previous model usable.
    auto next = std::make_unique<State>();
    check(LiteRtCreateEnvironment(0, nullptr, &next->environment), "create environment");
    check(LiteRtCreateModelFromFile(next->environment, model_path.string().c_str(), &next->model), "load model");
    LiteRtSignature signature = nullptr;
    check(LiteRtGetModelSignature(next->model, 0, &signature), "get default signature");
    LiteRtParamIndex input_count = 0;
    check(LiteRtGetNumSignatureInputs(signature, &input_count), "count inputs");
    if (input_count != 1) {
        throw std::runtime_error("LiteRT: expected exactly one image input");
    }
    LiteRtTensor tensor = nullptr;
    check(LiteRtGetSignatureInputTensorByIndex(signature, 0, &tensor), "get input");
    const auto input_type = tensor_type(tensor);
    next->input_shape = tensor_shape(input_type);
    litert_detail::validate_input(next->input_shape, input_shape);
    LiteRtParamIndex output_count = 0;
    check(LiteRtGetNumSignatureOutputs(signature, &output_count), "count outputs");
    if (output_count != 2 && output_count != 3) {
        throw std::runtime_error("LiteRT: expected two detection outputs or three segmentation/keypoint outputs");
    }
    std::vector<LiteRtRankedTensorType> output_types;
    for (LiteRtParamIndex i = 0; i < output_count; ++i) {
        check(LiteRtGetSignatureOutputTensorByIndex(signature, i, &tensor), "get output");
        output_types.push_back(tensor_type(tensor));
        next->output_shapes.push_back(tensor_shape(output_types.back()));
    }
    litert_detail::validate_outputs(next->output_shapes);
    check(LiteRtCreateOptions(&next->options), "create options");
    check(LiteRtSetOptionsHardwareAccelerators(next->options, kLiteRtHwAcceleratorCpu), "select CPU");
    check(LiteRtCreateCompiledModel(next->environment, next->model, next->options, &next->compiled), "compile model");
    LiteRtTensorBufferRequirements requirements = nullptr;
    check(LiteRtGetCompiledModelInputBufferRequirements(next->compiled, 0, 0, &requirements), "input requirements");
    check(LiteRtCreateManagedTensorBufferFromRequirements(next->environment, &input_type, requirements, &next->input),
          "allocate input");
    next->outputs.resize(output_count, nullptr);
    for (LiteRtParamIndex i = 0; i < output_count; ++i) {
        check(LiteRtGetCompiledModelOutputBufferRequirements(next->compiled, 0, i, &requirements),
              "output requirements");
        check(LiteRtCreateManagedTensorBufferFromRequirements(next->environment, &output_types[i], requirements,
                                                              &next->outputs[i]),
              "allocate output");
        next->output_data.emplace_back(litert_detail::element_count(next->output_shapes[i]));
    }
    state_ = std::move(next);
    return state_->input_shape;
}

std::vector<void *> LiteRTBackend::run_inference(std::span<const float> input_data,
                                                 const std::vector<int64_t> &input_shape) {
    if (!state_) {
        throw std::runtime_error("LiteRT: initialize before inference");
    }
    state_->has_results = false;
    if (input_shape != state_->input_shape || input_data.size() != litert_detail::element_count(input_shape)) {
        throw std::runtime_error("LiteRT: input shape or element count mismatch");
    }
    copy_buffer(state_->input, const_cast<float *>(input_data.data()), input_data.size_bytes(), true);
    check(
        LiteRtRunCompiledModel(state_->compiled, 0, 1, &state_->input, state_->outputs.size(), state_->outputs.data()),
        "run inference");
    std::vector<void *> result;
    result.reserve(state_->outputs.size());
    for (size_t i = 0; i < state_->outputs.size(); ++i) {
        auto &data = state_->output_data[i];
        copy_buffer(state_->outputs[i], data.data(), data.size() * sizeof(float), false);
        result.push_back(data.data());
    }
    state_->has_results = true;
    return result;
}

size_t LiteRTBackend::get_output_count() const { return state_ ? state_->outputs.size() : 0; }

std::vector<int64_t> LiteRTBackend::get_output_shape(size_t output_index) const {
    if (!state_) {
        throw std::runtime_error("LiteRT: initialize before reading outputs");
    }
    return state_->output_shapes.at(output_index);
}

void LiteRTBackend::get_output_data(size_t output_index, float *data, size_t size) {
    if (!state_ || !state_->has_results) {
        throw std::runtime_error("LiteRT: run inference before reading output data");
    }
    const auto &output = state_->output_data.at(output_index);
    if (data == nullptr || size != output.size()) {
        throw std::runtime_error("LiteRT: output destination or element count mismatch");
    }
    std::copy(output.begin(), output.end(), data);
}

} // namespace rfdetr::backend

#endif // USE_LITERT
