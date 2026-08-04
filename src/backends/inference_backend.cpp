#include "inference_backend.hpp"

#include <stdexcept>

#ifdef USE_ONNX_RUNTIME
#include "onnx_runtime_backend.hpp"
#endif

#ifdef USE_TENSORRT
#include "tensorrt_backend.hpp"
#endif

namespace rfdetr::backend {

namespace {

[[noreturn]] void no_device_io(const char *what) {
    throw std::runtime_error(std::string("Backend does not support device I/O: ") + what +
                             " is unavailable. Check supports_device_io() before calling it.");
}

} // namespace

void InferenceBackend::run_inference_device(const void * /*input_device*/,
                                            const std::vector<int64_t> & /*input_shape*/) {
    no_device_io("run_inference_device");
}

void *InferenceBackend::get_input_device_ptr() const { no_device_io("get_input_device_ptr"); }

const void *InferenceBackend::get_output_device_ptr(size_t /*output_index*/) const {
    no_device_io("get_output_device_ptr");
}

std::unique_ptr<InferenceBackend> create_backend() {
#ifdef USE_ONNX_RUNTIME
    return std::make_unique<OnnxRuntimeBackend>();
#elif defined(USE_TENSORRT)
    return std::make_unique<TensorRTBackend>();
#else
#error "No backend enabled. Build with -DUSE_ONNX_RUNTIME=ON or -DUSE_TENSORRT=ON"
#endif
}

} // namespace rfdetr::backend
