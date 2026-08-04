#include "inference_backend.hpp"

#include <stdexcept>

#ifdef USE_ONNX_RUNTIME
#include "onnx_runtime_backend.hpp"
#endif

#ifdef USE_TENSORRT
#include "tensorrt_backend.hpp"
#endif

#ifdef USE_EXECUTORCH
#include "executorch_backend.hpp"
#endif

namespace rfdetr::backend {

std::unique_ptr<InferenceBackend> create_backend() {
#ifdef USE_ONNX_RUNTIME
    return std::make_unique<OnnxRuntimeBackend>();
#elif defined(USE_TENSORRT)
    return std::make_unique<TensorRTBackend>();
#elif defined(USE_EXECUTORCH)
    return std::make_unique<ExecuTorchBackend>();
#else
#error "No backend enabled. Build with -DUSE_ONNX_RUNTIME=ON, -DUSE_TENSORRT=ON, or -DUSE_EXECUTORCH=ON"
#endif
}

} // namespace rfdetr::backend
