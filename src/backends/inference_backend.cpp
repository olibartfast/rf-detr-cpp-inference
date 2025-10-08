#include "inference_backend.hpp"
#include <stdexcept>

#ifdef USE_ONNX_RUNTIME
#include "onnx_runtime_backend.hpp"
#endif

#ifdef USE_TENSORRT
#include "tensorrt_backend.hpp"
#endif

namespace rfdetr {
namespace backend {

std::unique_ptr<InferenceBackend> create_backend() {
#ifdef USE_ONNX_RUNTIME
    return std::make_unique<OnnxRuntimeBackend>();
#elif defined(USE_TENSORRT)
    return std::make_unique<TensorRTBackend>();
#else
    #error "No backend enabled. Build with -DUSE_ONNX_RUNTIME=ON or -DUSE_TENSORRT=ON"
#endif
}

} // namespace backend
} // namespace rfdetr
