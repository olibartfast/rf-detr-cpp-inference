#pragma once

// Included only from translation units that already see the CUDA headers
// (gpu_context.cpp and the .cu kernels), never from the public GPU headers.

#include <cuda_runtime_api.h>
#include <stdexcept>
#include <string>

namespace rfdetr::gpu::detail {

inline void cuda_check(cudaError_t status, const char *expression, const char *file, int line) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(status) + " (" +
                                 cudaGetErrorName(status) + ") at " + file + ":" + std::to_string(line) + " in `" +
                                 expression + "`");
    }
}

} // namespace rfdetr::gpu::detail

#define CUDA_CHECK(expr) ::rfdetr::gpu::detail::cuda_check((expr), #expr, __FILE__, __LINE__)

/// Checks for an asynchronous launch failure. Call after every kernel launch —
/// launches report configuration errors here, not at the launch site.
#define CUDA_CHECK_LAST() ::rfdetr::gpu::detail::cuda_check(cudaGetLastError(), "kernel launch", __FILE__, __LINE__)
