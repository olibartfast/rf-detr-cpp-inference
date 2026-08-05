#include "gpu_context.hpp"

#if defined(USE_CUDA_POSTPROCESS) || defined(USE_DALI)

#include "cuda_check.hpp"

#include <cuda_runtime_api.h>
#include <utility>

namespace rfdetr::gpu {

namespace {

cudaStream_t as_stream(StreamHandle handle) noexcept { return static_cast<cudaStream_t>(handle); }

} // namespace

bool device_available() noexcept {
    int count = 0;
    // cudaGetDeviceCount is the one CUDA call that must not throw: it is the
    // probe used to decide whether anything else may run at all.
    if (cudaGetDeviceCount(&count) != cudaSuccess) {
        cudaGetLastError(); // clear the sticky error so later probes are clean
        return false;
    }
    return count > 0;
}

std::string device_name(int device_id) {
    cudaDeviceProp props{};
    if (cudaGetDeviceProperties(&props, device_id) != cudaSuccess) {
        cudaGetLastError();
        return {};
    }
    return props.name;
}

GpuContext::GpuContext(int device_id) : device_id_(device_id) {
    CUDA_CHECK(cudaSetDevice(device_id_));
    cudaStream_t cuda_stream = nullptr;
    // Non-blocking so this stream never implicitly synchronises with the legacy
    // default stream, which other libraries in the process may still use.
    CUDA_CHECK(cudaStreamCreateWithFlags(&cuda_stream, cudaStreamNonBlocking));
    stream_ = cuda_stream;
}

GpuContext::~GpuContext() {
    if (stream_ != nullptr) {
        cudaStreamDestroy(as_stream(stream_));
    }
}

void GpuContext::synchronize() const { CUDA_CHECK(cudaStreamSynchronize(as_stream(stream_))); }

DeviceBuffer::~DeviceBuffer() {
    if (ptr_ != nullptr) {
        cudaFree(ptr_);
    }
}

DeviceBuffer::DeviceBuffer(DeviceBuffer &&other) noexcept
    : ptr_(std::exchange(other.ptr_, nullptr)), capacity_(std::exchange(other.capacity_, 0)) {}

DeviceBuffer &DeviceBuffer::operator=(DeviceBuffer &&other) noexcept {
    if (this != &other) {
        if (ptr_ != nullptr) {
            cudaFree(ptr_);
        }
        ptr_ = std::exchange(other.ptr_, nullptr);
        capacity_ = std::exchange(other.capacity_, 0);
    }
    return *this;
}

void DeviceBuffer::reserve(size_t bytes) {
    if (bytes <= capacity_) {
        return;
    }
    if (ptr_ != nullptr) {
        CUDA_CHECK(cudaFree(ptr_));
        ptr_ = nullptr;
        capacity_ = 0;
    }
    CUDA_CHECK(cudaMalloc(&ptr_, bytes));
    capacity_ = bytes;
}

void copy_h2d(void *dst, const void *src, size_t bytes, StreamHandle stream) {
    CUDA_CHECK(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, as_stream(stream)));
}

void copy_d2h(void *dst, const void *src, size_t bytes, StreamHandle stream) {
    CUDA_CHECK(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, as_stream(stream)));
}

void copy_d2d(void *dst, const void *src, size_t bytes, StreamHandle stream) {
    CUDA_CHECK(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, as_stream(stream)));
}

void memset_device(void *dst, int value, size_t bytes, StreamHandle stream) {
    CUDA_CHECK(cudaMemsetAsync(dst, value, bytes, as_stream(stream)));
}

void stream_synchronize(StreamHandle stream) { CUDA_CHECK(cudaStreamSynchronize(as_stream(stream))); }

} // namespace rfdetr::gpu

#endif // USE_CUDA_POSTPROCESS || USE_DALI
