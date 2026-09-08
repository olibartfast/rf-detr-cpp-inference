#pragma once

#if defined(USE_CUDA_POSTPROCESS) || defined(USE_DALI)

#include <cstddef>
#include <string>

namespace rfdetr::gpu {

/// Opaque stand-in for `cudaStream_t`. Keeping CUDA out of this header lets the
/// inference and video-pipeline translation units — which are compiled without
/// the CUDA include path and, for the .cu consumers, by a different compiler —
/// pass streams around without seeing `cuda_runtime_api.h`.
using StreamHandle = void *;

/// True if a usable CUDA device is present. Tests gate on this and skip rather
/// than fail, because CI compiles the GPU targets but has no GPU.
[[nodiscard]] bool device_available() noexcept;

/// Human-readable name of `device_id`, or an empty string if unavailable.
[[nodiscard]] std::string device_name(int device_id = 0);

/// Owns a device selection and one non-default CUDA stream. Every GPU stage —
/// DALI output copy, TensorRT enqueue, postprocessing kernels — runs on this
/// stream, so no stage needs to synchronise against another.
class GpuContext {
  public:
    explicit GpuContext(int device_id = 0);
    ~GpuContext();

    GpuContext(const GpuContext &) = delete;
    GpuContext &operator=(const GpuContext &) = delete;
    GpuContext(GpuContext &&) = delete;
    GpuContext &operator=(GpuContext &&) = delete;

    [[nodiscard]] StreamHandle stream() const noexcept { return stream_; }
    [[nodiscard]] int device_id() const noexcept { return device_id_; }

    void synchronize() const;

  private:
    int device_id_{0};
    StreamHandle stream_{nullptr};
};

/// Grow-only device allocation. Sized on demand and kept at its high-water mark
/// so steady-state frames never call `cudaMalloc`.
class DeviceBuffer {
  public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(size_t bytes) { reserve(bytes); }
    ~DeviceBuffer();

    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;
    DeviceBuffer(DeviceBuffer &&other) noexcept;
    DeviceBuffer &operator=(DeviceBuffer &&other) noexcept;

    /// Ensures at least `bytes` of capacity. Reallocates (discarding contents)
    /// only when the request exceeds the current capacity.
    void reserve(size_t bytes);

    [[nodiscard]] void *get() const noexcept { return ptr_; }
    [[nodiscard]] size_t capacity() const noexcept { return capacity_; }

  private:
    void *ptr_{nullptr};
    size_t capacity_{0};
};

void copy_h2d(void *dst, const void *src, size_t bytes, StreamHandle stream);
void copy_d2h(void *dst, const void *src, size_t bytes, StreamHandle stream);
void copy_d2d(void *dst, const void *src, size_t bytes, StreamHandle stream);
void memset_device(void *dst, int value, size_t bytes, StreamHandle stream);
void stream_synchronize(StreamHandle stream);

} // namespace rfdetr::gpu

#endif // USE_CUDA_POSTPROCESS || USE_DALI
