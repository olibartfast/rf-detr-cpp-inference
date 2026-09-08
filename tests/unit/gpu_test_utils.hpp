#pragma once

// Shared scaffolding for the CUDA/DALI parity tests.
//
// CI compiles the GPU targets on runners that have no GPU, so every device
// test must SKIP rather than FAIL when no CUDA device is present. The macro
// and the device-serving mock backend below are shared by
// test_gpu_postprocess.cpp and test_gpu_parity.cpp.

#if defined(USE_CUDA_POSTPROCESS) || defined(USE_DALI)

#include "gpu/gpu_context.hpp"
#include "mock_backend.hpp"

#include <cstddef>
#include <gtest/gtest.h>
#include <memory>
#include <stdexcept>
#include <vector>

#define SKIP_WITHOUT_GPU()                                                                                             \
    do {                                                                                                               \
        if (!rfdetr::gpu::device_available()) {                                                                        \
            GTEST_SKIP() << "no CUDA device available";                                                                \
        }                                                                                                              \
    } while (false)

/// Mock backend that also publishes its outputs as device pointers, so the GPU
/// postprocessor can read exactly what the CPU postprocessor reads.
class MockDeviceBackend : public MockBackend {
  public:
    explicit MockDeviceBackend(rfdetr::gpu::StreamHandle stream) : stream_(stream) {}

    /// Stages the configured outputs into device memory. Call after set_outputs().
    void upload() {
        device_buffers_.clear();
        device_buffers_.reserve(output_data_.size());
        for (const auto &tensor : output_data_) {
            const size_t bytes = tensor.size() * sizeof(float);
            auto buffer = std::make_unique<rfdetr::gpu::DeviceBuffer>(bytes);
            rfdetr::gpu::copy_h2d(buffer->get(), tensor.data(), bytes, stream_);
            device_buffers_.push_back(std::move(buffer));
        }
        rfdetr::gpu::stream_synchronize(stream_);
    }

    [[nodiscard]] bool supports_device_io() const noexcept override { return true; }

    [[nodiscard]] const void *get_output_device_ptr(size_t output_index) const override {
        if (output_index >= device_buffers_.size()) {
            throw std::out_of_range("Device output index out of range");
        }
        return device_buffers_[output_index]->get();
    }

    [[nodiscard]] void *device_stream() const noexcept override { return stream_; }

    void synchronize_device() override { rfdetr::gpu::stream_synchronize(stream_); }

  private:
    rfdetr::gpu::StreamHandle stream_;
    std::vector<std::unique_ptr<rfdetr::gpu::DeviceBuffer>> device_buffers_;
};

#endif // USE_CUDA_POSTPROCESS || USE_DALI
