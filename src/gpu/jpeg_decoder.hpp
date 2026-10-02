#pragma once

#ifdef USE_CUDA_PREPROCESS

#include "gpu_context.hpp"

#include <cstdint>
#include <memory>
#include <optional>
#include <span>

namespace rfdetr::gpu {

/// Pixel dimensions of an encoded image, read from its header.
struct ImageSize {
    int width{0};
    int height{0};
};

/// Decodes JPEG bytes on the GPU with nvJPEG into interleaved BGR `uint8` — the
/// layout `preprocess_bgr_device` consumes, so the image and video-frame paths
/// share one kernel and one parity contract.
///
/// Owns one `nvjpegHandle_t` and one `nvjpegJpegState_t`. The decode state is
/// per-image scratch and is not safe to share: use one decoder per thread.
class JpegDecoder {
  public:
    JpegDecoder();
    ~JpegDecoder();

    JpegDecoder(const JpegDecoder &) = delete;
    JpegDecoder &operator=(const JpegDecoder &) = delete;
    JpegDecoder(JpegDecoder &&) = delete;
    JpegDecoder &operator=(JpegDecoder &&) = delete;

    /// Reads the header of `bytes`. Empty if they are not a JPEG this decoder
    /// can turn into BGR (not a JPEG at all, CMYK, unknown subsampling) — the
    /// caller then decodes on the CPU. Magic bytes decide, not file extensions.
    [[nodiscard]] std::optional<ImageSize> probe(std::span<const std::uint8_t> bytes) const;

    /// Decodes `bytes` into `dst` as `height * width * 3` tightly packed BGR
    /// bytes, growing `dst` if needed, with the GPU work enqueued on `stream`.
    /// The Huffman stage runs on the host before this returns; the rest is
    /// asynchronous, so `dst` is ready only for later work on the same stream.
    ///
    /// @throws std::runtime_error if nvJPEG rejects the bytes
    ImageSize decode(std::span<const std::uint8_t> bytes, DeviceBuffer &dst, StreamHandle stream);

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace rfdetr::gpu

#endif // USE_CUDA_PREPROCESS
