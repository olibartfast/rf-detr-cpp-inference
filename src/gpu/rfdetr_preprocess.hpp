#pragma once

#ifdef USE_CUDA_PREPROCESS

#include "gpu_context.hpp"

#include <cstdint>
#include <span>

namespace rfdetr::gpu {

/// Preprocesses an interleaved BGR `uint8` image already resident on the device
/// into the model's NCHW `float` input: plain bilinear stretch to
/// `resolution x resolution` (no letterbox, no antialias), BGR -> RGB, `/255`,
/// then `(v - mean) / std` per channel.
///
/// Mirrors `rfdetr::media::preprocess_bgr_image` and
/// `rfdetr::processing::normalize_image` operation for operation, so the device
/// tensor matches the CPU one to float rounding. Change one, change both.
///
/// Enqueued on `stream` and not synchronised: `dst_device` is normally the
/// TensorRT input binding, consumed by an enqueue on the same stream.
///
/// @param bgr_device  `height * width * 3` bytes, row-major, tightly packed
/// @param dst_device  at least `3 * resolution * resolution` floats
void preprocess_bgr_device(const std::uint8_t *bgr_device, int height, int width, void *dst_device, int resolution,
                           std::span<const float, 3> means, std::span<const float, 3> stds, StreamHandle stream);

} // namespace rfdetr::gpu

#endif // USE_CUDA_PREPROCESS
