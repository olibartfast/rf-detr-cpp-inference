#pragma once

#ifdef USE_DALI

#include "gpu_context.hpp"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <vector>

namespace rfdetr::gpu {

/// Hosts a serialised DALI pipeline in-process through the DALI C API and writes
/// its output straight into a caller-owned device buffer — normally the
/// TensorRT input binding, which removes the host-to-device copy of the
/// preprocessed float tensor entirely.
///
/// Pipelines are produced by deploy/dali/generate_preprocess_pipeline.py; see
/// scripts/generate_dali_pipelines.sh. Two variants exist because the two input
/// sources differ in what the GPU has to do:
///
///   EncodedImage — feeds compressed bytes and decodes on the GPU (nvJPEG).
///   BgrFrame     — feeds an already-decoded BGR frame from the video reader.
class DaliPreprocessor {
  public:
    enum class Source { EncodedImage, BgrFrame };

    /// @param serialized_pipeline Path to a .dali file
    /// @param source Which external-source contract the pipeline declares
    /// @param device_id CUDA device the pipeline was serialised for
    DaliPreprocessor(const std::filesystem::path &serialized_pipeline, Source source, int device_id = 0);
    ~DaliPreprocessor();

    DaliPreprocessor(const DaliPreprocessor &) = delete;
    DaliPreprocessor &operator=(const DaliPreprocessor &) = delete;
    DaliPreprocessor(DaliPreprocessor &&) = delete;
    DaliPreprocessor &operator=(DaliPreprocessor &&) = delete;

    /// Decodes and preprocesses `bytes`, writing an NCHW float tensor of
    /// `dst_bytes` into `dst_device`. Blocks until the write has completed.
    void process_encoded(std::span<const std::uint8_t> bytes, void *dst_device, std::size_t dst_bytes,
                         StreamHandle stream);

    /// Preprocesses an interleaved BGR frame already resident on the device.
    void process_frame(const void *bgr_device, int height, int width, void *dst_device, std::size_t dst_bytes,
                       StreamHandle stream);

    /// Shape of output 0 from the most recent call, e.g. {3, 560, 560}.
    [[nodiscard]] const std::vector<std::int64_t> &last_output_shape() const noexcept;

    [[nodiscard]] Source source() const noexcept;

  private:
    /// Runs the pipeline and copies output 0 into `dst_device`.
    void run_and_copy(void *dst_device, std::size_t dst_bytes, StreamHandle stream);

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/// Name of the external source each variant expects, matching the generator.
inline constexpr const char *kDaliEncodedInputName = "IMAGE";
inline constexpr const char *kDaliFrameInputName = "FRAME";

} // namespace rfdetr::gpu

#endif // USE_DALI
