#pragma once

#ifdef USE_CUDA_POSTPROCESS

#include "gpu_context.hpp"
#include "rfdetr_types.hpp"

#include <cstdint>
#include <memory>
#include <vector>

namespace rfdetr::gpu {

/// Everything the segmentation postprocess kernels need that is not a tensor.
/// Shapes come from the model's output tensors, never from constants, so one
/// build works across resolutions, query counts and mask sizes.
struct SegPostprocessParams {
    int num_queries{0};         ///< dets/labels dimension 1
    int num_classes{0};         ///< labels dimension 2, including the background logit
    int dets_stride{4};         ///< dets dimension 2 (cxcywh occupies the first 4)
    int mask_h{0};              ///< masks dimension 2
    int mask_w{0};              ///< masks dimension 3
    int resolution{0};          ///< model input resolution; boxes are normalised to it
    int orig_w{0};              ///< output frame width, and mask output width
    int orig_h{0};              ///< output frame height, and mask output height
    float scale_w{1.0F};        ///< orig_w / resolution
    float scale_h{1.0F};        ///< orig_h / resolution
    float threshold{0.5F};      ///< score threshold, applied after ranking
    float mask_threshold{0.0F}; ///< raw-logit threshold (NOT sigmoid-then-0.5)
    int max_detections{300};    ///< cap on the number of ranked candidates examined
    int num_labels{0};          ///< label-file size; class ids at or above it are dropped
};

/// Packed segmentation results, host side.
///
/// Masks are concatenated full-frame binary images: detection `i` occupies
/// `[mask_offsets[i], mask_offsets[i + 1])` of `mask_data`, row-major
/// `orig_h * orig_w`, each byte 0 or 255. `mask_offsets` always has
/// `count + 1` entries.
struct SegPostprocessResult {
    int count{0};
    std::vector<float> scores;
    std::vector<int32_t> class_ids;
    std::vector<BoundingBox> boxes;
    std::vector<int64_t> mask_offsets;
    std::vector<uint8_t> mask_data;
};

/// GPU segmentation postprocessor: score decode, ranking, box decode, and
/// per-instance mask resize + threshold, all on one CUDA stream.
///
/// Device scratch is owned by the instance and reused across frames, so a video
/// run allocates once. Construct it once per stream and call run() per frame.
class SegPostprocessor {
  public:
    SegPostprocessor(const SegPostprocessParams &params, StreamHandle stream);
    ~SegPostprocessor();

    SegPostprocessor(const SegPostprocessor &) = delete;
    SegPostprocessor &operator=(const SegPostprocessor &) = delete;
    SegPostprocessor(SegPostprocessor &&) = delete;
    SegPostprocessor &operator=(SegPostprocessor &&) = delete;

    /// Updates the per-frame geometry. Cheap; call when the source size changes.
    void set_frame_geometry(int orig_w, int orig_h, float scale_w, float scale_h);

    [[nodiscard]] const SegPostprocessParams &params() const noexcept;

    /// Reads the three model outputs from device memory and fills `out`.
    ///
    /// The pointers must be valid on the stream this object was constructed with;
    /// they are typically the TensorRT output bindings. Blocks until the results
    /// have been copied back to the host.
    void run(const void *dets_device, const void *labels_device, const void *masks_device, SegPostprocessResult &out);

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace rfdetr::gpu

#endif // USE_CUDA_POSTPROCESS
