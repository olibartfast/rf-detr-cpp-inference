#pragma once

#include <array>
#include <cstddef>
#include <opencv2/core.hpp>
#include <span>

namespace rfdetr::processing {

/// Sigmoid activation: maps logit to probability [0, 1]
[[nodiscard]] float sigmoid(float x) noexcept;

/// Normalize CHW image data in-place: (pixel - mean) / std per channel
void normalize_image(std::span<float> data, size_t channel_size, std::span<const float, 3> means,
                     std::span<const float, 3> stds);

/// Axis-aligned bounding box in xyxy format
struct BoundingBox {
    float x_min;
    float y_min;
    float x_max;
    float y_max;
};

/// Convert center-format (cx, cy, w, h) to corner-format (x_min, y_min, x_max, y_max)
[[nodiscard]] BoundingBox cxcywh_to_xyxy(float cx, float cy, float w, float h) noexcept;

/// Scale a bounding box by independent width/height factors
[[nodiscard]] BoundingBox scale_box(const BoundingBox &box, float scale_w, float scale_h) noexcept;

/// Deterministic color for a class ID (golden-angle hue distribution)
[[nodiscard]] cv::Scalar get_color_for_class(int class_id) noexcept;

/// Preprocess a BGR cv::Mat frame into a pre-allocated CHW float tensor.
/// output must have size >= 3 * resolution * resolution.
/// Performs: resize → BGR2RGB → float32 [0,1] → CHW split → normalize.
void preprocess_frame(const cv::Mat &bgr_frame, std::span<float> output, int resolution,
                      std::span<const float, 3> means, std::span<const float, 3> stds);

} // namespace rfdetr::processing
