#pragma once

#include <array>
#include <cstddef>
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

/// Clamp a bounding box to image bounds [0, max_w] x [0, max_h]
[[nodiscard]] BoundingBox clamp_box(const BoundingBox &box, float max_w, float max_h) noexcept;

} // namespace rfdetr::processing
