#include "processing_utils.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace rfdetr::processing {

float sigmoid(float x) noexcept { return 1.0f / (1.0f + std::exp(-x)); }

void normalize_image(std::span<float> data, size_t channel_size, std::span<const float, 3> means,
                     std::span<const float, 3> stds) {
    for (size_t c = 0; c < 3; ++c) {
        const float mean = means[c];
        const float std = stds[c];
        for (size_t i = 0; i < channel_size; ++i) {
            data[c * channel_size + i] = (data[c * channel_size + i] - mean) / std;
        }
    }
}

BoundingBox cxcywh_to_xyxy(float cx, float cy, float w, float h) noexcept {
    return {cx - w / 2.0f, cy - h / 2.0f, cx + w / 2.0f, cy + h / 2.0f};
}

BoundingBox scale_box(const BoundingBox &box, float scale_w, float scale_h) noexcept {
    return {box.x_min * scale_w, box.y_min * scale_h, box.x_max * scale_w, box.y_max * scale_h};
}

BoundingBox clamp_box(const BoundingBox &box, float max_w, float max_h) noexcept {
    return {std::clamp(box.x_min, 0.0f, max_w), std::clamp(box.y_min, 0.0f, max_h), std::clamp(box.x_max, 0.0f, max_w),
            std::clamp(box.y_max, 0.0f, max_h)};
}

} // namespace rfdetr::processing
