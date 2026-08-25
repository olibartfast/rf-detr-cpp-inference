#include "processing_utils.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>

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

int resolve_background_slot(std::optional<int> background_class_id, int num_classes) {
    if (num_classes <= 0) {
        throw std::invalid_argument("num_classes must be positive, got " + std::to_string(num_classes));
    }
    if (!background_class_id.has_value()) {
        return -1;
    }
    const int slot = *background_class_id;
    if (slot < -num_classes || slot >= num_classes) {
        throw std::invalid_argument("background class id must index one of " + std::to_string(num_classes) +
                                    " exported class slots, got " + std::to_string(slot));
    }
    return slot < 0 ? slot + num_classes : slot;
}

int foreground_class_count(int num_classes, int background_slot) noexcept {
    return background_slot < 0 ? num_classes : num_classes - 1;
}

int slot_for_foreground_column(int column, int background_slot) noexcept {
    return (background_slot >= 0 && column >= background_slot) ? column + 1 : column;
}

void build_foreground_scores(std::span<const float> logits, int num_queries, int num_classes, int background_slot,
                             std::vector<float> &scores) {
    const int num_foreground = foreground_class_count(num_classes, background_slot);
    scores.clear();
    if (num_queries <= 0 || num_foreground <= 0) {
        return;
    }
    scores.reserve(static_cast<size_t>(num_queries) * static_cast<size_t>(num_foreground));
    for (int q = 0; q < num_queries; ++q) {
        const size_t row = static_cast<size_t>(q) * static_cast<size_t>(num_classes);
        for (int column = 0; column < num_foreground; ++column) {
            const size_t slot = static_cast<size_t>(slot_for_foreground_column(column, background_slot));
            scores.push_back(sigmoid(logits[row + slot]));
        }
    }
}

std::vector<size_t> select_topk_multiclass(std::span<const float> scores, size_t num_select) {
    const size_t count = std::min(num_select, scores.size());
    std::vector<size_t> order(scores.size());
    if (count == 0) {
        order.clear();
        return order;
    }
    std::iota(order.begin(), order.end(), size_t{0});

    // Rank NaN ahead of every finite score, the way torch's descending sort does.
    // Substituting the key keeps the comparator a strict weak ordering, which a
    // raw NaN comparison would not be.
    const auto rank_key = [scores](size_t index) {
        const float score = scores[index];
        return std::isnan(score) ? std::numeric_limits<float>::infinity() : score;
    };
    std::partial_sort(order.begin(), order.begin() + static_cast<ptrdiff_t>(count), order.end(),
                      [&rank_key](size_t lhs, size_t rhs) {
                          const float lhs_key = rank_key(lhs);
                          const float rhs_key = rank_key(rhs);
                          // Descending score, then ascending flattened index.
                          return lhs_key != rhs_key ? lhs_key > rhs_key : lhs < rhs;
                      });
    order.resize(count);
    return order;
}

} // namespace rfdetr::processing
