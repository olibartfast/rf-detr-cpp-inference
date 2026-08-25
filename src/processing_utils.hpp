#pragma once

#include "rfdetr_types.hpp"

#include <array>
#include <cstddef>
#include <optional>
#include <span>
#include <vector>

namespace rfdetr::processing {

/// Sigmoid activation: maps logit to probability [0, 1]
[[nodiscard]] float sigmoid(float x) noexcept;

/// Normalize CHW image data in-place: (pixel - mean) / std per channel
void normalize_image(std::span<float> data, size_t channel_size, std::span<const float, 3> means,
                     std::span<const float, 3> stds);

using ::BoundingBox;

/// Convert center-format (cx, cy, w, h) to corner-format (x_min, y_min, x_max, y_max)
[[nodiscard]] BoundingBox cxcywh_to_xyxy(float cx, float cy, float w, float h) noexcept;

/// Scale a bounding box by independent width/height factors
[[nodiscard]] BoundingBox scale_box(const BoundingBox &box, float scale_w, float scale_h) noexcept;

/// Clamp a bounding box to image bounds [0, max_w] x [0, max_h]
[[nodiscard]] BoundingBox clamp_box(const BoundingBox &box, float max_w, float max_h) noexcept;

// --- Class layout and top-k selection ------------------------------------
//
// Mirrors of rfdetr's `export/_class_layout.py` and `export/_topk.py` (added in
// 1.9.4 and 1.9.3), which are themselves torch-free copies of
// `PostProcess._exclude_background_class` / `PostProcess._select_topk`. Keeping
// the decode here in one place lets the detection, segmentation and keypoint
// paths — and the CUDA kernels — share one selection rule.

/// Resolve which exported logit slot holds background.
///
/// `background_class_id` follows upstream's NumPy indexing convention: a
/// negative value counts from the end (`-1` = final slot) and `std::nullopt`
/// keeps every slot. Returns the resolved slot in `[0, num_classes)`, or -1 when
/// no slot is excluded.
///
/// Throws `std::invalid_argument` when the slot does not index one of the
/// exported classes, matching upstream's `ValueError`.
[[nodiscard]] int resolve_background_slot(std::optional<int> background_class_id, int num_classes);

/// Number of foreground slots left once `background_slot` (-1 for none) is excluded.
[[nodiscard]] int foreground_class_count(int num_classes, int background_slot) noexcept;

/// Exported logit slot backing foreground column `column`.
///
/// Excluding the background column leaves the remaining slots in their original
/// order, so foreground column `c` is the `c`-th label in the label file — that
/// positional mapping is what turns a decoded column into a class id.
[[nodiscard]] int slot_for_foreground_column(int column, int background_slot) noexcept;

/// Sigmoid a `(num_queries, num_classes)` logit grid into the
/// `(num_queries, foreground_class_count(...))` score grid that ranking consumes,
/// dropping the background column. Excluding it *before* ranking matters: a
/// background column left in would consume slots of the top-k cap.
void build_foreground_scores(std::span<const float> logits, int num_queries, int num_classes, int background_slot,
                             std::vector<float> &scores);

/// Rank a flattened `(Q, C_fg)` score grid and return the top `num_select` indices.
///
/// Mirror of `_select_topk_multiclass`: RF-DETR scores classes with independent
/// sigmoids rather than a softmax, so one query can legitimately clear the
/// threshold on several classes at once. Ranking the flattened query/class pairs
/// keeps all of them, where a per-query argmax would silently drop every class
/// but the strongest.
///
/// Ordering is the deterministic lexicographic rule 1.9.3 introduced on both
/// sides (`torch.argsort(..., stable=True)` upstream): descending score, then
/// ascending flattened index. NaN scores rank first, as they do in torch's
/// descending sort, so the caller's `score > threshold` test drops them instead
/// of letting a lower finite score take their place under the cap.
[[nodiscard]] std::vector<size_t> select_topk_multiclass(std::span<const float> scores, size_t num_select);

} // namespace rfdetr::processing
