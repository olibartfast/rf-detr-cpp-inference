#include "rfdetr_inference.hpp"

#include "processing_utils.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>

RFDETRInference::RFDETRInference(const std::filesystem::path &model_path, const std::filesystem::path &label_file_path,
                                 const Config &config)
    : backend_(create_backend()), config_(config), input_shape_({1, 3, config_.resolution, config_.resolution}) {

    std::cout << "Using backend: " << backend_->get_backend_name() << std::endl;

    // Initialize backend
    input_shape_ = backend_->initialize(model_path, input_shape_);

    // Update resolution if auto-detected
    if (config_.resolution == 0 && input_shape_.size() == 4) {
        config_.resolution = static_cast<int>(input_shape_[2]);
        std::cout << "Auto-detected model input resolution: " << config_.resolution << "x" << config_.resolution
                  << std::endl;
    }

    // Validate number of outputs
    const size_t num_outputs = backend_->get_output_count();
    const size_t num_expected = config_.model_type == ModelType::SEGMENTATION ? 3
                                : config_.model_type == ModelType::KEYPOINT   ? 3
                                                                              : 2;

    if (num_outputs < num_expected) {
        std::string type_str = config_.model_type == ModelType::SEGMENTATION ? "Segmentation"
                               : config_.model_type == ModelType::KEYPOINT   ? "Keypoint"
                                                                             : "Detection";
        throw std::runtime_error(type_str + " model requires " + std::to_string(num_expected) +
                                 " outputs, but model has only " + std::to_string(num_outputs));
    }

    // Load COCO labels
    load_coco_labels(label_file_path);
}

RFDETRInference::RFDETRInference(std::unique_ptr<InferenceBackend> backend,
                                 const std::filesystem::path &label_file_path, const Config &config)
    : backend_(std::move(backend)), config_(config), input_shape_({1, 3, config_.resolution, config_.resolution}) {
    load_coco_labels(label_file_path);
}

void RFDETRInference::load_coco_labels(const std::filesystem::path &label_file_path) {
    if (!std::filesystem::exists(label_file_path)) {
        throw std::runtime_error("Label file does not exist: " + label_file_path.string());
    }

    std::ifstream file(label_file_path);
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty()) {
            coco_labels_.push_back(line);
        }
    }
    if (coco_labels_.empty()) {
        throw std::runtime_error("No labels found in file: " + label_file_path.string());
    }
}

std::vector<float> RFDETRInference::preprocess_image(const std::filesystem::path &image_path, int &orig_h,
                                                     int &orig_w) {
    if (!std::filesystem::exists(image_path)) {
        throw std::runtime_error("Image file does not exist: " + image_path.string());
    }

    auto image = rfdetr::media::load_image(image_path);
    return preprocess_image(image, orig_h, orig_w);
}

std::vector<float> RFDETRInference::preprocess_image(const rfdetr::media::Image &bgr_image, int &orig_h, int &orig_w) {
    if (bgr_image.empty()) {
        throw std::runtime_error("Input image is empty");
    }
    orig_h = bgr_image.height;
    orig_w = bgr_image.width;

    const auto res = static_cast<size_t>(config_.resolution);
    std::vector<float> input_tensor_values(3 * res * res);
    rfdetr::media::preprocess_bgr_image(bgr_image, input_tensor_values, config_.resolution, config_.means,
                                        config_.stds);
    return input_tensor_values;
}

void RFDETRInference::run_inference(std::span<const float> input_data) {
    // Run inference through backend
    backend_->run_inference(input_data, input_shape_);

    // Cache output data and shapes for postprocessing
    const size_t num_outputs = backend_->get_output_count();
    output_data_cache_.clear();
    output_shapes_cache_.clear();

    for (size_t i = 0; i < num_outputs; ++i) {
        auto shape = backend_->get_output_shape(i);
        const size_t size = std::accumulate(shape.begin(), shape.end(), size_t{1},
                                            [](size_t acc, int64_t dim) { return acc * static_cast<size_t>(dim); });

        std::vector<float> data(size);
        backend_->get_output_data(i, data.data(), size);

        output_data_cache_.push_back(std::move(data));
        output_shapes_cache_.push_back(std::move(shape));
    }
}

void RFDETRInference::postprocess_outputs(float scale_w, float scale_h, std::vector<float> &scores,
                                          std::vector<int> &class_ids, std::vector<std::vector<float>> &boxes) {
    if (output_data_cache_.size() < 2) {
        throw std::runtime_error("Expected at least 2 output tensors, got " +
                                 std::to_string(output_data_cache_.size()));
    }

    const auto &dets_data = output_data_cache_[0];
    const auto &dets_shape = output_shapes_cache_[0];

    const auto &labels_data = output_data_cache_[1];
    const auto &labels_shape = output_shapes_cache_[1];

    const auto num_detections = static_cast<size_t>(dets_shape[1]);
    const auto num_classes = static_cast<size_t>(labels_shape[2]);
    const auto res = static_cast<float>(config_.resolution);
    const auto max_w = scale_w * res;
    const auto max_h = scale_h * res;

    for (size_t i = 0; i < num_detections; ++i) {
        const size_t det_offset = i * static_cast<size_t>(dets_shape[2]);
        const size_t label_offset = i * num_classes;

        float max_score = -1.0f;
        int max_class_idx = -1;
        for (size_t j = 0; j < num_classes; ++j) {
            const float logit = labels_data[label_offset + j];
            const float score = rfdetr::processing::sigmoid(logit);
            if (score > max_score) {
                max_score = score;
                max_class_idx = static_cast<int>(j);
            }
        }

        max_class_idx -= 1; // Fix the +1 offset

        if (max_score > config_.threshold && max_class_idx >= 0 &&
            static_cast<size_t>(max_class_idx) < coco_labels_.size()) {
            const float cx = dets_data[det_offset + 0] * res;
            const float cy = dets_data[det_offset + 1] * res;
            const float w = dets_data[det_offset + 2] * res;
            const float h = dets_data[det_offset + 3] * res;

            auto xyxy = rfdetr::processing::cxcywh_to_xyxy(cx, cy, w, h);
            auto scaled = rfdetr::processing::scale_box(xyxy, scale_w, scale_h);
            auto clamped = rfdetr::processing::clamp_box(scaled, max_w, max_h);

            std::vector<float> box = {clamped.x_min, clamped.y_min, clamped.x_max, clamped.y_max};

            scores.push_back(max_score);
            class_ids.push_back(max_class_idx);
            boxes.push_back(std::move(box));
        }
    }
}

void RFDETRInference::postprocess_segmentation_outputs(float scale_w, float scale_h, int orig_h, int orig_w,
                                                       std::vector<float> &scores, std::vector<int> &class_ids,
                                                       std::vector<std::vector<float>> &boxes,
                                                       std::vector<rfdetr::media::Mask> &masks) {
    if (output_data_cache_.size() != 3) {
        throw std::runtime_error("Expected 3 output tensors for segmentation, got " +
                                 std::to_string(output_data_cache_.size()));
    }

    // Get bounding boxes data
    const auto &dets_data = output_data_cache_[0];
    const auto &dets_shape = output_shapes_cache_[0];

    // Get labels data
    const auto &labels_data = output_data_cache_[1];
    const auto &labels_shape = output_shapes_cache_[1];

    // Get masks data
    const auto &masks_data = output_data_cache_[2];
    const auto &masks_shape = output_shapes_cache_[2];

    const auto num_detections = static_cast<size_t>(dets_shape[1]);
    const auto num_classes = static_cast<size_t>(labels_shape[2]);
    const auto mask_h = static_cast<size_t>(masks_shape[2]);
    const auto mask_w = static_cast<size_t>(masks_shape[3]);

    // Compute scores and apply sigmoid
    std::vector<float> all_scores;
    std::vector<size_t> all_indices;

    for (size_t i = 0; i < num_detections; ++i) {
        for (size_t j = 0; j < num_classes; ++j) {
            const size_t label_offset = i * num_classes;
            const float logit = labels_data[label_offset + j];
            const float score = rfdetr::processing::sigmoid(logit);
            all_scores.push_back(score);
            all_indices.push_back(i * num_classes + j);
        }
    }

    // Top-k selection
    const size_t num_select = std::min(static_cast<size_t>(config_.max_detections), all_scores.size());
    std::vector<size_t> topk_indices(all_scores.size());
    std::iota(topk_indices.begin(), topk_indices.end(), 0);
    std::partial_sort(topk_indices.begin(), topk_indices.begin() + static_cast<ptrdiff_t>(num_select),
                      topk_indices.end(),
                      [&all_scores](size_t i1, size_t i2) { return all_scores[i1] > all_scores[i2]; });

    // Process top-k detections
    for (size_t k = 0; k < num_select; ++k) {
        const size_t idx = topk_indices[k];
        const float score = all_scores[idx];

        if (score <= config_.threshold) {
            continue;
        }

        const size_t detection_idx = all_indices[idx] / num_classes;
        const size_t class_idx = all_indices[idx] % num_classes;
        const int class_id = static_cast<int>(class_idx) - 1; // Fix the +1 offset

        if (class_id < 0 || static_cast<size_t>(class_id) >= coco_labels_.size()) {
            continue;
        }

        // Get bounding box (in cxcywh format, normalized)
        const auto res = static_cast<float>(config_.resolution);
        const size_t det_offset = detection_idx * static_cast<size_t>(dets_shape[2]);
        const float cx = dets_data[det_offset + 0] * res;
        const float cy = dets_data[det_offset + 1] * res;
        const float w = dets_data[det_offset + 2] * res;
        const float h = dets_data[det_offset + 3] * res;

        auto xyxy = rfdetr::processing::cxcywh_to_xyxy(cx, cy, w, h);
        auto scaled = rfdetr::processing::scale_box(xyxy, scale_w, scale_h);
        auto clamped = rfdetr::processing::clamp_box(scaled, static_cast<float>(orig_w), static_cast<float>(orig_h));

        std::vector<float> box = {clamped.x_min, clamped.y_min, clamped.x_max, clamped.y_max};

        const size_t mask_offset = detection_idx * mask_h * mask_w;
        auto binary_mask = rfdetr::media::resize_threshold_mask(
            std::span<const float>(masks_data.data() + mask_offset, mask_h * mask_w), static_cast<int>(mask_w),
            static_cast<int>(mask_h), orig_w, orig_h, config_.mask_threshold);

        scores.push_back(score);
        class_ids.push_back(class_id);
        boxes.push_back(std::move(box));
        masks.push_back(binary_mask);
    }
}

void RFDETRInference::draw_detections(rfdetr::media::Image &image, std::span<const std::vector<float>> boxes,
                                      std::span<const int> class_ids, std::span<const float> scores) {
    (void)scores;
    if (boxes.size() != class_ids.size() || boxes.size() != scores.size()) {
        throw std::runtime_error("Mismatch in sizes of boxes, class_ids, and scores");
    }
    rfdetr::media::draw_detections(image, boxes, class_ids);
}

void RFDETRInference::draw_segmentation_masks(rfdetr::media::Image &image, std::span<const std::vector<float>> boxes,
                                              std::span<const int> class_ids, std::span<const float> scores,
                                              std::span<const rfdetr::media::Mask> masks) {
    (void)scores;
    if (boxes.size() != class_ids.size() || boxes.size() != scores.size() || boxes.size() != masks.size()) {
        throw std::runtime_error("Mismatch in sizes of boxes, class_ids, scores, and masks");
    }
    rfdetr::media::draw_segmentation_masks(image, boxes, class_ids, masks);
}

std::string RFDETRInference::get_label_name(int class_id) const {
    const auto idx = static_cast<size_t>(class_id);
    if (idx < coco_labels_.size()) {
        return coco_labels_[idx];
    }
    return "unknown";
}

void RFDETRInference::postprocess_keypoint_outputs(float scale_w, float scale_h, int orig_h, int orig_w,
                                                   std::vector<float> &scores, std::vector<int> &class_ids,
                                                   std::vector<std::vector<float>> &boxes,
                                                   std::vector<std::vector<KeypointResult>> &keypoints) {
    if (output_data_cache_.size() < 3) {
        throw std::runtime_error("Expected at least 3 output tensors for keypoint, got " +
                                 std::to_string(output_data_cache_.size()));
    }

    const auto &dets_data = output_data_cache_[0];
    const auto &dets_shape = output_shapes_cache_[0];

    const auto &labels_data = output_data_cache_[1];
    const auto &labels_shape = output_shapes_cache_[1];

    const auto &kp_data = output_data_cache_[2];
    const auto &kp_shape = output_shapes_cache_[2];

    const auto num_queries = static_cast<size_t>(dets_shape[1]);
    const auto num_classes = static_cast<size_t>(labels_shape[2]);
    constexpr size_t kp_channels = 8;
    const bool has_kp_channel_dim = kp_shape.size() >= 4;
    const auto num_keypoints =
        has_kp_channel_dim ? static_cast<size_t>(kp_shape[2]) : static_cast<size_t>(kp_shape[2]) / kp_channels;
    const size_t query_stride = has_kp_channel_dim ? static_cast<size_t>(kp_shape[2]) * static_cast<size_t>(kp_shape[3])
                                                   : static_cast<size_t>(kp_shape[2]);

    if (has_kp_channel_dim && static_cast<size_t>(kp_shape[3]) != kp_channels) {
        throw std::runtime_error("Keypoint tensor last dimension (" + std::to_string(kp_shape[3]) + ") must be " +
                                 std::to_string(kp_channels));
    }
    if (!has_kp_channel_dim && static_cast<size_t>(kp_shape[2]) % kp_channels != 0) {
        throw std::runtime_error("Flattened keypoint tensor channels (" + std::to_string(kp_shape[2]) +
                                 ") must be divisible by " + std::to_string(kp_channels));
    }

    // Build keypoint class mapping: which classes have keypoints and at what offset.
    // The ONNX keypoint tensor is padded: each keypoint class gets K_max slots,
    // even if the class has 0 active keypoints.  The stride per class is computed
    // from the tensor shape, not from the per-class counts.
    const auto &kp_counts = config_.keypoint_counts;
    const size_t num_kp_classes = kp_counts.empty() ? 0 : kp_counts.size();
    const size_t kp_stride =
        (num_kp_classes > 0) ? (num_keypoints / num_kp_classes) * kp_channels : num_keypoints * kp_channels;
    // Map: keypoint_class_index -> (num_kps, byte_offset_in_tensor)
    std::vector<std::pair<size_t, size_t>> kp_map;
    if (!kp_counts.empty()) {
        kp_map.reserve(kp_counts.size());
        for (size_t c = 0; c < kp_counts.size(); ++c) {
            const auto count = static_cast<size_t>(kp_counts[c] >= 0 ? kp_counts[c] : 0);
            kp_map.emplace_back(count, c * kp_stride);
        }
    }
    // Validate keypoint tensor shape: total channels must be divisible by num_kp_classes
    if (num_kp_classes > 0 && num_keypoints % num_kp_classes != 0) {
        throw std::runtime_error("Keypoint tensor channels (" + std::to_string(num_keypoints) +
                                 ") not divisible by number of keypoint classes (" + std::to_string(num_kp_classes) +
                                 ")");
    }
    // Find the keypoint class with the most active keypoints.
    // For single-class models (e.g. COCO person), all detections use this class.
    size_t default_kp_class = 0;
    size_t max_kps = 0;
    for (size_t c = 0; c < kp_map.size(); ++c) {
        if (kp_map[c].first > max_kps) {
            max_kps = kp_map[c].first;
            default_kp_class = c;
        }
    }

    const float res = static_cast<float>(config_.resolution);

    for (size_t q = 0; q < num_queries; ++q) {
        const size_t det_offset = q * static_cast<size_t>(dets_shape[2]);
        const size_t label_offset = q * num_classes;

        // RF-DETR keypoint labels use the same offset as detection: logit 0 is background,
        // logit 1 is the first real class (COCO person for the preview model).
        float best_score = -1.0f;
        int best_class_idx = -1;
        for (size_t j = 0; j < num_classes; ++j) {
            const float logit = labels_data[label_offset + j];
            const float score = rfdetr::processing::sigmoid(logit);
            if (score > best_score) {
                best_score = score;
                best_class_idx = static_cast<int>(j);
            }
        }

        const int class_id = best_class_idx - 1;
        if (best_score <= config_.threshold || class_id < 0 || static_cast<size_t>(class_id) >= coco_labels_.size()) {
            continue;
        }

        // Decode bbox
        const float cx = dets_data[det_offset + 0] * res;
        const float cy = dets_data[det_offset + 1] * res;
        const float w = dets_data[det_offset + 2] * res;
        const float h = dets_data[det_offset + 3] * res;

        auto xyxy = rfdetr::processing::cxcywh_to_xyxy(cx, cy, w, h);
        auto scaled = rfdetr::processing::scale_box(xyxy, scale_w, scale_h);
        auto clamped = rfdetr::processing::clamp_box(scaled, static_cast<float>(orig_w), static_cast<float>(orig_h));

        std::vector<float> box = {clamped.x_min, clamped.y_min, clamped.x_max, clamped.y_max};

        std::vector<KeypointResult> kp_results;
        size_t selected_kp_class = default_kp_class;
        if (best_class_idx >= 0 && static_cast<size_t>(best_class_idx) < kp_map.size() &&
            kp_map[static_cast<size_t>(best_class_idx)].first > 0) {
            selected_kp_class = static_cast<size_t>(best_class_idx);
        }

        size_t num_kps = 0;
        size_t kp_offset = 0;
        if (selected_kp_class < kp_map.size()) {
            num_kps = kp_map[selected_kp_class].first;
            kp_offset = kp_map[selected_kp_class].second;
        }

        kp_results.reserve(num_kps != 0 ? num_kps : 0);
        const size_t base = q * query_stride;

        for (size_t k = 0; k < num_kps; ++k) {
            const size_t ch_off = kp_offset + k * kp_channels;

            // Clip to available data
            if (base + ch_off + 7 >= kp_data.size()) {
                break;
            }

            KeypointResult kpr{};

            // RF-DETR keypoints are normalized image-relative coordinates.
            kpr.x = kp_data[base + ch_off + 0] * static_cast<float>(orig_w);
            kpr.y = kp_data[base + ch_off + 1] * static_cast<float>(orig_h);

            // Step e-f: Sigmoid findability and visibility
            kpr.findability = rfdetr::processing::sigmoid(kp_data[base + ch_off + 2]);
            kpr.visibility = rfdetr::processing::sigmoid(kp_data[base + ch_off + 3]);

            // Step g: Precision Cholesky -> pixel covariance
            const float log_l11 = kp_data[base + ch_off + 4];
            const float l21 = kp_data[base + ch_off + 5];
            const float log_l22 = kp_data[base + ch_off + 6];
            // Channel 7 = class_boost, skip (already aggregated into labels)

            constexpr float eps = 1e-6f;
            const float l11 = std::exp(log_l11) + eps;
            const float l22 = std::exp(log_l22) + eps;

            // L = [[l11, 0], [l21, l22]]
            // precision = L @ L^T
            const float p00 = l11 * l11;
            const float p01 = l11 * l21;
            const float p10 = l21 * l11;
            const float p11 = l21 * l21 + l22 * l22;

            // Invert 2x2 precision to get covariance
            const float det = p00 * p11 - p01 * p10;
            if (std::abs(det) > eps) {
                float inv_p00 = p11 / det;
                float inv_p01 = -p01 / det;
                float inv_p11 = p00 / det;

                // Pixel-space covariance: diag(width, height) * cov * diag(width, height).
                const float width = static_cast<float>(orig_w);
                const float height = static_cast<float>(orig_h);
                kpr.cov[0] = inv_p00 * width * width;
                kpr.cov[1] = inv_p01 * width * height;
                kpr.cov[2] = inv_p01 * width * height; // symmetric
                kpr.cov[3] = inv_p11 * height * height;
            } else {
                kpr.cov[0] = kpr.cov[1] = kpr.cov[2] = kpr.cov[3] = 0.0f;
            }

            kp_results.push_back(kpr);
        }

        // Step h: Uncertainty-weighted score fusion
        float final_score = best_score;
        if (config_.keypoint_uncertainty_alpha > 0.0f && !kp_results.empty()) {
            const float trace_sum =
                std::accumulate(kp_results.begin(), kp_results.end(), 0.0f,
                                [](float sum, const KeypointResult &kpr) { return sum + kpr.cov[0] + kpr.cov[3]; });
            const float avg_trace = trace_sum / static_cast<float>(kp_results.size());
            if (avg_trace > 0.0f) {
                const float penalty = config_.keypoint_uncertainty_alpha * std::log(avg_trace);
                final_score *= std::exp(-penalty);
            }
        }

        scores.push_back(final_score);
        class_ids.push_back(class_id);
        boxes.push_back(std::move(box));
        keypoints.push_back(std::move(kp_results));
    }
}

void RFDETRInference::draw_keypoints(rfdetr::media::Image &image, std::span<const std::vector<float>> boxes,
                                     std::span<const int> class_ids, std::span<const float> scores,
                                     std::span<const std::vector<KeypointResult>> keypoints) {
    (void)scores;
    if (boxes.size() != class_ids.size() || boxes.size() != scores.size() || boxes.size() != keypoints.size()) {
        throw std::runtime_error("Mismatch in sizes of boxes, class_ids, scores, and keypoints");
    }
    rfdetr::media::draw_keypoints(image, boxes, class_ids, keypoints, config_.skeleton, config_.keypoint_color);
}

std::optional<std::filesystem::path> RFDETRInference::save_output_image(const rfdetr::media::Image &image,
                                                                        const std::filesystem::path &output_path) {
    if (rfdetr::media::save_image(image, output_path)) {
        return output_path;
    }
    return std::nullopt;
}
