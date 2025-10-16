#include "rfdetr_inference.hpp"
#include <fstream>
#include <stdexcept>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <iostream>

RFDETRInference::RFDETRInference(
    const std::filesystem::path& model_path,
    const std::filesystem::path& label_file_path,
    const Config& config
)
    : config_(config),
      input_shape_({1, 3, config_.resolution, config_.resolution}) {
    
    // Create inference backend (determined at compile time)
    backend_ = create_backend();
    std::cout << "Using backend: " << backend_->get_backend_name() << std::endl;

    // Initialize backend
    input_shape_ = backend_->initialize(model_path, input_shape_);
    
    // Update resolution if auto-detected
    if (config_.resolution == 0 && input_shape_.size() == 4) {
        config_.resolution = static_cast<int>(input_shape_[2]);
        std::cout << "Auto-detected model input resolution: " << config_.resolution << "x" << config_.resolution << std::endl;
    }

    // Validate number of outputs
    const size_t num_outputs = backend_->get_output_count();
    const size_t num_expected = config_.model_type == ModelType::SEGMENTATION ? 3 : 2;
    
    if (num_outputs < num_expected) {
        throw std::runtime_error(
            (config_.model_type == ModelType::SEGMENTATION ? "Segmentation" : "Detection") +
            std::string(" model requires ") + std::to_string(num_expected) +
            " outputs, but model has only " + std::to_string(num_outputs)
        );
    }

    // Load COCO labels
    load_coco_labels(label_file_path);
}

void RFDETRInference::load_coco_labels(const std::filesystem::path& label_file_path) {
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

void RFDETRInference::normalize_image(std::span<float> data, size_t channel_size) {
    for (size_t c = 0; c < 3; ++c) {
        const float mean = config_.means[c];
        const float std = config_.stds[c];
        for (size_t i = 0; i < channel_size; ++i) {
            data[c * channel_size + i] = (data[c * channel_size + i] - mean) / std;
        }
    }
}

float RFDETRInference::sigmoid(float x) const noexcept {
    return 1.0f / (1.0f + std::exp(-x));
}

std::vector<float> RFDETRInference::preprocess_image(const std::filesystem::path& image_path, int& orig_h, int& orig_w) {
    if (!std::filesystem::exists(image_path)) {
        throw std::runtime_error("Image file does not exist: " + image_path.string());
    }

    cv::Mat image = cv::imread(image_path.string(), cv::IMREAD_COLOR);
    if (image.empty()) {
        throw std::runtime_error("Could not load image from: " + image_path.string());
    }
    orig_h = image.rows;
    orig_w = image.cols;

    cv::Mat resized_image;
    cv::resize(image, resized_image, cv::Size(config_.resolution, config_.resolution), 0, 0, cv::INTER_LINEAR);
    cv::cvtColor(resized_image, resized_image, cv::COLOR_BGR2RGB);
    resized_image.convertTo(resized_image, CV_32F, 1.0 / 255.0);

    const size_t input_tensor_size = 1 * 3 * config_.resolution * config_.resolution;
    std::vector<float> input_tensor_values(input_tensor_size);
    std::vector<cv::Mat> channels;
    cv::split(resized_image, channels);
    float* input_ptr = input_tensor_values.data();
    for (int c = 0; c < 3; ++c) {
        std::memcpy(input_ptr, channels[c].data, config_.resolution * config_.resolution * sizeof(float));
        input_ptr += config_.resolution * config_.resolution;
    }

    normalize_image(input_tensor_values, config_.resolution * config_.resolution);
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
        size_t size = 1;
        for (auto dim : shape) {
            size *= dim;
        }
        
        std::vector<float> data(size);
        backend_->get_output_data(i, data.data(), size);
        
        output_data_cache_.push_back(std::move(data));
        output_shapes_cache_.push_back(std::move(shape));
    }
}

void RFDETRInference::postprocess_outputs(
    float scale_w, float scale_h,
    std::vector<float>& scores,
    std::vector<int>& class_ids,
    std::vector<std::vector<float>>& boxes
) {
    if (output_data_cache_.size() < 2) {
        throw std::runtime_error("Expected at least 2 output tensors, got " + 
                                 std::to_string(output_data_cache_.size()));
    }

    const auto& dets_data = output_data_cache_[0];
    const auto& dets_shape = output_shapes_cache_[0];

    const auto& labels_data = output_data_cache_[1];
    const auto& labels_shape = output_shapes_cache_[1];

    const size_t num_detections = dets_shape[1];
    const size_t num_classes = labels_shape[2];

    for (size_t i = 0; i < num_detections; ++i) {
        const size_t det_offset = i * dets_shape[2];
        const size_t label_offset = i * num_classes;

        float max_score = -1.0f;
        int max_class_idx = -1;
        for (size_t j = 0; j < num_classes; ++j) {
            const float logit = labels_data[label_offset + j];
            const float score = sigmoid(logit);
            if (score > max_score) {
                max_score = score;
                max_class_idx = j;
            }
        }

        max_class_idx -= 1; // Fix the +1 offset

        if (max_score > config_.threshold && max_class_idx >= 0 &&
            static_cast<size_t>(max_class_idx) < coco_labels_.size()) {
            const float x_center = dets_data[det_offset + 0] * config_.resolution;
            const float y_center = dets_data[det_offset + 1] * config_.resolution;
            const float width = dets_data[det_offset + 2] * config_.resolution;
            const float height = dets_data[det_offset + 3] * config_.resolution;

            const float x_min = x_center - width / 2.0f;
            const float y_min = y_center - height / 2.0f;
            const float x_max = x_center + width / 2.0f;
            const float y_max = y_center + height / 2.0f;

            std::vector<float> box = {x_min * scale_w, y_min * scale_h, x_max * scale_w, y_max * scale_h};

            scores.push_back(max_score);
            class_ids.push_back(max_class_idx);
            boxes.push_back(std::move(box));
        }
    }
}

void RFDETRInference::postprocess_segmentation_outputs(
    float scale_w, float scale_h,
    int orig_h, int orig_w,
    std::vector<float>& scores,
    std::vector<int>& class_ids,
    std::vector<std::vector<float>>& boxes,
    std::vector<cv::Mat>& masks
) {
    if (output_data_cache_.size() != 3) {
        throw std::runtime_error("Expected 3 output tensors for segmentation, got " + 
                                 std::to_string(output_data_cache_.size()));
    }

    // Get bounding boxes data
    const auto& dets_data = output_data_cache_[0];
    const auto& dets_shape = output_shapes_cache_[0];

    // Get labels data
    const auto& labels_data = output_data_cache_[1];
    const auto& labels_shape = output_shapes_cache_[1];

    // Get masks data
    const auto& masks_data = output_data_cache_[2];
    const auto& masks_shape = output_shapes_cache_[2];

    const size_t num_detections = dets_shape[1];
    const size_t num_classes = labels_shape[2];
    const size_t mask_h = masks_shape[2];
    const size_t mask_w = masks_shape[3];

    // Compute scores and apply sigmoid
    std::vector<float> all_scores;
    std::vector<size_t> all_indices;
    
    for (size_t i = 0; i < num_detections; ++i) {
        for (size_t j = 0; j < num_classes; ++j) {
            const size_t label_offset = i * num_classes;
            const float logit = labels_data[label_offset + j];
            const float score = sigmoid(logit);
            all_scores.push_back(score);
            all_indices.push_back(i * num_classes + j);
        }
    }

    // Top-k selection
    const size_t num_select = std::min(static_cast<size_t>(config_.max_detections), all_scores.size());
    std::vector<size_t> topk_indices(all_scores.size());
    std::iota(topk_indices.begin(), topk_indices.end(), 0);
    std::partial_sort(topk_indices.begin(), topk_indices.begin() + num_select, topk_indices.end(),
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
        const size_t det_offset = detection_idx * dets_shape[2];
        const float x_center = dets_data[det_offset + 0] * config_.resolution;
        const float y_center = dets_data[det_offset + 1] * config_.resolution;
        const float width = dets_data[det_offset + 2] * config_.resolution;
        const float height = dets_data[det_offset + 3] * config_.resolution;

        // Convert to xyxy format
        const float x_min = x_center - width / 2.0f;
        const float y_min = y_center - height / 2.0f;
        const float x_max = x_center + width / 2.0f;
        const float y_max = y_center + height / 2.0f;

        std::vector<float> box = {x_min * scale_w, y_min * scale_h, x_max * scale_w, y_max * scale_h};

        // Get mask for this detection and resize to original image size
        const size_t mask_offset = detection_idx * mask_h * mask_w;
        cv::Mat mask_small(mask_h, mask_w, CV_32F);
        std::memcpy(mask_small.data, masks_data.data() + mask_offset, mask_h * mask_w * sizeof(float));

        // Resize mask to original image size using bilinear interpolation
        cv::Mat mask_resized;
        cv::resize(mask_small, mask_resized, cv::Size(orig_w, orig_h), 0, 0, cv::INTER_LINEAR);

        // Apply threshold to create binary mask
        cv::Mat binary_mask = mask_resized > config_.mask_threshold;

        scores.push_back(score);
        class_ids.push_back(class_id);
        boxes.push_back(std::move(box));
        masks.push_back(binary_mask);
    }
}

cv::Scalar RFDETRInference::get_color_for_class(int class_id) const noexcept {
    // Generate deterministic colors based on class_id
    const int hue = (class_id * 137) % 180; // Golden angle for good color distribution
    cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar(hue, 200, 200));
    cv::Mat bgr;
    cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
    return cv::Scalar(bgr.at<cv::Vec3b>(0, 0)[0], bgr.at<cv::Vec3b>(0, 0)[1], bgr.at<cv::Vec3b>(0, 0)[2]);
}

void RFDETRInference::draw_detections(
    cv::Mat& image,
    std::span<const std::vector<float>> boxes,
    std::span<const int> class_ids,
    std::span<const float> scores
) {
    if (boxes.size() != class_ids.size() || boxes.size() != scores.size()) {
        throw std::runtime_error("Mismatch in sizes of boxes, class_ids, and scores");
    }

    for (size_t i = 0; i < boxes.size(); ++i) {
        const auto& box = boxes[i];
        if (box.size() != 4) {
            throw std::runtime_error("Invalid box format at index " + std::to_string(i));
        }

        const cv::Point2f top_left(box[0], box[1]);
        const cv::Point2f bottom_right(box[2], box[3]);
        cv::rectangle(image, top_left, bottom_right, cv::Scalar(0, 0, 255), 2);

        const std::string label = coco_labels_[class_ids[i]] + ": " + std::to_string(scores[i]).substr(0, 4);
        int baseline = 0;
        constexpr double font_scale = 0.5;
        constexpr int thickness = 1;
        const cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, font_scale, thickness, &baseline);

        cv::Point2f text_pos(top_left.x, top_left.y - 5);
        if (text_pos.y - text_size.height < 0) {
            text_pos.y = top_left.y + text_size.height + 5;
        }
        if (text_pos.x + text_size.width > image.cols) {
            text_pos.x = image.cols - text_size.width - 5;
        }

        constexpr int padding = 2;
        const cv::Point2f rect_top_left(text_pos.x - padding, text_pos.y - text_size.height - padding);
        const cv::Point2f rect_bottom_right(text_pos.x + text_size.width + padding, text_pos.y + padding);
        cv::rectangle(image, rect_top_left, rect_bottom_right, cv::Scalar(0, 0, 0), cv::FILLED);

        cv::putText(
            image,
            label,
            cv::Point2f(text_pos.x, text_pos.y - padding),
            cv::FONT_HERSHEY_SIMPLEX,
            font_scale,
            cv::Scalar(255, 255, 255),
            thickness
        );
    }
}

void RFDETRInference::draw_segmentation_masks(
    cv::Mat& image,
    std::span<const std::vector<float>> boxes,
    std::span<const int> class_ids,
    std::span<const float> scores,
    std::span<const cv::Mat> masks
) {
    if (boxes.size() != class_ids.size() || boxes.size() != scores.size() || boxes.size() != masks.size()) {
        throw std::runtime_error("Mismatch in sizes of boxes, class_ids, scores, and masks");
    }

    // Create overlay for transparent masks
    cv::Mat overlay = image.clone();
    constexpr float alpha = 0.5f;

    for (size_t i = 0; i < boxes.size(); ++i) {
        const auto& box = boxes[i];
        if (box.size() != 4) {
            throw std::runtime_error("Invalid box format at index " + std::to_string(i));
        }

        const cv::Scalar color = get_color_for_class(class_ids[i]);

        // Draw mask
        const cv::Mat& mask = masks[i];
        if (mask.rows == image.rows && mask.cols == image.cols) {
            overlay.setTo(color, mask);
        }

        // Draw bounding box
        const cv::Point2f top_left(box[0], box[1]);
        const cv::Point2f bottom_right(box[2], box[3]);
        cv::rectangle(image, top_left, bottom_right, color, 2);

        // Draw label
        const std::string label = coco_labels_[class_ids[i]] + ": " + std::to_string(scores[i]).substr(0, 4);
        int baseline = 0;
        constexpr double font_scale = 0.5;
        constexpr int thickness = 1;
        const cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, font_scale, thickness, &baseline);

        cv::Point2f text_pos(top_left.x, top_left.y - 5);
        if (text_pos.y - text_size.height < 0) {
            text_pos.y = top_left.y + text_size.height + 5;
        }
        if (text_pos.x + text_size.width > image.cols) {
            text_pos.x = image.cols - text_size.width - 5;
        }

        constexpr int padding = 2;
        const cv::Point2f rect_top_left(text_pos.x - padding, text_pos.y - text_size.height - padding);
        const cv::Point2f rect_bottom_right(text_pos.x + text_size.width + padding, text_pos.y + padding);
        cv::rectangle(image, rect_top_left, rect_bottom_right, cv::Scalar(0, 0, 0), cv::FILLED);

        cv::putText(
            image,
            label,
            cv::Point2f(text_pos.x, text_pos.y - padding),
            cv::FONT_HERSHEY_SIMPLEX,
            font_scale,
            cv::Scalar(255, 255, 255),
            thickness
        );
    }

    // Blend overlay with original image
    cv::addWeighted(overlay, alpha, image, 1.0f - alpha, 0, image);
}

std::optional<std::filesystem::path> RFDETRInference::save_output_image(
    const cv::Mat& image,
    const std::filesystem::path& output_path
) {
    if (cv::imwrite(output_path.string(), image)) {
        return output_path;
    }
    return std::nullopt;
}