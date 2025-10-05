#include "rfdetr_inference.hpp"
#include <fstream>
#include <stdexcept>
#include <cmath>
#include <numeric>
#include <algorithm>

RFDETRInference::RFDETRInference(
    const std::filesystem::path& model_path,
    const std::filesystem::path& label_file_path,
    const Config& config
)
    : env_(std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "RFDETRInference")),
      memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)),
      config_(config),
      input_shape_({1, 3, config_.resolution, config_.resolution}) {
    // Validate model path
    if (!std::filesystem::exists(model_path)) {
        throw std::runtime_error("Model file does not exist: " + model_path.string());
    }

    // Initialize ONNX Runtime session
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    session_ = std::make_unique<Ort::Session>(*env_, model_path.c_str(), session_options);

    // Auto-detect input shape from model if resolution is set to 0
    if (config_.resolution == 0) {
        Ort::TypeInfo input_type_info = session_->GetInputTypeInfo(0);
        auto tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
        auto shape = tensor_info.GetShape();
        
        if (shape.size() == 4 && shape[2] == shape[3] && shape[2] > 0) {
            const_cast<Config&>(config_).resolution = static_cast<int>(shape[2]);
            input_shape_ = {1, 3, shape[2], shape[3]};
            std::cout << "Auto-detected model input resolution: " << config_.resolution << "x" << config_.resolution << std::endl;
        } else {
            throw std::runtime_error("Could not auto-detect valid input resolution from model. Please specify resolution in config.");
        }
    } else {
        input_shape_ = {1, 3, config_.resolution, config_.resolution};
    }

    // Auto-detect output names from model
    const size_t num_outputs = session_->GetOutputCount();
    std::cout << "Model has " << num_outputs << " outputs:" << std::endl;
    
    // Validate we have the expected number of outputs
    if (config_.model_type == ModelType::SEGMENTATION && num_outputs < 3) {
        throw std::runtime_error("Segmentation model requires 3 outputs, but model has only " + 
                                 std::to_string(num_outputs));
    }
    if (config_.model_type == ModelType::DETECTION && num_outputs < 2) {
        throw std::runtime_error("Detection model requires 2 outputs, but model has only " + 
                                 std::to_string(num_outputs));
    }
    
    // Store output names - first collect all strings, then get pointers
    const size_t num_expected = config_.model_type == ModelType::SEGMENTATION ? 3 : 2;
    output_name_strings_.reserve(num_expected); // Prevent reallocation
    
    for (size_t i = 0; i < num_expected; ++i) {
        Ort::AllocatedStringPtr output_name_ptr = session_->GetOutputNameAllocated(i, allocator_);
        std::string output_name(output_name_ptr.get());
        std::cout << "  Output " << i << ": " << output_name << std::endl;
        output_name_strings_.push_back(output_name);
    }
    
    // Now get pointers after all strings are stored (vector won't reallocate)
    for (const auto& name : output_name_strings_) {
        output_names_.push_back(name.c_str());
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

std::vector<Ort::Value> RFDETRInference::run_inference(std::span<const float> input_data) {
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info_,
        const_cast<float*>(input_data.data()),
        input_data.size(),
        input_shape_.data(),
        input_shape_.size()
    );
    if (!input_tensor.IsTensor()) {
        throw std::runtime_error("Failed to create input tensor");
    }

    return session_->Run(
        Ort::RunOptions{nullptr},
        &input_name_,
        &input_tensor,
        1,
        output_names_.data(),
        output_names_.size()
    );
}

void RFDETRInference::postprocess_outputs(
    std::span<const Ort::Value> output_tensors,
    float scale_w, float scale_h,
    std::vector<float>& scores,
    std::vector<int>& class_ids,
    std::vector<std::vector<float>>& boxes
) {
    if (output_tensors.size() != output_names_.size()) {
        throw std::runtime_error("Expected " + std::to_string(output_names_.size()) +
                                 " output tensors, got " + std::to_string(output_tensors.size()));
    }

    const Ort::Value& dets_tensor = output_tensors[0];
    if (!dets_tensor.IsTensor()) {
        throw std::runtime_error("dets output is not a tensor");
    }
    Ort::TensorTypeAndShapeInfo dets_info = dets_tensor.GetTensorTypeAndShapeInfo();
    const auto dets_shape = dets_info.GetShape();
    const float* dets_data = dets_tensor.GetTensorData<float>();

    const Ort::Value& labels_tensor = output_tensors[1];
    if (!labels_tensor.IsTensor()) {
        throw std::runtime_error("labels output is not a tensor");
    }
    Ort::TensorTypeAndShapeInfo labels_info = labels_tensor.GetTensorTypeAndShapeInfo();
    const auto labels_shape = labels_info.GetShape();
    const float* labels_data = labels_tensor.GetTensorData<float>();

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
    std::span<const Ort::Value> output_tensors,
    float scale_w, float scale_h,
    int orig_h, int orig_w,
    std::vector<float>& scores,
    std::vector<int>& class_ids,
    std::vector<std::vector<float>>& boxes,
    std::vector<cv::Mat>& masks
) {
    if (output_tensors.size() != 3) {
        throw std::runtime_error("Expected 3 output tensors for segmentation, got " + std::to_string(output_tensors.size()));
    }

    // Get bounding boxes tensor
    const Ort::Value& dets_tensor = output_tensors[0];
    if (!dets_tensor.IsTensor()) {
        throw std::runtime_error("dets output is not a tensor");
    }
    Ort::TensorTypeAndShapeInfo dets_info = dets_tensor.GetTensorTypeAndShapeInfo();
    const auto dets_shape = dets_info.GetShape();
    const float* dets_data = dets_tensor.GetTensorData<float>();

    // Get labels tensor
    const Ort::Value& labels_tensor = output_tensors[1];
    if (!labels_tensor.IsTensor()) {
        throw std::runtime_error("labels output is not a tensor");
    }
    Ort::TensorTypeAndShapeInfo labels_info = labels_tensor.GetTensorTypeAndShapeInfo();
    const auto labels_shape = labels_info.GetShape();
    const float* labels_data = labels_tensor.GetTensorData<float>();

    // Get masks tensor
    const Ort::Value& masks_tensor = output_tensors[2];
    if (!masks_tensor.IsTensor()) {
        throw std::runtime_error("masks output is not a tensor");
    }
    Ort::TensorTypeAndShapeInfo masks_info = masks_tensor.GetTensorTypeAndShapeInfo();
    const auto masks_shape = masks_info.GetShape();
    const float* masks_data = masks_tensor.GetTensorData<float>();

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
        std::memcpy(mask_small.data, masks_data + mask_offset, mask_h * mask_w * sizeof(float));

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