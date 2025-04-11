#include "rfdetr_inference.hpp"
#include <fstream>
#include <stdexcept>
#include <cmath>

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

std::optional<std::filesystem::path> RFDETRInference::save_output_image(
    const cv::Mat& image,
    const std::filesystem::path& output_path
) {
    if (cv::imwrite(output_path.string(), image)) {
        return output_path;
    }
    return std::nullopt;
}