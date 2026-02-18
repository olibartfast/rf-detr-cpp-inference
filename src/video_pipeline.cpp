#include "video_pipeline.hpp"

#include "processing_utils.hpp"

#include <fstream>
#include <iostream>
#include <stdexcept>

namespace rfdetr::video {

namespace {

void load_labels(const std::filesystem::path &path, std::vector<std::string> &labels) {
    if (!std::filesystem::exists(path)) {
        throw std::runtime_error("Label file does not exist: " + path.string());
    }
    std::ifstream file(path);
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty()) {
            labels.push_back(line);
        }
    }
    if (labels.empty()) {
        throw std::runtime_error("No labels found in file: " + path.string());
    }
}

void draw_on_frame(cv::Mat &image, std::span<const std::vector<float>> boxes, std::span<const int> class_ids,
                   std::span<const float> scores, const std::vector<std::string> &labels) {
    for (size_t i = 0; i < boxes.size(); ++i) {
        const auto &box = boxes[i];
        const cv::Point2f top_left(box[0], box[1]);
        const cv::Point2f bottom_right(box[2], box[3]);
        cv::rectangle(image, top_left, bottom_right, cv::Scalar(0, 0, 255), 2);

        const std::string label =
            labels[static_cast<size_t>(class_ids[i])] + ": " + std::to_string(scores[i]).substr(0, 4);
        int baseline = 0;
        constexpr double font_scale = 0.5;
        constexpr int thickness = 1;
        const cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, font_scale, thickness, &baseline);
        const auto text_h = static_cast<float>(text_size.height);
        const auto text_w = static_cast<float>(text_size.width);

        cv::Point2f text_pos(top_left.x, top_left.y - 5);
        if (text_pos.y - text_h < 0) {
            text_pos.y = top_left.y + text_h + 5;
        }
        if (text_pos.x + text_w > static_cast<float>(image.cols)) {
            text_pos.x = static_cast<float>(image.cols) - text_w - 5;
        }

        constexpr int padding = 2;
        const cv::Point2f rect_tl(text_pos.x - padding, text_pos.y - text_h - padding);
        const cv::Point2f rect_br(text_pos.x + text_w + padding, text_pos.y + padding);
        cv::rectangle(image, rect_tl, rect_br, cv::Scalar(0, 0, 0), cv::FILLED);
        cv::putText(image, label, cv::Point2f(text_pos.x, text_pos.y - padding), cv::FONT_HERSHEY_SIMPLEX, font_scale,
                    cv::Scalar(255, 255, 255), thickness);
    }
}

void draw_segmentation_on_frame(cv::Mat &image, std::span<const std::vector<float>> boxes,
                                std::span<const int> class_ids, std::span<const float> scores,
                                std::span<const cv::Mat> masks, const std::vector<std::string> &labels) {
    cv::Mat overlay = image.clone();
    constexpr float alpha = 0.5f;

    for (size_t i = 0; i < boxes.size(); ++i) {
        const auto &box = boxes[i];
        const cv::Scalar color = rfdetr::processing::get_color_for_class(class_ids[i]);

        const cv::Mat &mask = masks[i];
        if (mask.rows == image.rows && mask.cols == image.cols) {
            overlay.setTo(color, mask);
        }

        const cv::Point2f top_left(box[0], box[1]);
        const cv::Point2f bottom_right(box[2], box[3]);
        cv::rectangle(image, top_left, bottom_right, color, 2);

        const std::string label =
            labels[static_cast<size_t>(class_ids[i])] + ": " + std::to_string(scores[i]).substr(0, 4);
        int baseline = 0;
        constexpr double font_scale = 0.5;
        constexpr int thickness = 1;
        const cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, font_scale, thickness, &baseline);
        const auto text_h = static_cast<float>(text_size.height);
        const auto text_w = static_cast<float>(text_size.width);

        cv::Point2f text_pos(top_left.x, top_left.y - 5);
        if (text_pos.y - text_h < 0) {
            text_pos.y = top_left.y + text_h + 5;
        }
        if (text_pos.x + text_w > static_cast<float>(image.cols)) {
            text_pos.x = static_cast<float>(image.cols) - text_w - 5;
        }

        constexpr int padding = 2;
        const cv::Point2f rect_tl(text_pos.x - padding, text_pos.y - text_h - padding);
        const cv::Point2f rect_br(text_pos.x + text_w + padding, text_pos.y + padding);
        cv::rectangle(image, rect_tl, rect_br, cv::Scalar(0, 0, 0), cv::FILLED);
        cv::putText(image, label, cv::Point2f(text_pos.x, text_pos.y - padding), cv::FONT_HERSHEY_SIMPLEX, font_scale,
                    cv::Scalar(255, 255, 255), thickness);
    }

    cv::addWeighted(overlay, alpha, image, 1.0f - alpha, 0, image);
}

} // anonymous namespace

VideoPipeline::VideoPipeline(const VideoPipelineConfig &config)
    : config_(config), slots_(config.ring_buffer_size), decode_to_preprocess_(config.ring_buffer_size),
      preprocess_to_infer_(config.ring_buffer_size), infer_to_draw_(config.ring_buffer_size),
      free_slots_(config.ring_buffer_size) {

    load_labels(config_.label_path, labels_);

    for (size_t i = 0; i < slots_.size(); ++i) {
        slots_[i].allocate(config_.inference_config.resolution);
        free_slots_.push(i);
    }
}

VideoPipeline::~VideoPipeline() {
    // Safety net: push poison pills to unblock any threads still waiting.
    // Under normal flow, threads have already exited via the poison pill chain.
    decode_to_preprocess_.push(kPoisonPill);
    preprocess_to_infer_.push(kPoisonPill);
    infer_to_draw_.push(kPoisonPill);
    free_slots_.push(kPoisonPill);
}

size_t VideoPipeline::run() {
    // Launch consumers before producers so they are ready to pop
    draw_thread_ = std::jthread([this] { draw_write_stage(); });
    infer_thread_ = std::jthread([this] { infer_postprocess_stage(); });
    preprocess_thread_ = std::jthread([this] { preprocess_stage(); });
    decode_thread_ = std::jthread([this] { decode_stage(); });

    decode_thread_.join();
    preprocess_thread_.join();
    infer_thread_.join();
    draw_thread_.join();

    return frames_processed_.load();
}

void VideoPipeline::decode_stage() {
    cv::VideoCapture cap(config_.video_path.string());
    if (!cap.isOpened()) {
        throw std::runtime_error("Cannot open video: " + config_.video_path.string());
    }

    size_t frame_num = 0;
    while (true) {
        const size_t slot_idx = free_slots_.pop();
        if (slot_idx == kPoisonPill) {
            break;
        }

        FrameSlot &slot = slots_[slot_idx];
        if (!cap.read(slot.raw_frame)) {
            free_slots_.push(slot_idx);
            decode_to_preprocess_.push(kPoisonPill);
            break;
        }

        slot.orig_h = slot.raw_frame.rows;
        slot.orig_w = slot.raw_frame.cols;
        slot.frame_number = frame_num++;
        decode_to_preprocess_.push(slot_idx);
    }
}

void VideoPipeline::preprocess_stage() {
    const int res = config_.inference_config.resolution;
    const auto &means = config_.inference_config.means;
    const auto &stds = config_.inference_config.stds;

    while (true) {
        const size_t slot_idx = decode_to_preprocess_.pop();
        if (slot_idx == kPoisonPill) {
            preprocess_to_infer_.push(kPoisonPill);
            break;
        }

        FrameSlot &slot = slots_[slot_idx];
        rfdetr::processing::preprocess_frame(slot.raw_frame, slot.tensor, res, means, stds);
        preprocess_to_infer_.push(slot_idx);
    }
}

void VideoPipeline::infer_postprocess_stage() {
    RFDETRInference inference(config_.model_path, config_.label_path, config_.inference_config);
    const auto res = static_cast<float>(inference.get_resolution());

    while (true) {
        const size_t slot_idx = preprocess_to_infer_.pop();
        if (slot_idx == kPoisonPill) {
            infer_to_draw_.push(kPoisonPill);
            break;
        }

        FrameSlot &slot = slots_[slot_idx];
        slot.clear_results();

        inference.run_inference(slot.tensor);

        const float scale_w = static_cast<float>(slot.orig_w) / res;
        const float scale_h = static_cast<float>(slot.orig_h) / res;

        if (config_.inference_config.model_type == ModelType::SEGMENTATION) {
            inference.postprocess_segmentation_outputs(scale_w, scale_h, slot.orig_h, slot.orig_w, slot.scores,
                                                       slot.class_ids, slot.boxes, slot.masks);
        } else {
            inference.postprocess_outputs(scale_w, scale_h, slot.scores, slot.class_ids, slot.boxes);
        }

        infer_to_draw_.push(slot_idx);
    }
}

void VideoPipeline::draw_write_stage() {
    cv::VideoWriter writer;
    bool writer_initialized = false;

    cv::VideoCapture cap(config_.video_path.string());
    const double fps = cap.get(cv::CAP_PROP_FPS);
    cap.release();

    while (true) {
        const size_t slot_idx = infer_to_draw_.pop();
        if (slot_idx == kPoisonPill) {
            break;
        }

        FrameSlot &slot = slots_[slot_idx];

        if (!writer_initialized) {
            const int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
            writer.open(config_.output_path.string(), fourcc, fps, cv::Size(slot.orig_w, slot.orig_h));
            if (!writer.isOpened()) {
                throw std::runtime_error("Cannot open video writer: " + config_.output_path.string());
            }
            writer_initialized = true;
        }

        if (config_.inference_config.model_type == ModelType::SEGMENTATION) {
            draw_segmentation_on_frame(slot.raw_frame, slot.boxes, slot.class_ids, slot.scores, slot.masks, labels_);
        } else {
            draw_on_frame(slot.raw_frame, slot.boxes, slot.class_ids, slot.scores, labels_);
        }

        writer.write(slot.raw_frame);

        if (config_.display) {
            cv::imshow("RF-DETR Inference", slot.raw_frame);
            if (cv::waitKey(1) == 27) { // ESC to quit early
                break;
            }
        }

        frames_processed_.fetch_add(1, std::memory_order_relaxed);
        free_slots_.push(slot_idx);
    }

    writer.release();
    if (config_.display) {
        cv::destroyAllWindows();
    }
}

} // namespace rfdetr::video
