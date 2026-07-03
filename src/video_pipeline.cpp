#include "video_pipeline.hpp"

#include "display.hpp"
#include "video_reader.hpp"
#include "video_writer.hpp"

#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

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

std::string make_label(const std::vector<std::string> &labels, int class_id, float score) {
    const auto idx = static_cast<size_t>(class_id);
    std::string name = (idx < labels.size()) ? labels[idx] : std::string("cls") + std::to_string(class_id);
    name += ": ";
    // Keep 4 chars of the score like the previous implementation ("0.50").
    const std::string s = std::to_string(score);
    name.append(s, 0, std::min<std::size_t>(s.size(), 4));
    return name;
}

int choose_font_scale(int width, int height) {
    const int min_dim = std::max(1, std::min(width, height));
    return std::max(1, min_dim / 300);
}

void draw_on_frame(rfdetr::media::Image &image, std::span<const std::vector<float>> boxes,
                   std::span<const int> class_ids, std::span<const float> scores,
                   const std::vector<std::string> &labels) {
    const int scale = choose_font_scale(image.width, image.height);
    for (size_t i = 0; i < boxes.size(); ++i) {
        const auto color = rfdetr::media::get_color_for_class(class_ids[i]);
        rfdetr::media::draw_labeled_box(image, boxes[i], color, make_label(labels, class_ids[i], scores[i]),
                                        {255, 255, 255}, {0, 0, 0}, 2, scale);
    }
}

void draw_segmentation_on_frame(rfdetr::media::Image &image, std::span<const std::vector<float>> boxes,
                                std::span<const int> class_ids, std::span<const float> scores,
                                std::span<const rfdetr::media::Mask> masks, const std::vector<std::string> &labels) {
    rfdetr::media::draw_segmentation_masks(image, boxes, class_ids, masks);
    const int scale = choose_font_scale(image.width, image.height);
    for (size_t i = 0; i < boxes.size(); ++i) {
        const auto color = rfdetr::media::get_color_for_class(class_ids[i]);
        rfdetr::media::draw_labeled_box(image, boxes[i], color, make_label(labels, class_ids[i], scores[i]),
                                        {255, 255, 255}, {0, 0, 0}, 2, scale);
    }
}

} // anonymous namespace

VideoPipeline::VideoPipeline(const VideoPipelineConfig &config)
    : config_(config), slots_(config.ring_buffer_size), decode_to_preprocess_(config.ring_buffer_size, kPoisonPill),
      preprocess_to_infer_(config.ring_buffer_size, kPoisonPill), infer_to_draw_(config.ring_buffer_size, kPoisonPill),
      free_slots_(config.ring_buffer_size, kPoisonPill) {

    load_labels(config_.label_path, labels_);

    // Probe the input once so the writer/display can be sized before decode
    // produces the first frame.
    rfdetr::media::VideoReader probe(config_.video_path);
    width_ = probe.width();
    height_ = probe.height();
    fps_ = probe.fps();

    for (size_t i = 0; i < slots_.size(); ++i) {
        slots_[i].allocate(config_.inference_config.resolution);
        free_slots_.push(i);
    }
}

VideoPipeline::~VideoPipeline() { request_shutdown(); }

void VideoPipeline::request_shutdown() noexcept {
    stop_requested_.store(true, std::memory_order_release);
    decode_to_preprocess_.close();
    preprocess_to_infer_.close();
    infer_to_draw_.close();
    free_slots_.close();
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
    rfdetr::media::VideoReader reader(config_.video_path);

    size_t frame_num = 0;
    while (true) {
        const size_t slot_idx = free_slots_.pop();
        if (slot_idx == kPoisonPill || stop_requested_.load(std::memory_order_acquire)) {
            break;
        }

        FrameSlot &slot = slots_[slot_idx];
        if (!reader.read(slot.raw_frame)) {
            free_slots_.push(slot_idx);
            decode_to_preprocess_.push(kPoisonPill);
            break;
        }

        slot.orig_h = slot.raw_frame.height;
        slot.orig_w = slot.raw_frame.width;
        slot.frame_number = frame_num++;
        if (stop_requested_.load(std::memory_order_acquire)) {
            break;
        }
        decode_to_preprocess_.push(slot_idx);
    }
}

void VideoPipeline::preprocess_stage() {
    const int res = config_.inference_config.resolution;
    const auto &means = config_.inference_config.means;
    const auto &stds = config_.inference_config.stds;

    while (true) {
        const size_t slot_idx = decode_to_preprocess_.pop();
        if (slot_idx == kPoisonPill || stop_requested_.load(std::memory_order_acquire)) {
            preprocess_to_infer_.push(kPoisonPill);
            break;
        }

        FrameSlot &slot = slots_[slot_idx];
        rfdetr::media::preprocess_bgr_image(slot.raw_frame, slot.tensor, res, means, stds);
        if (stop_requested_.load(std::memory_order_acquire)) {
            break;
        }
        preprocess_to_infer_.push(slot_idx);
    }
}

void VideoPipeline::infer_postprocess_stage() {
    RFDETRInference inference(config_.model_path, config_.label_path, config_.inference_config);
    const auto res = static_cast<float>(inference.get_resolution());

    while (true) {
        const size_t slot_idx = preprocess_to_infer_.pop();
        if (slot_idx == kPoisonPill || stop_requested_.load(std::memory_order_acquire)) {
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
        } else if (config_.inference_config.model_type == ModelType::KEYPOINT) {
            inference.postprocess_keypoint_outputs(scale_w, scale_h, slot.orig_h, slot.orig_w, slot.scores,
                                                   slot.class_ids, slot.boxes, slot.keypoints);
            // Draw keypoints on the frame (needs the inference object for get_label_name + config)
            inference.draw_keypoints(slot.raw_frame, slot.boxes, slot.class_ids, slot.scores, slot.keypoints);
        } else {
            inference.postprocess_outputs(scale_w, scale_h, slot.scores, slot.class_ids, slot.boxes);
        }

        if (stop_requested_.load(std::memory_order_acquire)) {
            break;
        }

        infer_to_draw_.push(slot_idx);
    }
}

void VideoPipeline::draw_write_stage() {
    rfdetr::media::VideoWriter writer(config_.output_path, width_, height_, fps_);
    std::unique_ptr<rfdetr::media::Display> display;
    if (config_.display) {
        display = std::make_unique<rfdetr::media::Display>("RF-DETR Inference", width_, height_);
    }

    while (true) {
        const size_t slot_idx = infer_to_draw_.pop();
        if (slot_idx == kPoisonPill || stop_requested_.load(std::memory_order_acquire)) {
            break;
        }

        FrameSlot &slot = slots_[slot_idx];

        if (config_.inference_config.model_type == ModelType::SEGMENTATION) {
            draw_segmentation_on_frame(slot.raw_frame, slot.boxes, slot.class_ids, slot.scores, slot.masks, labels_);
        } else if (config_.inference_config.model_type != ModelType::KEYPOINT) {
            // Keypoint frames were already annotated in infer_postprocess_stage.
            draw_on_frame(slot.raw_frame, slot.boxes, slot.class_ids, slot.scores, labels_);
        }

        writer.write(slot.raw_frame);

        if (display != nullptr) {
            if (!display->show(slot.raw_frame)) {
                request_shutdown();
                break;
            }
        }

        frames_processed_.fetch_add(1, std::memory_order_relaxed);
        free_slots_.push(slot_idx);
    }
}

} // namespace rfdetr::video
