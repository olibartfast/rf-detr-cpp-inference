#pragma once

#include "rfdetr_inference.hpp"

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <queue>
#include <thread>
#include <vector>

namespace rfdetr::video {

/// Sentinel value pushed through queues to signal shutdown.
inline constexpr size_t kPoisonPill = SIZE_MAX;

/// Pre-allocated slot holding all per-frame data. Exactly one thread accesses
/// a slot at any given time — ownership is transferred via queue indices.
struct FrameSlot {
    cv::Mat raw_frame;
    int orig_h{0};
    int orig_w{0};
    std::vector<float> tensor; // pre-allocated to 3 * res * res
    std::vector<float> scores;
    std::vector<int> class_ids;
    std::vector<std::vector<float>> boxes;
    std::vector<cv::Mat> masks; // segmentation only
    size_t frame_number{0};

    void allocate(int resolution) {
        const auto res = static_cast<size_t>(resolution);
        tensor.resize(3 * res * res);
    }

    void clear_results() {
        scores.clear();
        class_ids.clear();
        boxes.clear();
        masks.clear();
    }
};

/// Thread-safe bounded queue. push() blocks when full; pop() blocks when empty.
template <typename T> class BoundedQueue {
  public:
    explicit BoundedQueue(size_t capacity) : capacity_(capacity) {}

    BoundedQueue(const BoundedQueue &) = delete;
    BoundedQueue &operator=(const BoundedQueue &) = delete;

    void push(T value) {
        std::unique_lock lock(mutex_);
        not_full_.wait(lock, [this] { return queue_.size() < capacity_; });
        queue_.push(std::move(value));
        lock.unlock();
        not_empty_.notify_one();
    }

    T pop() {
        std::unique_lock lock(mutex_);
        not_empty_.wait(lock, [this] { return !queue_.empty(); });
        T value = std::move(queue_.front());
        queue_.pop();
        lock.unlock();
        not_full_.notify_one();
        return value;
    }

  private:
    std::queue<T> queue_;
    size_t capacity_;
    std::mutex mutex_;
    std::condition_variable not_full_;
    std::condition_variable not_empty_;
};

/// Configuration for the video processing pipeline.
struct VideoPipelineConfig {
    std::filesystem::path video_path;
    std::filesystem::path model_path;
    std::filesystem::path label_path;
    std::filesystem::path output_path{"output_video.mp4"};
    Config inference_config;
    size_t ring_buffer_size{8};
    bool display{false};
};

/// Four-stage ring buffer pipeline for video inference.
///
/// Stages: Decode → Preprocess → Infer+Postprocess → Draw+Write
/// Each stage runs on its own std::jthread. Stages communicate by passing
/// slot indices through bounded queues — zero frame copies between stages.
class VideoPipeline {
  public:
    explicit VideoPipeline(const VideoPipelineConfig &config);
    ~VideoPipeline();

    VideoPipeline(const VideoPipeline &) = delete;
    VideoPipeline &operator=(const VideoPipeline &) = delete;

    /// Run the pipeline to completion (blocking). Returns total frames processed.
    size_t run();

  private:
    void decode_stage();
    void preprocess_stage();
    void infer_postprocess_stage();
    void draw_write_stage();

    VideoPipelineConfig config_;
    std::vector<std::string> labels_;

    // Ring buffer
    std::vector<FrameSlot> slots_;

    // Inter-stage queues (carry slot indices)
    BoundedQueue<size_t> decode_to_preprocess_;
    BoundedQueue<size_t> preprocess_to_infer_;
    BoundedQueue<size_t> infer_to_draw_;
    BoundedQueue<size_t> free_slots_;

    // Threads
    std::jthread decode_thread_;
    std::jthread preprocess_thread_;
    std::jthread infer_thread_;
    std::jthread draw_thread_;

    std::atomic<size_t> frames_processed_{0};
};

} // namespace rfdetr::video
