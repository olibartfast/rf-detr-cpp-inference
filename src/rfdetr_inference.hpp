#pragma once
#include "backends/inference_backend.hpp"
#include "media.hpp"

#if defined(USE_CUDA_POSTPROCESS) || defined(USE_DALI)
#include "gpu/gpu_context.hpp"
#endif
#ifdef USE_CUDA_POSTPROCESS
#include "gpu/rfdetr_postprocess.hpp"
#endif
#ifdef USE_DALI
#include "gpu/dali_preprocessor.hpp"
#endif

#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

// Bring backend namespace into scope for convenience
using rfdetr::backend::create_backend;
using rfdetr::backend::InferenceBackend;

enum class ModelType { DETECTION, SEGMENTATION, KEYPOINT };

struct Config {
    int resolution{560};
    float threshold{0.5f};
    std::array<float, 3> means{0.485f, 0.456f, 0.406f};
    std::array<float, 3> stds{0.229f, 0.224f, 0.225f};
    ModelType model_type{ModelType::DETECTION};
    int max_detections{300}; ///< Cap on ranked query/class candidates, upstream's `num_select`
    float mask_threshold{0.0f};

    /// Exported logit slot that holds background, excluded before ranking.
    ///
    /// Mirrors the `background_class_id` argument rfdetr 1.9.4 added to its own
    /// ONNX/TFLite decoders. Negative values count from the end (`-1` = final
    /// slot) and `std::nullopt` keeps every slot. The default 0 matches the
    /// background-first layout the shipped RF-DETR exports use: logit 0 is
    /// background and logit *n* is COCO category *n*, so the surviving slots map
    /// in order onto the label file (`data/coco-labels-91.txt`, indexed by COCO
    /// id). Upstream's own default is `-1`, which mis-decodes exactly these
    /// checkpoints — their final slot is the real category 90.
    std::optional<int> background_class_id{0};

    // Keypoint-specific configuration
    std::vector<int> keypoint_counts{
        0, 17}; ///< num_keypoints_per keypoint-class, default COCO: {background: 0, person: 17}
    std::vector<std::string> keypoint_names{
        "nose",           "left_eye",   "right_eye",   "left_ear",   "right_ear",   "left_shoulder",
        "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip",
        "right_hip",      "left_knee",  "right_knee",  "left_ankle", "right_ankle"};
    std::vector<std::pair<int, int>> skeleton{{15, 13}, {13, 11}, {16, 14}, {14, 12}, {11, 12}, {5, 11}, {6, 12},
                                              {5, 6},   {5, 7},   {6, 8},   {7, 9},   {8, 10},  {1, 2},  {0, 1},
                                              {0, 2},   {1, 3},   {2, 4},   {3, 5},   {4, 6}};
    float keypoint_uncertainty_alpha{0.2f};         ///< Uncertainty-weighted score fusion; 0 = disable
    bool draw_uncertainty{false};                   ///< Draw uncertainty ellipses on keypoints
    rfdetr::media::Color keypoint_color{0, 255, 0}; ///< Default keypoint color (green)

    // --- GPU pipeline (opt-in; requires the TensorRT backend) ---------------
    bool gpu_preprocess{false};                           ///< Preprocess with DALI on the GPU
    bool gpu_postprocess{false};                          ///< Postprocess segmentation with CUDA kernels
    std::filesystem::path dali_pipeline_dir{"data/dali"}; ///< Where the .dali files live
    int gpu_device_id{0};
};

class RFDETRInference {
  public:
    RFDETRInference(const std::filesystem::path &model_path, const std::filesystem::path &label_file_path,
                    const Config &config = Config{});

    // Test-friendly constructor: inject a custom backend (skips backend creation and model loading)
    RFDETRInference(std::unique_ptr<InferenceBackend> backend, const std::filesystem::path &label_file_path,
                    const Config &config = Config{});

    ~RFDETRInference() = default;

    // Preprocess the input image (from file path)
    std::vector<float> preprocess_image(const std::filesystem::path &image_path, int &orig_h, int &orig_w);

    // Preprocess the input image (from an in-memory BGR image, avoids disk I/O for video frames)
    std::vector<float> preprocess_image(const rfdetr::media::Image &bgr_image, int &orig_h, int &orig_w);

    // Run inference
    void run_inference(std::span<const float> input_data);

    // Post-process the inference outputs for detection
    void postprocess_outputs(float scale_w, float scale_h, std::vector<float> &scores, std::vector<int> &class_ids,
                             std::vector<BoundingBox> &boxes);

    // Post-process the inference outputs for segmentation
    void postprocess_segmentation_outputs(float scale_w, float scale_h, int orig_h, int orig_w,
                                          std::vector<float> &scores, std::vector<int> &class_ids,
                                          std::vector<BoundingBox> &boxes, std::vector<rfdetr::media::Mask> &masks);

    // Draw detections on the image
    void draw_detections(rfdetr::media::Image &image, std::span<const BoundingBox> boxes,
                         std::span<const int> class_ids, std::span<const float> scores);

    // Draw segmentation masks on the image
    void draw_segmentation_masks(rfdetr::media::Image &image, std::span<const BoundingBox> boxes,
                                 std::span<const int> class_ids, std::span<const float> scores,
                                 std::span<const rfdetr::media::Mask> masks);

    // Post-process inference outputs for keypoint detection
    void postprocess_keypoint_outputs(float scale_w, float scale_h, int orig_h, int orig_w, std::vector<float> &scores,
                                      std::vector<int> &class_ids, std::vector<BoundingBox> &boxes,
                                      std::vector<std::vector<KeypointResult>> &keypoints);

    // Draw keypoints on the image
    void draw_keypoints(rfdetr::media::Image &image, std::span<const BoundingBox> boxes, std::span<const int> class_ids,
                        std::span<const float> scores, std::span<const std::vector<KeypointResult>> keypoints);

    // Save the output image
    std::optional<std::filesystem::path> save_output_image(const rfdetr::media::Image &image,
                                                           const std::filesystem::path &output_path);

    // Getters for testing
    [[nodiscard]] const std::vector<std::string> &get_coco_labels() const noexcept { return coco_labels_; }
    [[nodiscard]] int get_resolution() const noexcept { return config_.resolution; }

    // Get label name by class index (with bounds check)
    [[nodiscard]] std::string get_label_name(int class_id) const;

    /// True if the GPU pipeline is compiled in, enabled, and the backend and
    /// device support it. False makes every gpu_* entry point below a no-op that
    /// the caller must not invoke.
    [[nodiscard]] bool gpu_preprocess_active() const noexcept;
    [[nodiscard]] bool gpu_postprocess_active() const noexcept;

#if defined(USE_CUDA_POSTPROCESS) || defined(USE_DALI)
    /// Preprocess straight into the backend's input binding and run inference on
    /// the device, leaving outputs in device memory. Requires an active GPU
    /// preprocess path. `orig_h`/`orig_w` are reported back for box scaling.
    void run_gpu_image(const std::filesystem::path &image_path, int &orig_h, int &orig_w);

    /// Same, but for an already-decoded BGR frame (the video path). The frame is
    /// uploaded once and preprocessed by the DALI `frame` pipeline; outputs stay
    /// in device memory.
    void run_gpu_frame(const rfdetr::media::Image &bgr_frame);

    /// Segmentation postprocessing on the GPU, reading the backend's output
    /// bindings. Must follow run_gpu_image() or run_inference_device().
    void postprocess_segmentation_outputs_gpu(float scale_w, float scale_h, int orig_h, int orig_w,
                                              std::vector<float> &scores, std::vector<int> &class_ids,
                                              std::vector<BoundingBox> &boxes, std::vector<rfdetr::media::Mask> &masks);

    /// Copies the backend's device outputs into the host cache so the existing
    /// CPU postprocessors can run after a device-side inference.
    void fetch_device_outputs();
#endif

  private:
    // Load COCO labels from file
    void load_coco_labels(const std::filesystem::path &label_file_path);

    // Inference backend (Strategy Pattern)
    std::unique_ptr<InferenceBackend> backend_;

    // Model parameters
    std::vector<std::string> coco_labels_;
    Config config_;
    std::vector<int64_t> input_shape_;

    // Output tensor cache
    std::vector<std::vector<float>> output_data_cache_;
    std::vector<std::vector<int64_t>> output_shapes_cache_;

    /// Reusable (num_queries, foreground classes) score grid that top-k ranking
    /// consumes, so a video run does not reallocate it per frame.
    std::vector<float> score_grid_;

#if defined(USE_CUDA_POSTPROCESS) || defined(USE_DALI)
    /// Lazily built on first use so a CPU-only run never touches the device.
    void ensure_gpu_ready();

    bool gpu_ready_{false};
#endif
#ifdef USE_DALI
    std::unique_ptr<rfdetr::gpu::DaliPreprocessor> dali_encoded_;
    std::unique_ptr<rfdetr::gpu::DaliPreprocessor> dali_frame_;
    /// Reusable host buffer for the encoded image bytes.
    std::vector<uint8_t> encoded_bytes_;
    /// Reusable device staging buffer for decoded BGR video frames.
    rfdetr::gpu::DeviceBuffer frame_device_;
#endif
#ifdef USE_CUDA_POSTPROCESS
    std::unique_ptr<rfdetr::gpu::SegPostprocessor> seg_postprocessor_;
    rfdetr::gpu::SegPostprocessResult seg_result_;
#endif
};
