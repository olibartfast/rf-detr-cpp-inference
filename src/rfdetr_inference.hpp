#pragma once
#include "backends/inference_backend.hpp"

#include <filesystem>
#include <memory>
#include <opencv2/opencv.hpp>
#include <optional>
#include <span>
#include <string>
#include <vector>

// Bring backend namespace into scope for convenience
using rfdetr::backend::create_backend;
using rfdetr::backend::InferenceBackend;

enum class ModelType { DETECTION, SEGMENTATION, KEYPOINT };

/// A single detected keypoint with associated metadata.
struct KeypointResult {
    float x;           ///< Pixel x-coordinate
    float y;           ///< Pixel y-coordinate
    float findability; ///< Sigmoid(findability_logit) — [0, 1], radius multiplier
    float visibility;  ///< Sigmoid(visibility_logit) — [0, 1], occlusion flag
    float cov[4];      ///< 2x2 pixel covariance matrix, row-major: [a, b, b, c]
};

struct Config {
    int resolution{560};
    float threshold{0.5f};
    std::array<float, 3> means{0.485f, 0.456f, 0.406f};
    std::array<float, 3> stds{0.229f, 0.224f, 0.225f};
    ModelType model_type{ModelType::DETECTION};
    int max_detections{300};
    float mask_threshold{0.0f};

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
    float keypoint_uncertainty_alpha{0.2f}; ///< Uncertainty-weighted score fusion; 0 = disable
    bool draw_uncertainty{false};           ///< Draw uncertainty ellipses on keypoints
    cv::Scalar keypoint_color{0, 255, 0};   ///< Default keypoint color (green)
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

    // Preprocess the input image (from cv::Mat, avoids disk I/O for video frames)
    std::vector<float> preprocess_image(const cv::Mat &bgr_image, int &orig_h, int &orig_w);

    // Run inference
    void run_inference(std::span<const float> input_data);

    // Post-process the inference outputs for detection
    void postprocess_outputs(float scale_w, float scale_h, std::vector<float> &scores, std::vector<int> &class_ids,
                             std::vector<std::vector<float>> &boxes);

    // Post-process the inference outputs for segmentation
    void postprocess_segmentation_outputs(float scale_w, float scale_h, int orig_h, int orig_w,
                                          std::vector<float> &scores, std::vector<int> &class_ids,
                                          std::vector<std::vector<float>> &boxes, std::vector<cv::Mat> &masks);

    // Draw detections on the image
    void draw_detections(cv::Mat &image, std::span<const std::vector<float>> boxes, std::span<const int> class_ids,
                         std::span<const float> scores);

    // Draw segmentation masks on the image
    void draw_segmentation_masks(cv::Mat &image, std::span<const std::vector<float>> boxes,
                                 std::span<const int> class_ids, std::span<const float> scores,
                                 std::span<const cv::Mat> masks);

    // Post-process inference outputs for keypoint detection
    void postprocess_keypoint_outputs(float scale_w, float scale_h, int orig_h, int orig_w, std::vector<float> &scores,
                                      std::vector<int> &class_ids, std::vector<std::vector<float>> &boxes,
                                      std::vector<std::vector<KeypointResult>> &keypoints);

    // Draw keypoints on the image
    void draw_keypoints(cv::Mat &image, std::span<const std::vector<float>> boxes, std::span<const int> class_ids,
                        std::span<const float> scores, std::span<const std::vector<KeypointResult>> keypoints);

    // Save the output image
    std::optional<std::filesystem::path> save_output_image(const cv::Mat &image,
                                                           const std::filesystem::path &output_path);

    // Getters for testing
    [[nodiscard]] const std::vector<std::string> &get_coco_labels() const noexcept { return coco_labels_; }
    [[nodiscard]] int get_resolution() const noexcept { return config_.resolution; }

    // Get label name by class index (with bounds check)
    [[nodiscard]] std::string get_label_name(int class_id) const;

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
};
