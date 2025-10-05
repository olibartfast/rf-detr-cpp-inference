#pragma once
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <memory>
#include <span>
#include <optional>
#include <filesystem>

enum class ModelType {
    DETECTION,
    SEGMENTATION
};

struct Config {
    int resolution{560};
    float threshold{0.5f};
    std::array<float, 3> means{0.485f, 0.456f, 0.406f};
    std::array<float, 3> stds{0.229f, 0.224f, 0.225f};
    ModelType model_type{ModelType::DETECTION};
    int max_detections{300};
    float mask_threshold{0.0f};
};


class RFDETRInference {
    public:

        RFDETRInference(
            const std::filesystem::path& model_path,
            const std::filesystem::path& label_file_path,
            const Config& config = Config{}
        );
        ~RFDETRInference() = default;
    
        // Preprocess the input image
        std::vector<float> preprocess_image(const std::filesystem::path& image_path, int& orig_h, int& orig_w);
    
        // Run inference
        std::vector<Ort::Value> run_inference(std::span<const float> input_data);
    
        // Post-process the inference outputs for detection
        void postprocess_outputs(
            std::span<const Ort::Value> output_tensors,
            float scale_w, float scale_h,
            std::vector<float>& scores,
            std::vector<int>& class_ids,
            std::vector<std::vector<float>>& boxes
        );
    
        // Post-process the inference outputs for segmentation
        void postprocess_segmentation_outputs(
            std::span<const Ort::Value> output_tensors,
            float scale_w, float scale_h,
            int orig_h, int orig_w,
            std::vector<float>& scores,
            std::vector<int>& class_ids,
            std::vector<std::vector<float>>& boxes,
            std::vector<cv::Mat>& masks
        );
    
        // Draw detections on the image
        void draw_detections(
            cv::Mat& image,
            std::span<const std::vector<float>> boxes,
            std::span<const int> class_ids,
            std::span<const float> scores
        );
    
        // Draw segmentation masks on the image
        void draw_segmentation_masks(
            cv::Mat& image,
            std::span<const std::vector<float>> boxes,
            std::span<const int> class_ids,
            std::span<const float> scores,
            std::span<const cv::Mat> masks
        );
    
        // Save the output image
        std::optional<std::filesystem::path> save_output_image(
            const cv::Mat& image,
            const std::filesystem::path& output_path
        );
    
        // Getters for testing
        const std::vector<std::string>& get_coco_labels() const noexcept { return coco_labels_; }
        int get_resolution() const noexcept { return config_.resolution; }
    
    private:
        // Load COCO labels from file
        void load_coco_labels(const std::filesystem::path& label_file_path);
    
        // Normalize image data (in-place)
        void normalize_image(std::span<float> data, size_t channel_size);
    
        // Sigmoid function for logits to probabilities
        [[nodiscard]] float sigmoid(float x) const noexcept;
    
        // ONNX Runtime session and environment
        std::unique_ptr<Ort::Env> env_;
        std::unique_ptr<Ort::Session> session_;
        Ort::AllocatorWithDefaultOptions allocator_;
        Ort::MemoryInfo memory_info_;
    
        // Model parameters
        std::vector<std::string> coco_labels_;
        const Config config_;
        std::vector<int64_t> input_shape_;
        const char* input_name_ = "input";
        std::vector<std::string> output_name_strings_;  // Store actual strings
        std::vector<const char*> output_names_;          // Pointers to strings above
    
        // Helper methods
        cv::Scalar get_color_for_class(int class_id) const noexcept;
    };
    