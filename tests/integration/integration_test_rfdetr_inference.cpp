#include "rfdetr_inference.hpp"

#include <array>
#include <cstdlib>
#include <fstream>
#include <gtest/gtest.h>

namespace {

std::filesystem::path resolve_test_model_path() {
    if (const char *env_path = std::getenv("RFDETR_TEST_MODEL")) {
        return env_path;
    }

    const std::filesystem::path home = std::getenv("HOME") ? std::getenv("HOME") : "";
    const std::array candidate_paths = {
        home / "Downloads" / "rfdetr-medium.onnx",
        home / "Downloads" / "rfdetr-seg-medium.onnx",
        home / "Downloads" / "inference_model.onnx",
        std::filesystem::path("exports") / "rfdetr-medium.onnx",
        std::filesystem::path("output") / "rfdetr-medium.onnx",
    };

    for (const auto &candidate : candidate_paths) {
        if (!candidate.empty() && std::filesystem::exists(candidate)) {
            return candidate;
        }
    }

    return {};
}

std::filesystem::path resolve_keypoint_model_path() {
    if (const char *env_path = std::getenv("RFDETR_KEYPOINT_MODEL")) {
        return env_path;
    }

    const std::filesystem::path home = std::getenv("HOME") ? std::getenv("HOME") : "";
    const std::array candidate_paths = {
        home / "Downloads" / "rfdetr-keypoint.onnx",
        home / "Downloads" / "rfdetr-keypoint-preview.onnx",
        std::filesystem::path("exports") / "rfdetr-keypoint.onnx",
        std::filesystem::path("exports") / "rfdetr-keypoint-preview.onnx",
        std::filesystem::path("output") / "rfdetr-keypoint.onnx",
        std::filesystem::path("output") / "rfdetr-keypoint-preview.onnx",
    };

    for (const auto &candidate : candidate_paths) {
        if (!candidate.empty() && std::filesystem::exists(candidate)) {
            return candidate;
        }
    }

    return {};
}

void save_white_test_image(const std::filesystem::path &path, int width = 100, int height = 100) {
    rfdetr::media::Image img;
    img.resize(width, height);
    std::fill(img.bgr.begin(), img.bgr.end(), 255);
    if (!rfdetr::media::save_image(img, path)) {
        throw std::runtime_error("Failed to write test image: " + path.string());
    }
}

} // namespace

#define SKIP_IF_NO_MODEL(fixture)                                                                                      \
    do {                                                                                                               \
        if (!(fixture).model_available_) {                                                                             \
            GTEST_SKIP() << "No ONNX model found. Export with rfdetr 1.8.3 or set RFDETR_TEST_MODEL.";                 \
        }                                                                                                              \
    } while (0)

class RFDETRIntegrationTest : public ::testing::Test {
  protected:
    void SetUp() override {
        // Create a small test image
        save_white_test_image("data/test_image.jpg");

        // Create a label file
        std::ofstream label_file("data/test_labels.txt");
        label_file << "person\nbicycle\ncar\nmotorbike\naeroplane\n";
        label_file.close();

        model_path_ = resolve_test_model_path();
        model_available_ = !model_path_.empty();
        image_path_ = "data/test_image.jpg";
        label_path_ = "data/test_labels.txt";
        output_path_ = "data/test_output.jpg";
    }

    void TearDown() override {
        // Clean up test files
        std::filesystem::remove("data/test_image.jpg");
        std::filesystem::remove("data/test_labels.txt");
        if (std::filesystem::exists(output_path_)) {
            std::filesystem::remove(output_path_);
        }
    }

    std::filesystem::path model_path_;
    bool model_available_{false};
    std::filesystem::path image_path_;
    std::filesystem::path label_path_;
    std::filesystem::path output_path_;
};

// Test the full end-to-end pipeline
TEST_F(RFDETRIntegrationTest, EndToEndPipeline) {
    SKIP_IF_NO_MODEL(*this);

    Config config;
    config.resolution = 0; // Use auto-detection

    // Create inference object
    RFDETRInference inference(model_path_, label_path_, config);

    // Preprocess
    int orig_h, orig_w;
    auto input_data = inference.preprocess_image(image_path_, orig_h, orig_w);
    EXPECT_EQ(orig_h, 100);
    EXPECT_EQ(orig_w, 100);
    EXPECT_EQ(input_data.size(), 1 * 3 * inference.get_resolution() * inference.get_resolution());

    // Run inference
    inference.run_inference(input_data);

    // Post-process
    std::vector<float> scores;
    std::vector<int> class_ids;
    std::vector<std::vector<float>> boxes;
    const float scale_w = static_cast<float>(orig_w) / static_cast<float>(inference.get_resolution());
    const float scale_h = static_cast<float>(orig_h) / static_cast<float>(inference.get_resolution());
    inference.postprocess_outputs(scale_w, scale_h, scores, class_ids, boxes);

    // Load image for drawing
    rfdetr::media::Image image = rfdetr::media::load_image(image_path_);
    ASSERT_FALSE(image.empty());

    // Draw detections
    inference.draw_detections(image, boxes, class_ids, scores);

    // Save output
    EXPECT_TRUE(inference.save_output_image(image, output_path_).has_value());
    EXPECT_TRUE(std::filesystem::exists(output_path_));
}

// Test with an invalid model path
TEST_F(RFDETRIntegrationTest, InvalidModelPath) {
    Config config;
    config.resolution = 224;
    const std::filesystem::path invalid_model_path = "invalid_model.onnx";

    EXPECT_THROW(RFDETRInference inference(invalid_model_path, label_path_, config), std::runtime_error);
}

// Test with an empty label file
TEST_F(RFDETRIntegrationTest, EmptyLabelFile) {
    SKIP_IF_NO_MODEL(*this);

    // Create an empty label file
    std::ofstream empty_label_file("data/empty_labels.txt");
    empty_label_file.close();

    Config config;
    config.resolution = 224;

    EXPECT_THROW(RFDETRInference inference(model_path_, "data/empty_labels.txt", config), std::runtime_error);

    std::filesystem::remove("data/empty_labels.txt");
}

// Test with an invalid image path
TEST_F(RFDETRIntegrationTest, InvalidImagePath) {
    SKIP_IF_NO_MODEL(*this);

    Config config;
    config.resolution = 224;
    RFDETRInference inference(model_path_, label_path_, config);

    int orig_h, orig_w;
    const std::filesystem::path invalid_image_path = "invalid_image.jpg";

    EXPECT_THROW(inference.preprocess_image(invalid_image_path, orig_h, orig_w), std::runtime_error);
}

// ============================================================================
// Keypoint Integration Tests
// ============================================================================

class RFDETRKeypointIntegrationTest : public ::testing::Test {
  protected:
    void SetUp() override {
        save_white_test_image("data/test_kp_image.jpg");

        std::ofstream label_file("data/test_labels.txt");
        label_file << "person\nbicycle\ncar\nmotorbike\naeroplane\n";
        label_file.close();

        model_path_ = resolve_keypoint_model_path();
        model_available_ = !model_path_.empty();
        image_path_ = "data/test_kp_image.jpg";
        label_path_ = "data/test_labels.txt";
        output_path_ = "data/test_kp_output.jpg";
    }

    void TearDown() override {
        std::filesystem::remove("data/test_kp_image.jpg");
        if (std::filesystem::exists(output_path_)) {
            std::filesystem::remove(output_path_);
        }
    }

    std::filesystem::path model_path_;
    bool model_available_{false};
    std::filesystem::path image_path_;
    std::filesystem::path label_path_;
    std::filesystem::path output_path_;
};

#define SKIP_IF_NO_KEYPOINT_MODEL(fixture)                                                                             \
    do {                                                                                                               \
        if (!(fixture).model_available_) {                                                                             \
            GTEST_SKIP() << "No keypoint ONNX model found. Export with deploy/export_keypoint.py or set "              \
                            "RFDETR_KEYPOINT_MODEL.";                                                                  \
        }                                                                                                              \
    } while (0)

TEST_F(RFDETRKeypointIntegrationTest, EndToEndKeypointPipeline) {
    SKIP_IF_NO_KEYPOINT_MODEL(*this);

    Config config;
    config.resolution = 0; // Use auto-detection
    config.model_type = ModelType::KEYPOINT;

    RFDETRInference inference(model_path_, label_path_, config);

    int orig_h, orig_w;
    auto input_data = inference.preprocess_image(image_path_, orig_h, orig_w);
    EXPECT_EQ(orig_h, 100);
    EXPECT_EQ(orig_w, 100);
    EXPECT_EQ(input_data.size(), 1 * 3 * inference.get_resolution() * inference.get_resolution());

    inference.run_inference(input_data);

    std::vector<float> scores;
    std::vector<int> class_ids;
    std::vector<std::vector<float>> boxes;
    std::vector<std::vector<KeypointResult>> keypoints;

    const float scale_w = static_cast<float>(orig_w) / static_cast<float>(inference.get_resolution());
    const float scale_h = static_cast<float>(orig_h) / static_cast<float>(inference.get_resolution());

    inference.postprocess_keypoint_outputs(scale_w, scale_h, orig_h, orig_w, scores, class_ids, boxes, keypoints);

    // Should produce at least some detections (the test image is all white, but the model may still fire)
    EXPECT_EQ(scores.size(), class_ids.size());
    EXPECT_EQ(scores.size(), boxes.size());
    EXPECT_EQ(scores.size(), keypoints.size());

    // Draw and save
    rfdetr::media::Image image = rfdetr::media::load_image(image_path_);
    ASSERT_FALSE(image.empty());

    inference.draw_keypoints(image, boxes, class_ids, scores, keypoints);

    EXPECT_TRUE(inference.save_output_image(image, output_path_).has_value());
    EXPECT_TRUE(std::filesystem::exists(output_path_));
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
