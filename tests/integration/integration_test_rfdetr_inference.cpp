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

} // namespace

#define SKIP_IF_NO_MODEL(fixture)                                                                        \
    do {                                                                                                 \
        if (!(fixture).model_available_) {                                                               \
            GTEST_SKIP() << "No ONNX model found. Export with rfdetr 1.7.0 or set RFDETR_TEST_MODEL.";  \
        }                                                                                                \
    } while (0)

class RFDETRIntegrationTest : public ::testing::Test {
  protected:
    void SetUp() override {
        // Create a small test image
        cv::Mat test_image(100, 100, CV_8UC3, cv::Scalar(255, 255, 255));
        cv::imwrite("data/test_image.jpg", test_image);

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
    config.resolution = 224; // Use a smaller resolution for testing

    // Create inference object
    RFDETRInference inference(model_path_, label_path_, config);

    // Preprocess
    int orig_h, orig_w;
    auto input_data = inference.preprocess_image(image_path_, orig_h, orig_w);
    EXPECT_EQ(orig_h, 100);
    EXPECT_EQ(orig_w, 100);
    EXPECT_EQ(input_data.size(), 1 * 3 * config.resolution * config.resolution);

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
    cv::Mat image = cv::imread(image_path_.string(), cv::IMREAD_COLOR);
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

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
