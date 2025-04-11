#include <gtest/gtest.h>
#include "rfdetr_inference.hpp"
#include <fstream>

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

        // Setup paths
        model_path_ = "/home/oli/Downloads/inference_model.onnx";
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
    std::filesystem::path image_path_;
    std::filesystem::path label_path_;
    std::filesystem::path output_path_;
};

// Test the full end-to-end pipeline
TEST_F(RFDETRIntegrationTest, EndToEndPipeline) {
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
    auto output_tensors = inference.run_inference(input_data);
    EXPECT_EQ(output_tensors.size(), 2);

    // Post-process
    std::vector<float> scores;
    std::vector<int> class_ids;
    std::vector<std::vector<float>> boxes;
    const float scale_w = static_cast<float>(orig_w) / inference.get_resolution();
    const float scale_h = static_cast<float>(orig_h) / inference.get_resolution();
    inference.postprocess_outputs(output_tensors, scale_w, scale_h, scores, class_ids, boxes);

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

    EXPECT_THROW(
        RFDETRInference inference(invalid_model_path, label_path_, config),
        std::runtime_error
    );
}

// Test with an empty label file
TEST_F(RFDETRIntegrationTest, EmptyLabelFile) {
    // Create an empty label file
    std::ofstream empty_label_file("data/empty_labels.txt");
    empty_label_file.close();

    Config config;
    config.resolution = 224;

    EXPECT_THROW(
        RFDETRInference inference(model_path_, "data/empty_labels.txt", config),
        std::runtime_error
    );

    std::filesystem::remove("data/empty_labels.txt");
}

// Test with an invalid image path
TEST_F(RFDETRIntegrationTest, InvalidImagePath) {
    Config config;
    config.resolution = 224;
    RFDETRInference inference(model_path_, label_path_, config);

    int orig_h, orig_w;
    const std::filesystem::path invalid_image_path = "invalid_image.jpg";

    EXPECT_THROW(
        inference.preprocess_image(invalid_image_path, orig_h, orig_w),
        std::runtime_error
    );
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}