#include "mock_backend.hpp"
#include "processing_utils.hpp"
#include "rfdetr_inference.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>

// ============================================================================
// Sigmoid tests
// ============================================================================

TEST(Sigmoid, BasicValues) {
    EXPECT_FLOAT_EQ(rfdetr::processing::sigmoid(0.0f), 0.5f);
    EXPECT_NEAR(rfdetr::processing::sigmoid(100.0f), 1.0f, 1e-6f);
    EXPECT_NEAR(rfdetr::processing::sigmoid(-100.0f), 0.0f, 1e-6f);
}

TEST(Sigmoid, Symmetry) {
    for (float x : {0.5f, 1.0f, 2.0f, 5.0f, 10.0f}) {
        EXPECT_NEAR(rfdetr::processing::sigmoid(x) + rfdetr::processing::sigmoid(-x), 1.0f, 1e-6f);
    }
}

// ============================================================================
// NormalizeImage tests
// ============================================================================

TEST(NormalizeImage, AppliesMeanStd) {
    // 1 pixel, 3 channels: values = [0.5, 0.5, 0.5]
    std::vector<float> data = {0.5f, 0.5f, 0.5f};
    std::array<float, 3> means = {0.485f, 0.456f, 0.406f};
    std::array<float, 3> stds = {0.229f, 0.224f, 0.225f};

    rfdetr::processing::normalize_image(data, 1, means, stds);

    EXPECT_NEAR(data[0], (0.5f - 0.485f) / 0.229f, 1e-5f);
    EXPECT_NEAR(data[1], (0.5f - 0.456f) / 0.224f, 1e-5f);
    EXPECT_NEAR(data[2], (0.5f - 0.406f) / 0.225f, 1e-5f);
}

TEST(NormalizeImage, AllChannels) {
    // 2 pixels per channel, 3 channels = 6 floats
    std::vector<float> data = {0.1f, 0.2f,  // channel 0
                               0.3f, 0.4f,  // channel 1
                               0.5f, 0.6f}; // channel 2
    std::array<float, 3> means = {0.0f, 0.0f, 0.0f};
    std::array<float, 3> stds = {0.5f, 0.25f, 0.1f};

    rfdetr::processing::normalize_image(data, 2, means, stds);

    // Channel 0: divide by 0.5
    EXPECT_NEAR(data[0], 0.2f, 1e-5f);
    EXPECT_NEAR(data[1], 0.4f, 1e-5f);
    // Channel 1: divide by 0.25
    EXPECT_NEAR(data[2], 1.2f, 1e-5f);
    EXPECT_NEAR(data[3], 1.6f, 1e-5f);
    // Channel 2: divide by 0.1
    EXPECT_NEAR(data[4], 5.0f, 1e-5f);
    EXPECT_NEAR(data[5], 6.0f, 1e-5f);
}

// ============================================================================
// CxCyWhToXyxy tests
// ============================================================================

TEST(CxCyWhToXyxy, BasicConversion) {
    auto box = rfdetr::processing::cxcywh_to_xyxy(50.0f, 50.0f, 20.0f, 10.0f);
    EXPECT_FLOAT_EQ(box.x_min, 40.0f);
    EXPECT_FLOAT_EQ(box.y_min, 45.0f);
    EXPECT_FLOAT_EQ(box.x_max, 60.0f);
    EXPECT_FLOAT_EQ(box.y_max, 55.0f);
}

TEST(CxCyWhToXyxy, ZeroSize) {
    auto box = rfdetr::processing::cxcywh_to_xyxy(10.0f, 20.0f, 0.0f, 0.0f);
    EXPECT_FLOAT_EQ(box.x_min, 10.0f);
    EXPECT_FLOAT_EQ(box.y_min, 20.0f);
    EXPECT_FLOAT_EQ(box.x_max, 10.0f);
    EXPECT_FLOAT_EQ(box.y_max, 20.0f);
}

// ============================================================================
// ScaleBox tests
// ============================================================================

TEST(ScaleBox, Scaling) {
    rfdetr::processing::BoundingBox box{10.0f, 20.0f, 30.0f, 40.0f};
    auto scaled = rfdetr::processing::scale_box(box, 2.0f, 0.5f);
    EXPECT_FLOAT_EQ(scaled.x_min, 20.0f);
    EXPECT_FLOAT_EQ(scaled.y_min, 10.0f);
    EXPECT_FLOAT_EQ(scaled.x_max, 60.0f);
    EXPECT_FLOAT_EQ(scaled.y_max, 20.0f);
}

// ============================================================================
// GetColorForClass tests
// ============================================================================

TEST(GetColorForClass, Deterministic) {
    auto c1 = rfdetr::processing::get_color_for_class(5);
    auto c2 = rfdetr::processing::get_color_for_class(5);
    EXPECT_EQ(c1, c2);

    // Different classes should (very likely) give different colors
    auto c3 = rfdetr::processing::get_color_for_class(0);
    auto c4 = rfdetr::processing::get_color_for_class(1);
    EXPECT_NE(c3, c4);
}

// ============================================================================
// Helper: create a temporary label file
// ============================================================================

class TempLabelFile {
  public:
    explicit TempLabelFile(const std::string &content, const std::string &name = "test_labels.txt")
        : path_(std::filesystem::temp_directory_path() / name) {
        std::ofstream f(path_);
        f << content;
    }
    ~TempLabelFile() { std::filesystem::remove(path_); }
    [[nodiscard]] const std::filesystem::path &path() const { return path_; }

  private:
    std::filesystem::path path_;
};

// ============================================================================
// Label loading tests
// ============================================================================

TEST(LabelLoading, ValidFile) {
    TempLabelFile labels("person\nbicycle\ncar\n");
    Config config;
    config.resolution = 560;

    auto backend = std::make_unique<MockBackend>();
    backend->set_outputs({{}, {}}, {{1, 1, 4}, {1, 1, 4}});

    RFDETRInference inference(std::move(backend), labels.path(), config);
    const auto &loaded = inference.get_coco_labels();
    ASSERT_EQ(loaded.size(), 3u);
    EXPECT_EQ(loaded[0], "person");
    EXPECT_EQ(loaded[1], "bicycle");
    EXPECT_EQ(loaded[2], "car");
}

TEST(LabelLoading, EmptyFile) {
    TempLabelFile labels("");
    Config config;

    auto backend = std::make_unique<MockBackend>();
    EXPECT_THROW(RFDETRInference(std::move(backend), labels.path(), config), std::runtime_error);
}

TEST(LabelLoading, MissingFile) {
    Config config;
    auto backend = std::make_unique<MockBackend>();
    EXPECT_THROW(RFDETRInference(std::move(backend), "/nonexistent/labels.txt", config), std::runtime_error);
}

// ============================================================================
// Preprocess tests
// ============================================================================

TEST(Preprocess, OutputDimensions) {
    // Create a temporary test image
    auto tmp_img = std::filesystem::temp_directory_path() / "test_preprocess.jpg";
    cv::Mat img(100, 200, CV_8UC3, cv::Scalar(128, 128, 128));
    cv::imwrite(tmp_img.string(), img);

    TempLabelFile labels("person\ncar\n");
    Config config;
    config.resolution = 224;

    auto backend = std::make_unique<MockBackend>();
    backend->set_outputs({{}, {}}, {{1, 1, 4}, {1, 1, 3}});

    RFDETRInference inference(std::move(backend), labels.path(), config);

    int orig_h = 0;
    int orig_w = 0;
    auto data = inference.preprocess_image(tmp_img, orig_h, orig_w);

    EXPECT_EQ(orig_h, 100);
    EXPECT_EQ(orig_w, 200);
    EXPECT_EQ(data.size(), static_cast<size_t>(3 * 224 * 224));

    std::filesystem::remove(tmp_img);
}

TEST(Preprocess, InvalidImage) {
    TempLabelFile labels("person\ncar\n");
    Config config;
    config.resolution = 224;

    auto backend = std::make_unique<MockBackend>();
    backend->set_outputs({{}, {}}, {{1, 1, 4}, {1, 1, 3}});

    RFDETRInference inference(std::move(backend), labels.path(), config);

    int orig_h = 0;
    int orig_w = 0;
    EXPECT_THROW(inference.preprocess_image("/nonexistent/image.jpg", orig_h, orig_w), std::runtime_error);
}

// ============================================================================
// Postprocess tests (using MockBackend)
// ============================================================================

class PostprocessTest : public ::testing::Test {
  protected:
    void SetUp() override {
        labels_file_ = std::make_unique<TempLabelFile>("person\nbicycle\ncar\nmotorbike\naeroplane\n");
    }

    // Create a mock-backed inference instance with given output tensors
    std::unique_ptr<RFDETRInference> make_inference(std::vector<std::vector<float>> output_data,
                                                    std::vector<std::vector<int64_t>> output_shapes,
                                                    float threshold = 0.5f, int resolution = 560) {
        Config config;
        config.resolution = resolution;
        config.threshold = threshold;

        auto backend = std::make_unique<MockBackend>();
        backend->set_outputs(std::move(output_data), std::move(output_shapes));

        auto inference = std::make_unique<RFDETRInference>(std::move(backend), labels_file_->path(), config);

        // Simulate run_inference by feeding dummy input to populate the output cache
        const auto res = static_cast<size_t>(resolution);
        std::vector<float> dummy_input(3 * res * res, 0.0f);
        inference->run_inference(dummy_input);

        return inference;
    }

    std::unique_ptr<TempLabelFile> labels_file_;
};

TEST_F(PostprocessTest, ThresholdFiltering) {
    // 2 detections, 6 classes (5 real + 1 background at index 0)
    // Detection 0: high score at class index 1 (maps to class 0 = "person")
    // Detection 1: low score everywhere
    const int num_dets = 2;
    const int num_classes = 6;

    // Boxes: [batch=1, num_dets=2, coords=4] — normalized cxcywh
    std::vector<float> dets_data = {
        0.5f, 0.5f, 0.2f, 0.1f, // det 0: center=(0.5, 0.5), size=(0.2, 0.1)
        0.3f, 0.3f, 0.1f, 0.1f, // det 1: center=(0.3, 0.3), size=(0.1, 0.1)
    };

    // Labels: [batch=1, num_dets=2, num_classes=6] — logits
    // sigmoid(5.0) ≈ 0.993, sigmoid(-5.0) ≈ 0.007
    std::vector<float> labels_data(static_cast<size_t>(num_dets * num_classes), -5.0f);
    labels_data[1] = 5.0f; // det 0, class index 1 → high score, class_id = 0 ("person")

    auto inference =
        make_inference({dets_data, labels_data}, {{1, num_dets, 4}, {1, num_dets, num_classes}}, 0.5f, 560);

    std::vector<float> scores;
    std::vector<int> class_ids;
    std::vector<std::vector<float>> boxes;
    inference->postprocess_outputs(1.0f, 1.0f, scores, class_ids, boxes);

    // Only detection 0 should pass the threshold
    ASSERT_EQ(scores.size(), 1u);
    EXPECT_GT(scores[0], 0.9f);
    EXPECT_EQ(class_ids[0], 0); // "person"
}

TEST_F(PostprocessTest, CoordinateConversion) {
    const int num_dets = 1;
    const int num_classes = 6;
    const int resolution = 100; // use 100 for easy math

    // Box at center (0.5, 0.5) with size (0.2, 0.1) in normalized coords
    // After * resolution: cx=50, cy=50, w=20, h=10
    // xyxy: (40, 45, 60, 55)
    std::vector<float> dets_data = {0.5f, 0.5f, 0.2f, 0.1f};

    std::vector<float> labels_data(static_cast<size_t>(num_classes), -10.0f);
    labels_data[1] = 10.0f; // high score at class index 1

    auto inference =
        make_inference({dets_data, labels_data}, {{1, num_dets, 4}, {1, num_dets, num_classes}}, 0.5f, resolution);

    std::vector<float> scores;
    std::vector<int> class_ids;
    std::vector<std::vector<float>> boxes;
    inference->postprocess_outputs(1.0f, 1.0f, scores, class_ids, boxes);

    ASSERT_EQ(boxes.size(), 1u);
    EXPECT_NEAR(boxes[0][0], 40.0f, 0.01f); // x_min
    EXPECT_NEAR(boxes[0][1], 45.0f, 0.01f); // y_min
    EXPECT_NEAR(boxes[0][2], 60.0f, 0.01f); // x_max
    EXPECT_NEAR(boxes[0][3], 55.0f, 0.01f); // y_max
}

TEST_F(PostprocessTest, ClassIdOffset) {
    const int num_dets = 1;
    const int num_classes = 6;

    std::vector<float> dets_data = {0.5f, 0.5f, 0.2f, 0.1f};

    // Put high score at class index 3 → class_id should be 2 ("car")
    std::vector<float> labels_data(static_cast<size_t>(num_classes), -10.0f);
    labels_data[3] = 10.0f;

    auto inference =
        make_inference({dets_data, labels_data}, {{1, num_dets, 4}, {1, num_dets, num_classes}}, 0.5f, 560);

    std::vector<float> scores;
    std::vector<int> class_ids;
    std::vector<std::vector<float>> boxes;
    inference->postprocess_outputs(1.0f, 1.0f, scores, class_ids, boxes);

    ASSERT_EQ(class_ids.size(), 1u);
    EXPECT_EQ(class_ids[0], 2); // index 3 - 1 = 2 → "car"
}

TEST_F(PostprocessTest, EmptyResults) {
    const int num_dets = 3;
    const int num_classes = 6;

    std::vector<float> dets_data(static_cast<size_t>(num_dets * 4), 0.5f);
    // All logits very negative → all sigmoid scores ≈ 0
    std::vector<float> labels_data(static_cast<size_t>(num_dets * num_classes), -20.0f);

    auto inference =
        make_inference({dets_data, labels_data}, {{1, num_dets, 4}, {1, num_dets, num_classes}}, 0.5f, 560);

    std::vector<float> scores;
    std::vector<int> class_ids;
    std::vector<std::vector<float>> boxes;
    inference->postprocess_outputs(1.0f, 1.0f, scores, class_ids, boxes);

    EXPECT_TRUE(scores.empty());
    EXPECT_TRUE(class_ids.empty());
    EXPECT_TRUE(boxes.empty());
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
