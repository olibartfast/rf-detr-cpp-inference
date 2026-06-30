#include "mock_backend.hpp"
#include "processing_utils.hpp"
#include "rfdetr_inference.hpp"
#include "video_pipeline.hpp"

#include <atomic>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <thread>

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
// ClampBox tests
// ============================================================================

TEST(ClampBox, PreservesInBoundsBox) {
    rfdetr::processing::BoundingBox box{10.0f, 20.0f, 30.0f, 40.0f};
    auto clamped = rfdetr::processing::clamp_box(box, 100.0f, 100.0f);
    EXPECT_FLOAT_EQ(clamped.x_min, 10.0f);
    EXPECT_FLOAT_EQ(clamped.y_min, 20.0f);
    EXPECT_FLOAT_EQ(clamped.x_max, 30.0f);
    EXPECT_FLOAT_EQ(clamped.y_max, 40.0f);
}

TEST(ClampBox, ClampsNegativeToZero) {
    rfdetr::processing::BoundingBox box{-5.0f, -10.0f, 30.0f, 40.0f};
    auto clamped = rfdetr::processing::clamp_box(box, 100.0f, 100.0f);
    EXPECT_FLOAT_EQ(clamped.x_min, 0.0f);
    EXPECT_FLOAT_EQ(clamped.y_min, 0.0f);
    EXPECT_FLOAT_EQ(clamped.x_max, 30.0f);
    EXPECT_FLOAT_EQ(clamped.y_max, 40.0f);
}

TEST(ClampBox, ClampsOverflowToMax) {
    rfdetr::processing::BoundingBox box{10.0f, 20.0f, 150.0f, 200.0f};
    auto clamped = rfdetr::processing::clamp_box(box, 100.0f, 120.0f);
    EXPECT_FLOAT_EQ(clamped.x_min, 10.0f);
    EXPECT_FLOAT_EQ(clamped.y_min, 20.0f);
    EXPECT_FLOAT_EQ(clamped.x_max, 100.0f);
    EXPECT_FLOAT_EQ(clamped.y_max, 120.0f);
}

TEST(ClampBox, ClampsWidthAndHeightIndependently) {
    rfdetr::processing::BoundingBox box{-1.0f, -1.0f, 200.0f, 80.0f};
    auto clamped = rfdetr::processing::clamp_box(box, 160.0f, 90.0f);
    EXPECT_FLOAT_EQ(clamped.x_min, 0.0f);
    EXPECT_FLOAT_EQ(clamped.y_min, 0.0f);
    EXPECT_FLOAT_EQ(clamped.x_max, 160.0f);
    EXPECT_FLOAT_EQ(clamped.y_max, 80.0f);
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

TEST_F(PostprocessTest, BoxesClampedToImageBounds) {
    // resolution=100, scale_w=scale_h=1.0 -> clamp bounds = [0, 100] x [0, 100]
    const int num_dets = 2;
    const int num_classes = 6;
    const int resolution = 100;

    // det 0: center (0.05, 0.05), size (0.3, 0.3) -> xyxy=(-10,-10,20,20) before clamp
    // det 1: center (0.95, 0.95), size (0.3, 0.3) -> xyxy=(80,80,110,110) before clamp
    std::vector<float> dets_data = {
        0.05f, 0.05f, 0.3f, 0.3f, 0.95f, 0.95f, 0.3f, 0.3f,
    };

    std::vector<float> labels_data(static_cast<size_t>(num_dets * num_classes), -10.0f);
    labels_data[1] = 10.0f;               // det 0, class index 1
    labels_data[1 + num_classes] = 10.0f; // det 1, class index 1

    auto inference =
        make_inference({dets_data, labels_data}, {{1, num_dets, 4}, {1, num_dets, num_classes}}, 0.5f, resolution);

    std::vector<float> scores;
    std::vector<int> class_ids;
    std::vector<std::vector<float>> boxes;
    inference->postprocess_outputs(1.0f, 1.0f, scores, class_ids, boxes);

    ASSERT_EQ(boxes.size(), 2u);
    // det 0: negative x_min/y_min clamped to 0
    EXPECT_NEAR(boxes[0][0], 0.0f, 0.01f);
    EXPECT_NEAR(boxes[0][1], 0.0f, 0.01f);
    EXPECT_NEAR(boxes[0][2], 20.0f, 0.01f);
    EXPECT_NEAR(boxes[0][3], 20.0f, 0.01f);
    // det 1: overflowing x_max/y_max clamped to 100
    EXPECT_NEAR(boxes[1][0], 80.0f, 0.01f);
    EXPECT_NEAR(boxes[1][1], 80.0f, 0.01f);
    EXPECT_NEAR(boxes[1][2], 100.0f, 0.01f);
    EXPECT_NEAR(boxes[1][3], 100.0f, 0.01f);
}

// ============================================================================
// PreprocessFrame free function tests
// ============================================================================

TEST(PreprocessFrame, OutputDimensions) {
    cv::Mat img(100, 200, CV_8UC3, cv::Scalar(128, 128, 128));
    const int res = 224;
    std::vector<float> tensor(3 * 224 * 224);
    std::array<float, 3> means = {0.485f, 0.456f, 0.406f};
    std::array<float, 3> stds = {0.229f, 0.224f, 0.225f};

    rfdetr::processing::preprocess_frame(img, tensor, res, means, stds);

    for (float v : tensor) {
        EXPECT_TRUE(std::isfinite(v));
    }
}

// ============================================================================
// cv::Mat preprocess overload tests
// ============================================================================

TEST(Preprocess, MatOverload) {
    cv::Mat img(100, 200, CV_8UC3, cv::Scalar(128, 128, 128));
    TempLabelFile labels("person\ncar\n");
    Config config;
    config.resolution = 224;

    auto backend = std::make_unique<MockBackend>();
    backend->set_outputs({{}, {}}, {{1, 1, 4}, {1, 1, 3}});
    RFDETRInference inference(std::move(backend), labels.path(), config);

    int orig_h = 0;
    int orig_w = 0;
    auto data = inference.preprocess_image(img, orig_h, orig_w);

    EXPECT_EQ(orig_h, 100);
    EXPECT_EQ(orig_w, 200);
    EXPECT_EQ(data.size(), static_cast<size_t>(3 * 224 * 224));
}

TEST(Preprocess, MatOverloadEmptyImage) {
    TempLabelFile labels("person\ncar\n");
    Config config;
    config.resolution = 224;

    auto backend = std::make_unique<MockBackend>();
    backend->set_outputs({{}, {}}, {{1, 1, 4}, {1, 1, 3}});
    RFDETRInference inference(std::move(backend), labels.path(), config);

    cv::Mat empty;
    int orig_h = 0;
    int orig_w = 0;
    EXPECT_THROW(inference.preprocess_image(empty, orig_h, orig_w), std::runtime_error);
}

// ============================================================================
// BoundedQueue tests
// ============================================================================

TEST(BoundedQueue, BasicPushPop) {
    rfdetr::video::BoundedQueue<size_t> q(4);
    q.push(42);
    EXPECT_EQ(q.pop(), 42u);
}

TEST(BoundedQueue, FIFO) {
    rfdetr::video::BoundedQueue<size_t> q(4);
    q.push(1);
    q.push(2);
    q.push(3);
    EXPECT_EQ(q.pop(), 1u);
    EXPECT_EQ(q.pop(), 2u);
    EXPECT_EQ(q.pop(), 3u);
}

TEST(BoundedQueue, BlocksWhenFull) {
    rfdetr::video::BoundedQueue<size_t> q(2);
    q.push(1);
    q.push(2);
    // Queue is full. Push from another thread should block until we pop.
    std::atomic<bool> pushed{false};
    std::thread t([&] {
        q.push(3);
        pushed.store(true);
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    EXPECT_FALSE(pushed.load());
    q.pop(); // unblocks the push
    t.join();
    EXPECT_TRUE(pushed.load());
}

TEST(BoundedQueue, PoisonPill) {
    rfdetr::video::BoundedQueue<size_t> q(4);
    q.push(rfdetr::video::kPoisonPill);
    EXPECT_EQ(q.pop(), rfdetr::video::kPoisonPill);
}

// ============================================================================
// Keypoint postprocessing tests
// ============================================================================

class KeypointPostprocessTest : public ::testing::Test {
  protected:
    void SetUp() override { labels_file_ = std::make_unique<TempLabelFile>("person\nbicycle\ncar\n"); }

    std::unique_ptr<RFDETRInference> make_inference(std::vector<std::vector<float>> output_data,
                                                    std::vector<std::vector<int64_t>> output_shapes,
                                                    float threshold = 0.5f, int resolution = 560) {
        Config config;
        config.resolution = resolution;
        config.threshold = threshold;
        config.model_type = ModelType::KEYPOINT;

        // RFDETRKeypointPreview: background has 0 keypoints, person has 17 COCO keypoints.
        config.keypoint_counts = {0, 17};

        auto backend = std::make_unique<MockBackend>();
        backend->set_outputs(std::move(output_data), std::move(output_shapes));

        auto inference = std::make_unique<RFDETRInference>(std::move(backend), labels_file_->path(), config);

        const auto res = static_cast<size_t>(resolution);
        std::vector<float> dummy_input(3 * res * res, 0.0f);
        inference->run_inference(dummy_input);

        return inference;
    }

    std::unique_ptr<TempLabelFile> labels_file_;
};

TEST_F(KeypointPostprocessTest, ThreeOutputsRequired) {
    // Only 2 outputs should fail validation for KEYPOINT
    const int num_dets = 1;
    const int num_classes = 92; // background + 91 COCO classes

    std::vector<float> dets_data(static_cast<size_t>(num_dets * 4), 0.5f);
    std::vector<float> labels_data(static_cast<size_t>(num_dets * num_classes), -10.0f);

    auto inference =
        make_inference({dets_data, labels_data}, {{1, num_dets, 4}, {1, num_dets, num_classes}}, 0.5f, 560);

    std::vector<float> scores;
    std::vector<int> class_ids;
    std::vector<std::vector<float>> boxes;
    std::vector<std::vector<KeypointResult>> keypoints;

    // With MockBackend, run_inference only caches what was set — so 2 outputs won't throw
    // But postprocess_keypoint_outputs should throw for < 3 outputs
    EXPECT_THROW(inference->postprocess_keypoint_outputs(1.0f, 1.0f, 100, 200, scores, class_ids, boxes, keypoints),
                 std::runtime_error);
}

TEST_F(KeypointPostprocessTest, ClassSelectionAndBboxDecode) {
    // 1 query, 92 classes (background + 91 COCO classes), keypoints shape [1, 1, 34, 8]
    // 34 = 2 keypoint classes * 17 padded slots
    const int num_dets = 1;
    const int num_classes = 92;
    const int num_kp_channels = 272;

    // Detection at center (0.5, 0.5), size (0.2, 0.1), resolution=100
    std::vector<float> dets_data = {0.5f, 0.5f, 0.2f, 0.1f};

    // Labels: high score at class index 1 (person), low elsewhere
    std::vector<float> labels_data(static_cast<size_t>(num_dets * num_classes), -10.0f);
    labels_data[1] = 10.0f; // class index 1 → class_id 0 = "person"

    // Keypoints: [batch=1, num_dets=1, slots=34] = 2 keypoint classes * 17 padded slots
    std::vector<float> kp_data(static_cast<size_t>(num_dets * num_kp_channels), 0.0f);
    // Set first keypoint at normalized image coordinate (0.25, 0.5) -> (50, 50)
    // orig_w=200, orig_h=100
    kp_data[136] = 0.25f; // normalized x
    kp_data[137] = 0.5f;  // normalized y
    kp_data[138] = 10.0f; // findability logit → sigmoid ≈ 1.0
    kp_data[139] = 10.0f; // visibility logit → sigmoid ≈ 1.0

    auto inference = make_inference({dets_data, labels_data, kp_data},
                                    {{1, num_dets, 4}, {1, num_dets, num_classes}, {1, num_dets, 34, 8}}, 0.5f, 100);

    std::vector<float> scores;
    std::vector<int> class_ids;
    std::vector<std::vector<float>> boxes;
    std::vector<std::vector<KeypointResult>> keypoints;

    inference->postprocess_keypoint_outputs(1.0f, 1.0f, 100, 200, scores, class_ids, boxes, keypoints);

    ASSERT_EQ(scores.size(), 1u);
    EXPECT_EQ(class_ids[0], 0); // "person"
    EXPECT_GT(scores[0], 0.0f); // Still positive after uncertainty fusion

    // Bbox: cx=50, cy=50, w=20, h=10 → xyxy=(40, 45, 60, 55), scale=1.0
    ASSERT_EQ(boxes.size(), 1u);
    EXPECT_NEAR(boxes[0][0], 40.0f, 0.01f);
    EXPECT_NEAR(boxes[0][1], 45.0f, 0.01f);
    EXPECT_NEAR(boxes[0][2], 60.0f, 0.01f);
    EXPECT_NEAR(boxes[0][3], 55.0f, 0.01f);

    // Keypoints
    ASSERT_EQ(keypoints.size(), 1u);
    ASSERT_EQ(keypoints[0].size(), 17u); // COCO person keypoints

    // First keypoint: x=0.25*200=50, y=0.5*100=50
    EXPECT_NEAR(keypoints[0][0].x, 50.0f, 0.01f);
    EXPECT_NEAR(keypoints[0][0].y, 50.0f, 0.01f);
    EXPECT_NEAR(keypoints[0][0].findability, 1.0f, 1e-4f);
    EXPECT_NEAR(keypoints[0][0].visibility, 1.0f, 1e-4f);
}

TEST_F(KeypointPostprocessTest, KeypointCoordinateDecode) {
    // One query, class 1 (person), normalized image-relative coordinate
    const int num_dets = 1;
    const int num_classes = 92;

    // Box at normalized (0.3, 0.4) with size (0.4, 0.2), resolution=100
    // cx=30, cy=40, w=40, h=20
    std::vector<float> dets_data = {0.3f, 0.4f, 0.4f, 0.2f};

    std::vector<float> labels_data(static_cast<size_t>(num_dets * num_classes), -10.0f);
    labels_data[1] = 10.0f;

    // Keypoint at normalized image coordinate (0.2, 0.3)
    // kp_x = 0.2 * 200 = 40, kp_y = 0.3 * 100 = 30
    std::vector<float> kp_data(static_cast<size_t>(num_dets * 272), 0.0f);
    kp_data[136] = 0.2f;
    kp_data[137] = 0.3f;
    kp_data[138] = 5.0f; // findability logit
    kp_data[139] = 5.0f; // visibility logit

    auto inference = make_inference({dets_data, labels_data, kp_data},
                                    {{1, num_dets, 4}, {1, num_dets, num_classes}, {1, num_dets, 34, 8}}, 0.5f, 100);

    std::vector<float> scores;
    std::vector<int> class_ids;
    std::vector<std::vector<float>> boxes;
    std::vector<std::vector<KeypointResult>> keypoints;

    inference->postprocess_keypoint_outputs(1.0f, 1.0f, 100, 200, scores, class_ids, boxes, keypoints);

    ASSERT_EQ(keypoints.size(), 1u);
    ASSERT_GE(keypoints[0].size(), 1u);
    EXPECT_NEAR(keypoints[0][0].x, 40.0f, 0.01f);
    EXPECT_NEAR(keypoints[0][0].y, 30.0f, 0.01f);
}

TEST_F(KeypointPostprocessTest, ScaleApplied) {
    // Test that scale_w/scale_h are applied properly
    const int num_dets = 1;
    const int num_classes = 92;

    // Box at (0.5, 0.5), size (0.2, 0.1), res=100 → cx=50, cy=50, w=20, h=10
    std::vector<float> dets_data = {0.5f, 0.5f, 0.2f, 0.1f};

    std::vector<float> labels_data(static_cast<size_t>(num_dets * num_classes), -10.0f);
    labels_data[1] = 10.0f;

    // Keypoint at normalized image coordinate (0.5, 0.5). scale_w/scale_h are ignored for image-relative keypoints.
    std::vector<float> kp_data(static_cast<size_t>(num_dets * 272), 0.0f);
    kp_data[136] = 0.5f;
    kp_data[137] = 0.5f;
    kp_data[138] = 5.0f;
    kp_data[139] = 5.0f;

    auto inference = make_inference({dets_data, labels_data, kp_data},
                                    {{1, num_dets, 4}, {1, num_dets, num_classes}, {1, num_dets, 34, 8}}, 0.5f, 100);

    std::vector<float> scores;
    std::vector<int> class_ids;
    std::vector<std::vector<float>> boxes;
    std::vector<std::vector<KeypointResult>> keypoints;

    // scale_w=2.0, scale_h=3.0, orig image size 200x300
    inference->postprocess_keypoint_outputs(2.0f, 3.0f, 300, 200, scores, class_ids, boxes, keypoints);

    ASSERT_GE(keypoints.size(), 1u);
    ASSERT_GE(keypoints[0].size(), 1u);
    EXPECT_NEAR(keypoints[0][0].x, 100.0f, 0.01f); // 0.5 * 200
    EXPECT_NEAR(keypoints[0][0].y, 150.0f, 0.01f); // 0.5 * 300
}

TEST_F(KeypointPostprocessTest, NoDetectionsBelowThreshold) {
    const int num_dets = 1;
    const int num_classes = 92;

    std::vector<float> dets_data = {0.5f, 0.5f, 0.2f, 0.1f};
    std::vector<float> labels_data(static_cast<size_t>(num_dets * num_classes), -20.0f); // all low
    std::vector<float> kp_data(static_cast<size_t>(num_dets * 272), 0.0f);

    auto inference = make_inference({dets_data, labels_data, kp_data},
                                    {{1, num_dets, 4}, {1, num_dets, num_classes}, {1, num_dets, 34, 8}}, 0.5f, 100);

    std::vector<float> scores;
    std::vector<int> class_ids;
    std::vector<std::vector<float>> boxes;
    std::vector<std::vector<KeypointResult>> keypoints;

    inference->postprocess_keypoint_outputs(1.0f, 1.0f, 100, 200, scores, class_ids, boxes, keypoints);

    EXPECT_TRUE(scores.empty());
    EXPECT_TRUE(class_ids.empty());
    EXPECT_TRUE(boxes.empty());
    EXPECT_TRUE(keypoints.empty());
}

TEST_F(KeypointPostprocessTest, CholeskyToCovariance) {
    // Test precision Cholesky → pixel covariance math
    const int num_dets = 1;
    const int num_classes = 92;

    std::vector<float> dets_data = {0.5f, 0.5f, 0.2f, 0.1f};

    std::vector<float> labels_data(static_cast<size_t>(num_dets * num_classes), -10.0f);
    labels_data[1] = 10.0f;

    // Set Cholesky params: log_l11 = log(2.0), l21 = 0.5, log_l22 = log(3.0)
    // L = [[2, 0], [0.5, 3]]
    // precision = L @ L^T = [[4, 1], [1, 9.25]]
    // cov = inv(precision) = 1/(4*9.25 - 1) * [[9.25, -1], [-1, 4]]
    //   = 1/36 * [[9.25, -1], [-1, 4]] = [[0.2569..., -0.02778...], [-0.02778..., 0.1111...]]
    // Scale by pixel_scale = img_w * img_h
    std::vector<float> kp_data(static_cast<size_t>(num_dets * 272), 0.0f);
    kp_data[136] = 0.25f;
    kp_data[137] = 0.5f;
    kp_data[138] = 5.0f;
    kp_data[139] = 5.0f;
    kp_data[140] = std::log(2.0f); // log_l11
    kp_data[141] = 0.5f;           // l21
    kp_data[142] = std::log(3.0f); // log_l22

    auto inference = make_inference({dets_data, labels_data, kp_data},
                                    {{1, num_dets, 4}, {1, num_dets, num_classes}, {1, num_dets, 34, 8}}, 0.5f, 100);

    std::vector<float> scores;
    std::vector<int> class_ids;
    std::vector<std::vector<float>> boxes;
    std::vector<std::vector<KeypointResult>> keypoints;

    // orig_w=200, orig_h=100
    inference->postprocess_keypoint_outputs(1.0f, 1.0f, 100, 200, scores, class_ids, boxes, keypoints);

    ASSERT_GE(keypoints.size(), 1u);
    ASSERT_GE(keypoints[0].size(), 1u);

    const auto &kpr = keypoints[0][0];
    const float width = 200.0f;
    const float height = 100.0f;

    // Expected covariance (normalized): [[9.25, -1], [-1, 4]] / 36
    // scaled by diag(width, height) on both sides
    const float det = 4.0f * 9.25f - 1.0f;
    const float inv_det = 1.0f / det;
    const float expected_cov00 = inv_det * 9.25f * width * width;
    const float expected_cov01 = inv_det * (-1.0f) * width * height;
    const float expected_cov11 = inv_det * 4.0f * height * height;

    EXPECT_NEAR(kpr.cov[0], expected_cov00, expected_cov00 * 1e-5f);
    EXPECT_NEAR(kpr.cov[1], expected_cov01, std::abs(expected_cov01) * 1e-5f);
    EXPECT_NEAR(kpr.cov[2], expected_cov01, std::abs(expected_cov01) * 1e-5f); // symmetric
    EXPECT_NEAR(kpr.cov[3], expected_cov11, expected_cov11 * 1e-5f);
}

TEST_F(KeypointPostprocessTest, BackgroundColumnIgnored) {
    // Logit column 0 is background; column 1 maps to the first label.
    const int num_dets = 1;
    const int num_classes = 4;

    std::vector<float> dets_data = {0.5f, 0.5f, 0.2f, 0.1f};

    // High score only in the background column should be ignored.
    std::vector<float> labels_data = {10.0f, -10.0f, -10.0f, -10.0f};

    std::vector<float> kp_data(static_cast<size_t>(num_dets * 272), 0.0f);

    auto inference = make_inference({dets_data, labels_data, kp_data},
                                    {{1, num_dets, 4}, {1, num_dets, num_classes}, {1, num_dets, 34, 8}}, 0.5f, 100);

    std::vector<float> scores;
    std::vector<int> class_ids;
    std::vector<std::vector<float>> boxes;
    std::vector<std::vector<KeypointResult>> keypoints;

    inference->postprocess_keypoint_outputs(1.0f, 1.0f, 100, 200, scores, class_ids, boxes, keypoints);

    // Background maps to class_id -1 and is skipped.
    EXPECT_TRUE(scores.empty());
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
