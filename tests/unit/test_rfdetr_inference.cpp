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
#include <limits>
#include <optional>
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
    auto c1 = rfdetr::media::get_color_for_class(5);
    auto c2 = rfdetr::media::get_color_for_class(5);
    EXPECT_EQ(c1, c2);

    // Different classes should (very likely) give different colors
    auto c3 = rfdetr::media::get_color_for_class(0);
    auto c4 = rfdetr::media::get_color_for_class(1);
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
    rfdetr::media::Image img;
    img.resize(200, 100); // width=200, height=100
    std::fill(img.bgr.begin(), img.bgr.end(), 128);
    ASSERT_TRUE(rfdetr::media::save_image(img, tmp_img));

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
        return make_inference(std::move(output_data), std::move(output_shapes), config);
    }

    // Same, for tests that need to vary more of the Config than threshold/resolution.
    std::unique_ptr<RFDETRInference> make_inference(std::vector<std::vector<float>> output_data,
                                                    std::vector<std::vector<int64_t>> output_shapes,
                                                    const Config &config) {
        const int resolution = config.resolution;
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
    std::vector<BoundingBox> boxes;
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
    std::vector<BoundingBox> boxes;
    inference->postprocess_outputs(1.0f, 1.0f, scores, class_ids, boxes);

    ASSERT_EQ(boxes.size(), 1u);
    EXPECT_NEAR(boxes[0].x_min, 40.0f, 0.01f); // x_min
    EXPECT_NEAR(boxes[0].y_min, 45.0f, 0.01f); // y_min
    EXPECT_NEAR(boxes[0].x_max, 60.0f, 0.01f); // x_max
    EXPECT_NEAR(boxes[0].y_max, 55.0f, 0.01f); // y_max
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
    std::vector<BoundingBox> boxes;
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
    std::vector<BoundingBox> boxes;
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
    std::vector<BoundingBox> boxes;
    inference->postprocess_outputs(1.0f, 1.0f, scores, class_ids, boxes);

    ASSERT_EQ(boxes.size(), 2u);
    // det 0: negative x_min/y_min clamped to 0
    EXPECT_NEAR(boxes[0].x_min, 0.0f, 0.01f);
    EXPECT_NEAR(boxes[0].y_min, 0.0f, 0.01f);
    EXPECT_NEAR(boxes[0].x_max, 20.0f, 0.01f);
    EXPECT_NEAR(boxes[0].y_max, 20.0f, 0.01f);
    // det 1: overflowing x_max/y_max clamped to 100
    EXPECT_NEAR(boxes[1].x_min, 80.0f, 0.01f);
    EXPECT_NEAR(boxes[1].y_min, 80.0f, 0.01f);
    EXPECT_NEAR(boxes[1].x_max, 100.0f, 0.01f);
    EXPECT_NEAR(boxes[1].y_max, 100.0f, 0.01f);
}

// --- Multi-class top-k selection (rfdetr 1.9.3, PR #1320) -------------------
//
// Class scores are independent sigmoids, so ranking the flattened query/class
// grid is what keeps a query that clears the threshold on several classes. The
// per-query argmax these paths used before silently dropped all but the
// strongest class.

TEST_F(PostprocessTest, MultiLabelQueryYieldsEveryClassAboveThreshold) {
    const int num_dets = 1;
    const int num_classes = 6;

    std::vector<float> dets_data = {0.5f, 0.5f, 0.2f, 0.1f};

    // One query scoring high on two classes at once ("car" and "truck" in the
    // upstream example): slot 1 -> label 0, slot 3 -> label 2.
    std::vector<float> labels_data(static_cast<size_t>(num_classes), -10.0f);
    labels_data[1] = 4.0f;
    labels_data[3] = 6.0f;

    auto inference =
        make_inference({dets_data, labels_data}, {{1, num_dets, 4}, {1, num_dets, num_classes}}, 0.5f, 560);

    std::vector<float> scores;
    std::vector<int> class_ids;
    std::vector<BoundingBox> boxes;
    inference->postprocess_outputs(1.0f, 1.0f, scores, class_ids, boxes);

    ASSERT_EQ(scores.size(), 2u);
    // Ranked by descending score, so the stronger class comes first.
    EXPECT_EQ(class_ids[0], 2);
    EXPECT_EQ(class_ids[1], 0);
    EXPECT_GT(scores[0], scores[1]);
    // Both detections address the same query, so they share its box.
    EXPECT_FLOAT_EQ(boxes[0].x_min, boxes[1].x_min);
    EXPECT_FLOAT_EQ(boxes[0].y_max, boxes[1].y_max);
}

TEST_F(PostprocessTest, ResultsAreRankedByDescendingScore) {
    const int num_dets = 3;
    const int num_classes = 6;

    std::vector<float> dets_data(static_cast<size_t>(num_dets * 4), 0.5f);
    std::vector<float> labels_data(static_cast<size_t>(num_dets * num_classes), -10.0f);
    labels_data[1] = 2.0f;                     // query 0, weakest
    labels_data[num_classes + 1] = 8.0f;       // query 1, strongest
    labels_data[(2 * num_classes) + 1] = 5.0f; // query 2

    auto inference =
        make_inference({dets_data, labels_data}, {{1, num_dets, 4}, {1, num_dets, num_classes}}, 0.5f, 560);

    std::vector<float> scores;
    std::vector<int> class_ids;
    std::vector<BoundingBox> boxes;
    inference->postprocess_outputs(1.0f, 1.0f, scores, class_ids, boxes);

    ASSERT_EQ(scores.size(), 3u);
    EXPECT_GT(scores[0], scores[1]);
    EXPECT_GT(scores[1], scores[2]);
}

TEST_F(PostprocessTest, TiesResolveByAscendingFlattenedIndex) {
    const int num_dets = 2;
    const int num_classes = 6;

    std::vector<float> dets_data(static_cast<size_t>(num_dets * 4), 0.5f);
    std::vector<float> labels_data(static_cast<size_t>(num_dets * num_classes), -10.0f);
    // Four exactly equal scores across both queries. Upstream's stable rule
    // (descending score, then ascending flattened query/class index) fixes the
    // order: (q0,slot3), (q0,slot4), (q1,slot1), (q1,slot2).
    labels_data[3] = 7.0f;
    labels_data[4] = 7.0f;
    labels_data[num_classes + 1] = 7.0f;
    labels_data[num_classes + 2] = 7.0f;

    auto inference =
        make_inference({dets_data, labels_data}, {{1, num_dets, 4}, {1, num_dets, num_classes}}, 0.5f, 560);

    std::vector<float> scores;
    std::vector<int> class_ids;
    std::vector<BoundingBox> boxes;
    inference->postprocess_outputs(1.0f, 1.0f, scores, class_ids, boxes);

    ASSERT_EQ(class_ids.size(), 4u);
    EXPECT_EQ(class_ids, (std::vector<int>{2, 3, 0, 1}));
}

TEST_F(PostprocessTest, MaxDetectionsCapsCandidatesBeforeThresholding) {
    const int num_dets = 4;
    const int num_classes = 6;

    Config config;
    config.resolution = 560;
    config.threshold = 0.5f;
    config.max_detections = 2;

    std::vector<float> dets_data(static_cast<size_t>(num_dets * 4), 0.5f);
    std::vector<float> labels_data(static_cast<size_t>(num_dets * num_classes), -10.0f);
    for (int q = 0; q < num_dets; ++q) {
        labels_data[static_cast<size_t>(q * num_classes) + 1] = 5.0f + static_cast<float>(q);
    }

    auto inference = make_inference({dets_data, labels_data}, {{1, num_dets, 4}, {1, num_dets, num_classes}}, config);

    std::vector<float> scores;
    std::vector<int> class_ids;
    std::vector<BoundingBox> boxes;
    inference->postprocess_outputs(1.0f, 1.0f, scores, class_ids, boxes);

    // All four clear the threshold; only the two highest-scoring are ranked.
    EXPECT_EQ(scores.size(), 2u);
}

TEST_F(PostprocessTest, NegativeMaxDetectionsRejectedAtConstruction) {
    Config config;
    config.max_detections = -1;

    auto backend = std::make_unique<MockBackend>();
    backend->set_outputs({{}, {}}, {{1, 1, 4}, {1, 1, 6}});

    EXPECT_THROW(RFDETRInference(std::move(backend), labels_file_->path(), config), std::invalid_argument);
}

TEST_F(PostprocessTest, NaNScoreIsDroppedNotKept) {
    const int num_dets = 1;
    const int num_classes = 6;

    std::vector<float> dets_data = {0.5f, 0.5f, 0.2f, 0.1f};
    std::vector<float> labels_data(static_cast<size_t>(num_classes), -10.0f);
    labels_data[2] = std::numeric_limits<float>::quiet_NaN();

    auto inference =
        make_inference({dets_data, labels_data}, {{1, num_dets, 4}, {1, num_dets, num_classes}}, 0.5f, 560);

    std::vector<float> scores;
    std::vector<int> class_ids;
    std::vector<BoundingBox> boxes;
    inference->postprocess_outputs(1.0f, 1.0f, scores, class_ids, boxes);

    // A NaN ranks first (as it does in torch's descending sort) but fails the
    // `score > threshold` test, so it never reaches the output.
    EXPECT_TRUE(scores.empty());
}

// --- Background logit slot (rfdetr 1.9.4, PR #1397) -------------------------

TEST_F(PostprocessTest, BackgroundClassIdNoneKeepsEverySlot) {
    const int num_dets = 1;
    const int num_classes = 5; // exactly as many slots as the fixture has labels

    Config config;
    config.resolution = 560;
    config.threshold = 0.5f;
    config.background_class_id = std::nullopt;

    std::vector<float> dets_data = {0.5f, 0.5f, 0.2f, 0.1f};
    std::vector<float> labels_data(static_cast<size_t>(num_classes), -10.0f);
    labels_data[0] = 10.0f; // slot 0 is a real class when no slot is excluded

    auto inference = make_inference({dets_data, labels_data}, {{1, num_dets, 4}, {1, num_dets, num_classes}}, config);

    std::vector<float> scores;
    std::vector<int> class_ids;
    std::vector<BoundingBox> boxes;
    inference->postprocess_outputs(1.0f, 1.0f, scores, class_ids, boxes);

    ASSERT_EQ(class_ids.size(), 1u);
    EXPECT_EQ(class_ids[0], 0); // no shift: slot 0 -> "person"
}

TEST_F(PostprocessTest, BackgroundClassIdCountsFromTheEnd) {
    const int num_dets = 1;
    const int num_classes = 6;

    Config config;
    config.resolution = 560;
    config.threshold = 0.5f;
    config.background_class_id = -1; // upstream's own default: final slot

    std::vector<float> dets_data = {0.5f, 0.5f, 0.2f, 0.1f};
    std::vector<float> labels_data(static_cast<size_t>(num_classes), -10.0f);
    labels_data[0] = 10.0f;               // now a foreground slot -> label 0
    labels_data[num_classes - 1] = 20.0f; // excluded as background

    auto inference = make_inference({dets_data, labels_data}, {{1, num_dets, 4}, {1, num_dets, num_classes}}, config);

    std::vector<float> scores;
    std::vector<int> class_ids;
    std::vector<BoundingBox> boxes;
    inference->postprocess_outputs(1.0f, 1.0f, scores, class_ids, boxes);

    ASSERT_EQ(class_ids.size(), 1u);
    EXPECT_EQ(class_ids[0], 0);
}

TEST_F(PostprocessTest, BackgroundClassIdOutOfRangeRejected) {
    const int num_dets = 1;
    const int num_classes = 6;

    Config config;
    config.resolution = 560;
    config.background_class_id = num_classes; // one past the last slot

    std::vector<float> dets_data = {0.5f, 0.5f, 0.2f, 0.1f};
    std::vector<float> labels_data(static_cast<size_t>(num_classes), -10.0f);

    auto inference = make_inference({dets_data, labels_data}, {{1, num_dets, 4}, {1, num_dets, num_classes}}, config);

    std::vector<float> scores;
    std::vector<int> class_ids;
    std::vector<BoundingBox> boxes;
    EXPECT_THROW(inference->postprocess_outputs(1.0f, 1.0f, scores, class_ids, boxes), std::invalid_argument);
}

// --- Segmentation shares the detection path's selection --------------------

TEST_F(PostprocessTest, SegmentationRanksFlattenedQueryClassPairs) {
    const int num_dets = 1;
    const int num_classes = 6;
    const int mask_h = 2;
    const int mask_w = 2;

    std::vector<float> dets_data = {0.5f, 0.5f, 0.2f, 0.1f};

    std::vector<float> labels_data(static_cast<size_t>(num_classes), -10.0f);
    labels_data[1] = 4.0f; // -> label 0
    labels_data[3] = 6.0f; // -> label 2

    // One mask per query, positive everywhere so the whole frame is foreground.
    std::vector<float> masks_data(static_cast<size_t>(mask_h * mask_w), 1.0f);

    auto inference =
        make_inference({dets_data, labels_data, masks_data},
                       {{1, num_dets, 4}, {1, num_dets, num_classes}, {1, num_dets, mask_h, mask_w}}, 0.5f, 560);

    std::vector<float> scores;
    std::vector<int> class_ids;
    std::vector<BoundingBox> boxes;
    std::vector<rfdetr::media::Mask> masks;
    inference->postprocess_segmentation_outputs(1.0f, 1.0f, 4, 4, scores, class_ids, boxes, masks);

    // Both classes of the single query survive, strongest first, each carrying
    // its own copy of that query's mask.
    ASSERT_EQ(class_ids.size(), 2u);
    EXPECT_EQ(class_ids[0], 2);
    EXPECT_EQ(class_ids[1], 0);
    ASSERT_EQ(masks.size(), 2u);
    EXPECT_EQ(masks[0].data, masks[1].data);
}

// ============================================================================
// preprocess_bgr_image free function tests
// ============================================================================

TEST(PreprocessFrame, OutputDimensions) {
    rfdetr::media::Image img;
    img.resize(200, 100); // width=200, height=100
    std::fill(img.bgr.begin(), img.bgr.end(), 128);
    const int res = 224;
    std::vector<float> tensor(3 * 224 * 224);
    std::array<float, 3> means = {0.485f, 0.456f, 0.406f};
    std::array<float, 3> stds = {0.229f, 0.224f, 0.225f};

    rfdetr::media::preprocess_bgr_image(img, tensor, res, means, stds);

    for (float v : tensor) {
        EXPECT_TRUE(std::isfinite(v));
    }
}

// Upstream rf-detr 1.9.0 (PR #1206) set antialias=False in predict()'s resize to match the
// antialias-free bilinear (cv2.INTER_LINEAR) resize used during training. preprocess_bgr_image
// must stay antialias-free or inference drifts from the pretrained checkpoints.
//
// Downscaling 4x, each output pixel maps to src = 4*i + 1.5, so bilinear samples exactly the
// 2x2 block at offsets {1,2} within each 4x4 source cell and ignores the cell's other 12 pixels.
// The two patterns below make that footprint unrepresentative of the cell as a whole: an
// averaging (antialiasing) filter would pull both toward the cell mean instead.
TEST(PreprocessFrame, ResizeIsAntialiasFree) {
    constexpr int kSrc = 448;
    constexpr int kRes = 112; // 4x downscale
    const std::array<float, 3> means = {0.0f, 0.0f, 0.0f};
    const std::array<float, 3> stds = {1.0f, 1.0f, 1.0f};

    // Fill the sampled 2x2 of every cell, leaving the surrounding 12 pixels black.
    // Point-sampled: 1.0. Area-averaged over the cell: 4/16 = 0.25.
    rfdetr::media::Image bright;
    bright.resize(kSrc, kSrc);
    std::fill(bright.bgr.begin(), bright.bgr.end(), 0);
    for (int y = 0; y < kSrc; ++y) {
        for (int x = 0; x < kSrc; ++x) {
            const bool sampled = (x % 4 == 1 || x % 4 == 2) && (y % 4 == 1 || y % 4 == 2);
            if (sampled) {
                const size_t idx = (static_cast<size_t>(y) * kSrc + static_cast<size_t>(x)) * 3U;
                bright.bgr[idx] = bright.bgr[idx + 1] = bright.bgr[idx + 2] = 255;
            }
        }
    }

    std::vector<float> tensor(3UL * kRes * kRes);
    rfdetr::media::preprocess_bgr_image(bright, tensor, kRes, means, stds);
    for (float v : tensor) {
        EXPECT_NEAR(v, 1.0f, 1e-4f) << "resize is averaging beyond the bilinear 2x2 footprint";
    }

    // Inverse pattern: the sampled 2x2 is black, the other 12 pixels of each cell are white.
    // Point-sampled: 0.0. Area-averaged over the cell: 12/16 = 0.75.
    rfdetr::media::Image dark;
    dark.resize(kSrc, kSrc);
    std::fill(dark.bgr.begin(), dark.bgr.end(), 255);
    for (int y = 0; y < kSrc; ++y) {
        for (int x = 0; x < kSrc; ++x) {
            const bool sampled = (x % 4 == 1 || x % 4 == 2) && (y % 4 == 1 || y % 4 == 2);
            if (sampled) {
                const size_t idx = (static_cast<size_t>(y) * kSrc + static_cast<size_t>(x)) * 3U;
                dark.bgr[idx] = dark.bgr[idx + 1] = dark.bgr[idx + 2] = 0;
            }
        }
    }

    rfdetr::media::preprocess_bgr_image(dark, tensor, kRes, means, stds);
    for (float v : tensor) {
        EXPECT_NEAR(v, 0.0f, 1e-4f) << "resize is averaging beyond the bilinear 2x2 footprint";
    }
}

// ============================================================================
// Half-pixel bilinear resize convention
//
// rfdetr 1.9.1 made the convention explicit in rfdetr/export/_resize.py: bilinear,
// half-pixel centers (src = (dst + 0.5) * scale - 0.5), source coordinate clamped
// into the source extent, no antialias filter — the same as
// F.interpolate(mode="bilinear", align_corners=False), which is what predict()
// resizes with. preprocess_bgr_image and resize_threshold_mask must both match it.
// ============================================================================

TEST(MaskResize, HalfPixelCenterBilinear) {
    // 4x4 logits, constant down each column: 0, 4, 4, 0. Upscaled 3x with a threshold
    // of 2.0, only the columns whose interpolated value exceeds 2.0 survive, which
    // pins the sample positions: 0.0, 0.0, 1.333, 2.667, 4, 4, 4, 4, 2.667, 1.333, 0.0, 0.0.
    constexpr int kSrc = 4;
    constexpr int kOut = 12;
    std::array<float, kSrc * kSrc> mask{};
    for (int y = 0; y < kSrc; ++y) {
        for (int x = 0; x < kSrc; ++x) {
            mask[static_cast<size_t>(y) * kSrc + static_cast<size_t>(x)] = (x == 1 || x == 2) ? 4.0f : 0.0f;
        }
    }

    const auto out = rfdetr::media::resize_threshold_mask(mask, kSrc, kSrc, kOut, kOut, 2.0f);
    ASSERT_EQ(out.data.size(), static_cast<size_t>(kOut * kOut));

    for (int y = 0; y < kOut; ++y) {
        for (int x = 0; x < kOut; ++x) {
            const uint8_t expected = (x >= 3 && x <= 8) ? 255 : 0;
            EXPECT_EQ(out.data[static_cast<size_t>(y) * kOut + static_cast<size_t>(x)], expected)
                << "half-pixel sample position wrong at (" << x << ", " << y << ")";
        }
    }
}

TEST(MaskResize, LeadingEdgeClampsInsteadOfExtrapolating) {
    // Every source value is above the threshold, so every output pixel must be too.
    // The leading output pixel maps to source coordinate -0.333: clamping the sample
    // index instead of the coordinate leaves a negative weight there, extrapolating
    // 1.333 * 1.0 - 0.333 * 5.0 = -0.333 and dropping the pixel below the threshold.
    constexpr int kSrc = 4;
    constexpr int kOut = 12;
    std::array<float, kSrc * kSrc> mask{};
    for (int y = 0; y < kSrc; ++y) {
        for (int x = 0; x < kSrc; ++x) {
            mask[static_cast<size_t>(y) * kSrc + static_cast<size_t>(x)] = (x == 1 || x == 2) ? 5.0f : 1.0f;
        }
    }

    const auto out = rfdetr::media::resize_threshold_mask(mask, kSrc, kSrc, kOut, kOut, 0.0f);
    for (size_t i = 0; i < out.data.size(); ++i) {
        EXPECT_EQ(out.data[i], 255) << "border extrapolated past the source edge at index " << i;
    }
}

TEST(PreprocessFrame, UpscaleDoesNotExtrapolatePastEdge) {
    // Source smaller than the model resolution, so the resize upscales and the leading
    // row/column land on negative source coordinates. Bilinear resampling of a
    // non-negative image cannot produce a negative sample; extrapolation can.
    constexpr int kSrc = 4;
    constexpr int kRes = 12;
    const std::array<float, 3> means = {0.0f, 0.0f, 0.0f};
    const std::array<float, 3> stds = {1.0f, 1.0f, 1.0f};

    rfdetr::media::Image img;
    img.resize(kSrc, kSrc);
    for (int y = 0; y < kSrc; ++y) {
        for (int x = 0; x < kSrc; ++x) {
            const uint8_t value = (x == 1 || x == 2 || y == 1 || y == 2) ? 255 : 51;
            const size_t idx = (static_cast<size_t>(y) * kSrc + static_cast<size_t>(x)) * 3U;
            img.bgr[idx] = img.bgr[idx + 1] = img.bgr[idx + 2] = value;
        }
    }

    std::vector<float> tensor(3UL * kRes * kRes);
    rfdetr::media::preprocess_bgr_image(img, tensor, kRes, means, stds);

    for (float v : tensor) {
        EXPECT_GE(v, 0.0f) << "resize extrapolated past the source edge";
    }
    // Corner pixel: both coordinates clamp to 0, so it is the source corner exactly (51/255).
    EXPECT_NEAR(tensor[0], 51.0f / 255.0f, 1e-4f);
}

// ============================================================================
// Image preprocess overload tests
// ============================================================================

TEST(Preprocess, ImageOverload) {
    rfdetr::media::Image img;
    img.resize(200, 100); // width=200, height=100
    std::fill(img.bgr.begin(), img.bgr.end(), 128);
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

TEST(Preprocess, ImageOverloadEmptyImage) {
    TempLabelFile labels("person\ncar\n");
    Config config;
    config.resolution = 224;

    auto backend = std::make_unique<MockBackend>();
    backend->set_outputs({{}, {}}, {{1, 1, 4}, {1, 1, 3}});
    RFDETRInference inference(std::move(backend), labels.path(), config);

    rfdetr::media::Image empty;
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

TEST(BoundedQueue, CloseWakesEmptyPop) {
    rfdetr::video::BoundedQueue<size_t> q(4, rfdetr::video::kPoisonPill);
    q.close();
    EXPECT_EQ(q.pop(), rfdetr::video::kPoisonPill);
}

TEST(BoundedQueue, CloseWakesBlockedPush) {
    rfdetr::video::BoundedQueue<size_t> q(1, rfdetr::video::kPoisonPill);
    q.push(1);
    std::atomic<bool> returned{false};
    std::thread t([&] {
        q.push(2);
        returned.store(true);
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    EXPECT_FALSE(returned.load());
    q.close();
    t.join();
    EXPECT_TRUE(returned.load());
    EXPECT_EQ(q.pop(), 1u);
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
    std::vector<BoundingBox> boxes;
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
    std::vector<BoundingBox> boxes;
    std::vector<std::vector<KeypointResult>> keypoints;

    inference->postprocess_keypoint_outputs(1.0f, 1.0f, 100, 200, scores, class_ids, boxes, keypoints);

    ASSERT_EQ(scores.size(), 1u);
    EXPECT_EQ(class_ids[0], 0); // "person"
    EXPECT_GT(scores[0], 0.0f); // Still positive after uncertainty fusion

    // Bbox: cx=50, cy=50, w=20, h=10 → xyxy=(40, 45, 60, 55), scale=1.0
    ASSERT_EQ(boxes.size(), 1u);
    EXPECT_NEAR(boxes[0].x_min, 40.0f, 0.01f);
    EXPECT_NEAR(boxes[0].y_min, 45.0f, 0.01f);
    EXPECT_NEAR(boxes[0].x_max, 60.0f, 0.01f);
    EXPECT_NEAR(boxes[0].y_max, 55.0f, 0.01f);

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
    std::vector<BoundingBox> boxes;
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
    std::vector<BoundingBox> boxes;
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
    std::vector<BoundingBox> boxes;
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
    std::vector<BoundingBox> boxes;
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
    std::vector<BoundingBox> boxes;
    std::vector<std::vector<KeypointResult>> keypoints;

    inference->postprocess_keypoint_outputs(1.0f, 1.0f, 100, 200, scores, class_ids, boxes, keypoints);

    // Background maps to class_id -1 and is skipped.
    EXPECT_TRUE(scores.empty());
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
