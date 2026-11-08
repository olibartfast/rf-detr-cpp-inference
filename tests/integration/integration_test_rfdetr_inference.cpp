#include "../unit/mock_backend.hpp"
#include "processing_utils.hpp"
#include "rfdetr_inference.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <gtest/gtest.h>

namespace {

// Model formats the compiled-in backend can actually load, in preference order. Backend selection
// is compile-time, so probing for a fixed ".onnx" would make an ExecuTorch build pick up a stray
// ONNX export and fail with a confusing format error instead of skipping cleanly.
//
// TensorRT keeps ".onnx" as a lower-priority candidate on purpose: TensorRTBackend::initialize()
// accepts an ONNX model and builds (or reuses a cached) engine from it, so dropping it would lose
// the ONNX-to-engine conversion path from integration coverage. Prebuilt engines come first because
// they load directly and are much faster.
#if defined(USE_TENSORRT)
constexpr std::array kModelExtensions = {".engine", ".trt", ".onnx"};
constexpr const char *kModelFormatName = "TensorRT engine or ONNX model (.engine/.trt/.onnx)";
#elif defined(USE_LITERT)
constexpr std::array kModelExtensions = {".tflite"};
constexpr const char *kModelFormatName = "LiteRT model (.tflite)";
#elif defined(USE_EXECUTORCH)
constexpr std::array kModelExtensions = {".pte"};
constexpr const char *kModelFormatName = "ExecuTorch program (.pte)";
#else
constexpr std::array kModelExtensions = {".onnx"};
constexpr const char *kModelFormatName = "ONNX model (.onnx)";
#endif

// Returns the first <stem><ext> that exists. Extensions are the outer loop so format preference is
// global: for TensorRT, a prebuilt .engine anywhere beats a .onnx anywhere, because loading an
// engine is instant while building one from ONNX costs minutes. A stem-first search would let an
// early .onnx win over a later prebuilt engine, contradicting the documented priority.
template <std::size_t N> std::filesystem::path first_existing_model(const std::array<std::filesystem::path, N> &stems) {
    for (const char *extension : kModelExtensions) {
        for (const auto &stem : stems) {
            if (stem.empty()) {
                continue;
            }
            std::filesystem::path candidate = stem;
            candidate += extension;
            if (std::filesystem::exists(candidate)) {
                return candidate;
            }
        }
    }
    return {};
}

std::filesystem::path resolve_test_model_path() {
    if (const char *env_path = std::getenv("RFDETR_TEST_MODEL")) {
        return env_path;
    }

    const std::filesystem::path home = std::getenv("HOME") ? std::getenv("HOME") : "";
    const std::array<std::filesystem::path, 5> stems = {
        home / "Downloads" / "rfdetr-medium",
        home / "Downloads" / "rfdetr-seg-medium",
        home / "Downloads" / "inference_model",
        std::filesystem::path("exports") / "rfdetr-medium",
        std::filesystem::path("output") / "rfdetr-medium",
    };

    return first_existing_model(stems);
}

std::filesystem::path resolve_keypoint_model_path() {
    if (const char *env_path = std::getenv("RFDETR_KEYPOINT_MODEL")) {
        return env_path;
    }

    const std::filesystem::path home = std::getenv("HOME") ? std::getenv("HOME") : "";
    const std::array<std::filesystem::path, 6> stems = {
        home / "Downloads" / "rfdetr-keypoint",
        home / "Downloads" / "rfdetr-keypoint-preview",
        std::filesystem::path("exports") / "rfdetr-keypoint",
        std::filesystem::path("exports") / "rfdetr-keypoint-preview",
        std::filesystem::path("output") / "rfdetr-keypoint",
        std::filesystem::path("output") / "rfdetr-keypoint-preview",
    };

    return first_existing_model(stems);
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
            GTEST_SKIP() << "No " << kModelFormatName                                                                  \
                         << " found for the compiled-in backend. Export with rfdetr 1.9.4 or set "                     \
                            "RFDETR_TEST_MODEL.";                                                                      \
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
    std::vector<BoundingBox> boxes;
    const float scale_w = static_cast<float>(orig_w) / static_cast<float>(inference.get_resolution());
    const float scale_h = static_cast<float>(orig_h) / static_cast<float>(inference.get_resolution());
    inference.postprocess_outputs(scale_w, scale_h, scores, class_ids, boxes);

    // Load image for drawing
    rfdetr::media::Image image = rfdetr::media::load_image(image_path_);
    ASSERT_FALSE(image.empty());

    // Draw detections
    RFDETRInference::draw_detections(image, boxes, class_ids, scores);

    // Save output
    EXPECT_TRUE(RFDETRInference::save_output_image(image, output_path_).has_value());
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
            GTEST_SKIP() << "No keypoint " << kModelFormatName                                                         \
                         << " found for the compiled-in backend. Export with deploy/export_keypoint.py or set "        \
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
    std::vector<BoundingBox> boxes;
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

    EXPECT_TRUE(RFDETRInference::save_output_image(image, output_path_).has_value());
    EXPECT_TRUE(std::filesystem::exists(output_path_));
}

namespace {
std::vector<float> read_pose_tensor(const std::filesystem::path &path, size_t count) {
    std::vector<float> values(count);
    std::ifstream input(path, std::ios::binary);
    if (!input.read(reinterpret_cast<char *>(values.data()), static_cast<std::streamsize>(count * sizeof(float))) ||
        input.peek() != std::char_traits<char>::eof()) {
        throw std::runtime_error("Invalid pose reference tensor: " + path.string());
    }
    return values;
}
} // namespace

TEST(PoseScoreReference, MatchesUpstreamDecodedScoresAndCoordinates) {
    const char *reference = std::getenv("RFDETR_POSE_REFERENCE");
    if (reference == nullptr) {
        GTEST_SKIP() << "Set RFDETR_POSE_REFERENCE from generate_pose_reference.py";
    }
    const std::filesystem::path directory(reference);
    size_t queries = 0;
    int resolution = 0;
    std::ifstream metadata(directory / "metadata.txt");
    ASSERT_TRUE(static_cast<bool>(metadata >> queries >> resolution));
    ASSERT_GT(queries, 0U);
    ASSERT_GT(resolution, 0);
    auto backend = std::make_unique<MockBackend>();
    const auto query_count = static_cast<int64_t>(queries);
    backend->set_outputs({read_pose_tensor(directory / "output_0.bin", queries * 4),
                          read_pose_tensor(directory / "output_1.bin", queries * 2),
                          read_pose_tensor(directory / "output_2.bin", queries * 34 * 8)},
                         {{1, query_count, 4}, {1, query_count, 2}, {1, query_count, 34, 8}});
    Config config;
    config.model_type = ModelType::KEYPOINT;
    config.resolution = resolution;
    config.threshold = 0.0F;
    RFDETRInference inference(std::move(backend), directory / "labels.txt", config);
    inference.run_inference({});
    std::vector<float> scores;
    std::vector<int> classes;
    std::vector<BoundingBox> boxes;
    std::vector<std::vector<KeypointResult>> keypoints;
    const float scale_width = 640.0F / static_cast<float>(resolution);
    const float scale_height = 480.0F / static_cast<float>(resolution);
    inference.postprocess_keypoint_outputs(scale_width, scale_height, 480, 640, scores, classes, boxes, keypoints);
    ASSERT_EQ(scores.size(), queries);
    std::ifstream expected(directory / "expected.txt");
    ASSERT_TRUE(expected.is_open());
    for (size_t i = 0; i < queries; ++i) {
        float value = 0;
        ASSERT_TRUE(static_cast<bool>(expected >> value));
        EXPECT_NEAR(scores[i], value, 1e-5F) << "score " << i;
        EXPECT_EQ(classes[i], 0);
        for (float actual : {boxes[i].x_min, boxes[i].y_min, boxes[i].x_max, boxes[i].y_max}) {
            ASSERT_TRUE(static_cast<bool>(expected >> value));
            EXPECT_NEAR(actual, value, 1e-3F);
        }
        ASSERT_EQ(keypoints[i].size(), 17U);
        for (const auto &keypoint : keypoints[i]) {
            for (float actual : {keypoint.x, keypoint.y}) {
                ASSERT_TRUE(static_cast<bool>(expected >> value));
                EXPECT_NEAR(actual, value, 1e-3F);
            }
        }
    }
}

#ifdef USE_ONNX_RUNTIME
TEST(OnnxThreading, ConfigurableThreadsPreserveOutputs) {
    const char *model = std::getenv("RFDETR_TEST_MODEL");
    if (model == nullptr) {
        GTEST_SKIP() << "Set RFDETR_TEST_MODEL to run ONNX thread parity and timing";
    }
    std::vector<std::vector<float>> baseline;
    for (int threads : {1, 2, 4, 0}) {
        auto backend = rfdetr::backend::create_backend({threads});
        const auto shape = backend->initialize(model, {1, 3, 0, 0});
        size_t count = 1;
        for (int64_t dimension : shape) {
            ASSERT_GT(dimension, 0);
            count *= static_cast<size_t>(dimension);
        }
        std::vector<float> input(count);
        for (size_t i = 0; i < count; ++i) {
            input[i] = static_cast<float>(i % 251) / 125.0F - 1.0F;
        }
        (void)backend->run_inference(input, shape);
        std::vector<double> durations;
        for (int repeat = 0; repeat < 5; ++repeat) {
            const auto start = std::chrono::steady_clock::now();
            (void)backend->run_inference(input, shape);
            durations.push_back(
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
        }
        std::ranges::sort(durations);
        RecordProperty("median_ms_threads_" + std::to_string(threads), std::to_string(durations[2]));
        std::vector<std::vector<float>> outputs;
        for (size_t index = 0; index < backend->get_output_count(); ++index) {
            size_t size = 1;
            for (int64_t dimension : backend->get_output_shape(index)) {
                ASSERT_GT(dimension, 0);
                size *= static_cast<size_t>(dimension);
            }
            outputs.emplace_back(size);
            backend->get_output_data(index, outputs.back().data(), size);
        }
        if (threads == 1) {
            baseline = std::move(outputs);
            continue;
        }
        ASSERT_EQ(outputs.size(), baseline.size());
        for (size_t index = 0; index < outputs.size(); ++index) {
            ASSERT_EQ(outputs[index].size(), baseline[index].size());
            float maximum = 0;
            for (size_t i = 0; i < outputs[index].size(); ++i) {
                ASSERT_TRUE(std::isfinite(outputs[index][i]));
                ASSERT_TRUE(std::isfinite(baseline[index][i]));
                maximum = std::max(maximum, std::abs(outputs[index][i] - baseline[index][i]));
            }
            const float tolerance = index == 0 ? 1.0F / static_cast<float>(shape[2]) : 1e-3F;
            EXPECT_LE(maximum, tolerance) << "threads=" << threads << " output=" << index;
            if (index == 1) {
                float score_delta = 0;
                for (size_t i = 0; i < outputs[index].size(); ++i) {
                    score_delta = std::max(score_delta, std::abs(rfdetr::processing::sigmoid(outputs[index][i]) -
                                                                 rfdetr::processing::sigmoid(baseline[index][i])));
                }
                EXPECT_LE(score_delta, 1e-4F);
                RecordProperty("max_score_delta_threads_" + std::to_string(threads), std::to_string(score_delta));
            }
            RecordProperty("max_abs_delta_threads_" + std::to_string(threads) + "_output_" + std::to_string(index),
                           std::to_string(maximum));
        }
    }
}
#endif

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
