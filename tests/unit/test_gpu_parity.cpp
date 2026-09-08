// GPU parity gate for the DALI preprocessor and the CUDA segmentation
// postprocessor, measured against golden CPU fixtures under
// tests/data/gpu_parity/ (see that directory's README for provenance).
//
// The CPU-determinism test runs with no device and is never skipped; the
// preprocess and postprocess cases skip rather than fail when no CUDA device is
// present, because CI compiles the GPU targets on runners that have no GPU.

#include "gpu_parity_fixtures.hpp"
#include "gpu_test_utils.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#ifndef GPU_PARITY_FIXTURE_DIR
#define GPU_PARITY_FIXTURE_DIR "tests/data/gpu_parity"
#endif
#ifndef GPU_PARITY_DALI_DIR
#define GPU_PARITY_DALI_DIR "data/dali"
#endif

namespace {

const std::filesystem::path kFixtureDir = GPU_PARITY_FIXTURE_DIR;

class TempLabelFile {
  public:
    explicit TempLabelFile(const std::string &content, const std::string &name = "gpu_parity_labels.txt")
        : path_(std::filesystem::temp_directory_path() / name) {
        std::ofstream file(path_);
        file << content;
    }
    ~TempLabelFile() { std::filesystem::remove(path_); }
    TempLabelFile(const TempLabelFile &) = delete;
    TempLabelFile &operator=(const TempLabelFile &) = delete;
    [[nodiscard]] const std::filesystem::path &path() const { return path_; }

  private:
    std::filesystem::path path_;
};

std::filesystem::path fixture_file(const std::string &name, const char *ext) { return kFixtureDir / (name + ext); }

// ============================================================================
// Group 2: CPU determinism — no device, never skipped
// ============================================================================

TEST(GpuParityCpuDeterminism, FixturesReproduceBitIdentically) {
    for (const auto &spec : gpu_parity::kNaturalFixtures) {
        SCOPED_TRACE(spec.name);
        const auto image = rfdetr::media::load_image(fixture_file(spec.name, ".jpg"));
        ASSERT_FALSE(image.empty()) << "missing fixture image for " << spec.name;

        const auto first = gpu_parity::cpu_preprocess(image);
        const auto second = gpu_parity::cpu_preprocess(image);
        EXPECT_EQ(first, second) << "CPU preprocess is not bit-reproducible within one process";

        const auto stored = gpu_parity::read_preprocessed(fixture_file(spec.name, ".preprocessed.bin"));
        ASSERT_EQ(stored.size(), first.size());
        EXPECT_EQ(first, stored) << "re-derived tensor differs from the stored golden fixture";
    }
}

// ============================================================================
// Group 3: DALI preprocess vs CPU preprocess (needs USE_DALI)
// ============================================================================

#ifdef USE_DALI

std::vector<float> gpu_preprocess_encoded(const std::filesystem::path &image_path, rfdetr::gpu::GpuContext &context) {
    std::ifstream file(image_path, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error("missing fixture image: " + image_path.string());
    }
    const auto size = static_cast<size_t>(file.tellg());
    file.seekg(0, std::ios::beg);
    std::vector<std::uint8_t> bytes(size);
    if (!file.read(reinterpret_cast<char *>(bytes.data()), static_cast<std::streamsize>(size))) {
        throw std::runtime_error("failed to read fixture image: " + image_path.string());
    }

    const auto pipeline = std::filesystem::path(GPU_PARITY_DALI_DIR) /
                          ("preprocess_encoded_" + std::to_string(gpu_parity::kResolution) + ".dali");
    rfdetr::gpu::DaliPreprocessor preprocessor(pipeline, rfdetr::gpu::DaliPreprocessor::Source::EncodedImage, 0);

    const size_t tensor_bytes =
        3 * static_cast<size_t>(gpu_parity::kResolution) * static_cast<size_t>(gpu_parity::kResolution) * sizeof(float);
    rfdetr::gpu::DeviceBuffer buffer(tensor_bytes);
    preprocessor.process_encoded(bytes, buffer.get(), tensor_bytes, context.stream());

    std::vector<float> tensor(3 * static_cast<size_t>(gpu_parity::kResolution) *
                              static_cast<size_t>(gpu_parity::kResolution));
    rfdetr::gpu::copy_d2h(tensor.data(), buffer.get(), tensor_bytes, context.stream());
    rfdetr::gpu::stream_synchronize(context.stream());
    return tensor;
}

float max_abs_delta(const std::vector<float> &a, const std::vector<float> &b) {
    float max_delta = 0.0F;
    for (size_t i = 0; i < a.size(); ++i) {
        max_delta = std::max(max_delta, std::abs(a[i] - b[i]));
    }
    return max_delta;
}

/// Frame-path preprocess: the already-decoded BGR image is uploaded once and
/// resized by DALI, so the JPEG decode (nvJPEG vs stb) drops out of the
/// comparison and only resize + normalisation are measured.
std::vector<float> gpu_preprocess_frame(const rfdetr::media::Image &image, rfdetr::gpu::GpuContext &context) {
    const auto pipeline = std::filesystem::path(GPU_PARITY_DALI_DIR) /
                          ("preprocess_frame_" + std::to_string(gpu_parity::kResolution) + ".dali");
    rfdetr::gpu::DaliPreprocessor preprocessor(pipeline, rfdetr::gpu::DaliPreprocessor::Source::BgrFrame, 0);

    const size_t tensor_bytes =
        3 * static_cast<size_t>(gpu_parity::kResolution) * static_cast<size_t>(gpu_parity::kResolution) * sizeof(float);
    rfdetr::gpu::DeviceBuffer frame(image.bytes());
    rfdetr::gpu::copy_h2d(frame.get(), image.data(), image.bytes(), context.stream());
    rfdetr::gpu::DeviceBuffer out(tensor_bytes);
    preprocessor.process_frame(frame.get(), image.height, image.width, out.get(), tensor_bytes, context.stream());

    std::vector<float> tensor(3 * static_cast<size_t>(gpu_parity::kResolution) *
                              static_cast<size_t>(gpu_parity::kResolution));
    rfdetr::gpu::copy_d2h(tensor.data(), out.get(), tensor_bytes, context.stream());
    rfdetr::gpu::stream_synchronize(context.stream());
    return tensor;
}

// Resize + normalisation parity: the load-bearing rule in
// specs/gpu-pipeline.md is "resize is a tolerance gate". Both sides decode the
// same bytes here, so the comparison isolates what DALI controls beyond decode.
TEST(GpuParityPreprocess, ResizeMatchesCpu) {
    SKIP_WITHOUT_GPU();
    rfdetr::gpu::GpuContext context(0);

    for (const auto &spec : gpu_parity::kNaturalFixtures) {
        SCOPED_TRACE(spec.name);
        const auto image = rfdetr::media::load_image(fixture_file(spec.name, ".jpg"));
        ASSERT_FALSE(image.empty());

        const auto cpu = gpu_parity::cpu_preprocess(image);
        const auto gpu = gpu_preprocess_frame(image, context);
        ASSERT_EQ(gpu.size(), cpu.size());

        const float max_delta = max_abs_delta(cpu, gpu);
        std::cout << "[gpu-parity] " << spec.name << " resize max |delta| = " << max_delta << '\n';
        EXPECT_LE(max_delta, 2e-2F) << "DALI resize/normalise diverged from the CPU path";
    }
}

// Full encoded-path preprocess: nvJPEG decode + resize + normalise. nvJPEG and
// stb are different JPEG decoders, so this path cannot bit-match the CPU tensor;
// the bound is generous enough to admit the decode difference while still
// catching a broken pipeline (wrong mean/std, letterbox, wrong resize).
TEST(GpuParityPreprocess, EncodedPathBounded) {
    SKIP_WITHOUT_GPU();
    rfdetr::gpu::GpuContext context(0);

    for (const auto &spec : gpu_parity::kNaturalFixtures) {
        SCOPED_TRACE(spec.name);
        const auto image_path = fixture_file(spec.name, ".jpg");
        const auto image = rfdetr::media::load_image(image_path);
        ASSERT_FALSE(image.empty());

        const auto cpu = gpu_parity::cpu_preprocess(image);
        const auto gpu = gpu_preprocess_encoded(image_path, context);
        ASSERT_EQ(gpu.size(), cpu.size());

        const float max_delta = max_abs_delta(cpu, gpu);
        std::cout << "[gpu-parity] " << spec.name << " encoded max |delta| = " << max_delta << '\n';
        EXPECT_LE(max_delta, 1e-1F) << "encoded preprocess diverged far beyond the decode difference";
    }
}

/// True if a tensor border (0=top, 1=bottom, 2=left, 3=right) is constant across
/// all three channels — the signature of letterbox padding.
bool border_is_constant(const std::vector<float> &t, int h, int w, int border) {
    const auto at = [&](int c, int y, int x) {
        return t[static_cast<size_t>(c) * static_cast<size_t>(h) * static_cast<size_t>(w) +
                 static_cast<size_t>(y) * static_cast<size_t>(w) + static_cast<size_t>(x)];
    };
    float first = 0.0F;
    bool have = false;
    for (int c = 0; c < 3; ++c) {
        for (int i = 0; i < (border < 2 ? w : h); ++i) {
            const float v = (border == 0)   ? at(c, 0, i)
                            : (border == 1) ? at(c, h - 1, i)
                            : (border == 2) ? at(c, i, 0)
                                            : at(c, i, w - 1);
            if (!have) {
                first = v;
                have = true;
            } else if (std::abs(v - first) > 1e-6F) {
                return false;
            }
        }
    }
    return have;
}

TEST(GpuParityPreprocess, NoLetterboxBorders) {
    SKIP_WITHOUT_GPU();
    rfdetr::gpu::GpuContext context(0);
    const int res = gpu_parity::kResolution;

    for (const char *name : {"wide", "tall"}) {
        SCOPED_TRACE(name);
        const auto image = rfdetr::media::load_image(fixture_file(name, ".jpg"));
        ASSERT_FALSE(image.empty());

        const auto cpu = gpu_parity::cpu_preprocess(image);
        const auto gpu = gpu_preprocess_encoded(fixture_file(name, ".jpg"), context);
        ASSERT_EQ(gpu.size(), cpu.size());

        // Sanity: the synthetic source has no constant border on the CPU path.
        for (int border = 0; border < 4; ++border) {
            EXPECT_FALSE(border_is_constant(cpu, res, res, border))
                << "CPU fixture border " << border << " is constant";
            EXPECT_FALSE(border_is_constant(gpu, res, res, border))
                << "GPU tensor border " << border << " is constant (letterbox suspected)";
        }
    }
}

#endif // USE_DALI

// ============================================================================
// Group 3 (end-to-end): GPU postprocess vs golden detections (needs USE_CUDA_POSTPROCESS)
// ============================================================================

#ifdef USE_CUDA_POSTPROCESS

Config make_config() {
    Config config;
    config.resolution = gpu_parity::kResolution;
    config.threshold = gpu_parity::kThreshold;
    config.mask_threshold = gpu_parity::kMaskThreshold;
    config.max_detections = gpu_parity::kMaxDetections;
    config.model_type = ModelType::SEGMENTATION;
    config.gpu_postprocess = true;
    config.gpu_preprocess = false;
    return config;
}

std::vector<gpu_parity::Detection> gpu_decode(const gpu_parity::SyntheticOutputs &outputs,
                                              const std::filesystem::path &labels, int orig_w, int orig_h) {
    rfdetr::gpu::GpuContext context(0);
    auto backend = std::make_unique<MockDeviceBackend>(context.stream());
    backend->set_outputs(outputs.tensors(), outputs.shapes());
    backend->upload();
    RFDETRInference inference(std::move(backend), labels, make_config());

    std::vector<float> scores;
    std::vector<int> class_ids;
    std::vector<BoundingBox> boxes;
    std::vector<rfdetr::media::Mask> masks;
    const float scale_w = static_cast<float>(orig_w) / static_cast<float>(gpu_parity::kResolution);
    const float scale_h = static_cast<float>(orig_h) / static_cast<float>(gpu_parity::kResolution);
    inference.postprocess_segmentation_outputs_gpu(scale_w, scale_h, orig_h, orig_w, scores, class_ids, boxes, masks);

    std::vector<gpu_parity::Detection> detections;
    detections.reserve(scores.size());
    for (size_t i = 0; i < scores.size(); ++i) {
        detections.push_back({class_ids[i], scores[i], boxes[i].x_min, boxes[i].y_min, boxes[i].x_max, boxes[i].y_max});
    }
    return detections;
}

/// Compare two detection sets: same class set and count, scores within 1e-3, box
/// centres within 1 px. Compared as a set, never on order.
void expect_detection_sets(const std::vector<gpu_parity::Detection> &actual,
                           const std::vector<gpu_parity::Detection> &expected) {
    ASSERT_EQ(actual.size(), expected.size()) << "detection counts differ";

    auto key = [](const gpu_parity::Detection &d) {
        return std::tuple<int, float, float, float, float, float>{d.class_id, d.score, d.x_min,
                                                                  d.y_min,    d.x_max, d.y_max};
    };
    auto sorted_actual = actual;
    auto sorted_expected = expected;
    std::sort(sorted_actual.begin(), sorted_actual.end(),
              [&](const auto &a, const auto &b) { return key(a) < key(b); });
    std::sort(sorted_expected.begin(), sorted_expected.end(),
              [&](const auto &a, const auto &b) { return key(a) < key(b); });

    for (size_t i = 0; i < sorted_actual.size(); ++i) {
        SCOPED_TRACE("detection " + std::to_string(i));
        const auto &a = sorted_actual[i];
        const auto &e = sorted_expected[i];
        EXPECT_EQ(a.class_id, e.class_id);
        EXPECT_NEAR(a.score, e.score, 1e-3F);
        EXPECT_NEAR((a.x_min + a.x_max) / 2.0F, (e.x_min + e.x_max) / 2.0F, 1.0F);
        EXPECT_NEAR((a.y_min + a.y_max) / 2.0F, (e.y_min + e.y_max) / 2.0F, 1.0F);
    }
}

void expect_fixture_regression(const std::string &name) {
    SCOPED_TRACE(name);
    TempLabelFile labels("c0\nc1\nc2\nc3\nc4\nc5\nc6\nc7\nc8\n");

    const auto outputs = gpu_parity::read_outputs(fixture_file(name, ".outputs.bin"));
    const auto expected = gpu_parity::read_expected(fixture_file(name, ".expected.txt"));

    const auto actual = gpu_decode(outputs, labels.path(), expected.orig_w, expected.orig_h);
    expect_detection_sets(actual, expected.detections);
}

TEST(GpuParityRegression, NaturalFixtures) {
    SKIP_WITHOUT_GPU();
    for (const auto &spec : gpu_parity::kNaturalFixtures) {
        expect_fixture_regression(spec.name);
    }
}

TEST(GpuParityRegression, DenseFixture) {
    SKIP_WITHOUT_GPU();
    const auto expected = gpu_parity::read_expected(fixture_file("dense", ".expected.txt"));
    ASSERT_GT(expected.detections.size(), 100U) << "dense fixture must yield > 100 above-threshold detections";
    expect_fixture_regression("dense");
}

#endif // USE_CUDA_POSTPROCESS

} // namespace
