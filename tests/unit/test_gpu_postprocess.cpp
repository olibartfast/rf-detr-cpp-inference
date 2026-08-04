// CPU-versus-GPU parity gate for the CUDA segmentation postprocessor.
//
// No model and no DALI are involved: synthetic dets/labels/masks tensors are fed
// to a mock backend that serves them from both host and device memory, so the two
// postprocessors consume byte-identical inputs and any difference is theirs.
//
// Every test skips (rather than fails) when no CUDA device is present, because CI
// compiles these targets on runners that have no GPU.

#include "gpu/gpu_context.hpp"
#include "mock_backend.hpp"
#include "rfdetr_inference.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace {

#define SKIP_WITHOUT_GPU()                                                                                             \
    do {                                                                                                               \
        if (!rfdetr::gpu::device_available()) {                                                                        \
            GTEST_SKIP() << "no CUDA device available";                                                                \
        }                                                                                                              \
    } while (false)

/// Mock backend that also publishes its outputs as device pointers, so the GPU
/// postprocessor can read exactly what the CPU postprocessor reads.
class MockDeviceBackend : public MockBackend {
  public:
    explicit MockDeviceBackend(rfdetr::gpu::StreamHandle stream) : stream_(stream) {}

    /// Stages the configured outputs into device memory. Call after set_outputs().
    void upload() {
        device_buffers_.clear();
        device_buffers_.reserve(output_data_.size());
        for (const auto &tensor : output_data_) {
            const size_t bytes = tensor.size() * sizeof(float);
            auto buffer = std::make_unique<rfdetr::gpu::DeviceBuffer>(bytes);
            rfdetr::gpu::copy_h2d(buffer->get(), tensor.data(), bytes, stream_);
            device_buffers_.push_back(std::move(buffer));
        }
        rfdetr::gpu::stream_synchronize(stream_);
    }

    [[nodiscard]] bool supports_device_io() const noexcept override { return true; }

    [[nodiscard]] const void *get_output_device_ptr(size_t output_index) const override {
        if (output_index >= device_buffers_.size()) {
            throw std::out_of_range("Device output index out of range");
        }
        return device_buffers_[output_index]->get();
    }

    [[nodiscard]] void *device_stream() const noexcept override { return stream_; }

    void synchronize_device() override { rfdetr::gpu::stream_synchronize(stream_); }

  private:
    rfdetr::gpu::StreamHandle stream_;
    std::vector<std::unique_ptr<rfdetr::gpu::DeviceBuffer>> device_buffers_;
};

class TempLabelFile {
  public:
    explicit TempLabelFile(const std::string &content, const std::string &name = "gpu_test_labels.txt")
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

/// One synthetic segmentation model output set.
struct Fixture {
    int num_queries;
    int num_classes;
    int mask_h;
    int mask_w;
    std::vector<float> dets;
    std::vector<float> labels;
    std::vector<float> masks;

    [[nodiscard]] std::vector<std::vector<float>> tensors() const { return {dets, labels, masks}; }
    [[nodiscard]] std::vector<std::vector<int64_t>> shapes() const {
        return {{1, num_queries, 4}, {1, num_queries, num_classes}, {1, num_queries, mask_h, mask_w}};
    }
};

/// Builds a fixture with `num_hits` queries scored above the threshold.
///
/// Mask logits are a smooth ramp crossed by a blob, deliberately spanning the
/// threshold so the bilinear resample has to agree near the decision boundary —
/// a mask that is uniformly far from the threshold would pass even a badly wrong
/// interpolation.
Fixture make_fixture(int num_queries, int num_classes, int mask_h, int mask_w, int num_hits, uint32_t seed = 7) {
    Fixture f{num_queries, num_classes, mask_h, mask_w, {}, {}, {}};
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> unit(0.0F, 1.0F);

    f.dets.resize(static_cast<size_t>(num_queries) * 4);
    for (int q = 0; q < num_queries; ++q) {
        const float cx = 0.15F + 0.7F * unit(rng);
        const float cy = 0.15F + 0.7F * unit(rng);
        const float w = 0.05F + 0.25F * unit(rng);
        const float h = 0.05F + 0.25F * unit(rng);
        f.dets[static_cast<size_t>(q) * 4 + 0] = cx;
        f.dets[static_cast<size_t>(q) * 4 + 1] = cy;
        f.dets[static_cast<size_t>(q) * 4 + 2] = w;
        f.dets[static_cast<size_t>(q) * 4 + 3] = h;
    }

    // Background (logit 0) stays low so it never wins; the class index shifts by
    // one on the way out, so logit 1 becomes class 0.
    f.labels.assign(static_cast<size_t>(num_queries) * static_cast<size_t>(num_classes), -8.0F);
    for (int q = 0; q < num_hits && q < num_queries; ++q) {
        const int class_slot = 1 + (q % (num_classes - 1));
        // Distinct scores, all comfortably above 0.5, so ranking is unambiguous
        // and tie-breaking never enters into it.
        f.labels[static_cast<size_t>(q) * static_cast<size_t>(num_classes) + static_cast<size_t>(class_slot)] =
            2.0F + 0.01F * static_cast<float>(num_hits - q);
    }

    f.masks.resize(static_cast<size_t>(num_queries) * static_cast<size_t>(mask_h) * static_cast<size_t>(mask_w));
    for (int q = 0; q < num_queries; ++q) {
        for (int y = 0; y < mask_h; ++y) {
            for (int x = 0; x < mask_w; ++x) {
                const float fx = static_cast<float>(x) / static_cast<float>(mask_w);
                const float fy = static_cast<float>(y) / static_cast<float>(mask_h);
                const float blob = std::exp(-12.0F * ((fx - 0.5F) * (fx - 0.5F) + (fy - 0.5F) * (fy - 0.5F)));
                const float value = 4.0F * blob - 1.5F + 0.5F * (fx - fy) + 0.1F * static_cast<float>(q % 5);
                f.masks[(static_cast<size_t>(q) * static_cast<size_t>(mask_h) + static_cast<size_t>(y)) *
                            static_cast<size_t>(mask_w) +
                        static_cast<size_t>(x)] = value;
            }
        }
    }
    return f;
}

struct SegOutputs {
    std::vector<float> scores;
    std::vector<int> class_ids;
    std::vector<BoundingBox> boxes;
    std::vector<rfdetr::media::Mask> masks;
};

Config make_config(int resolution, float threshold, float mask_threshold, int max_detections) {
    Config config;
    config.resolution = resolution;
    config.threshold = threshold;
    config.mask_threshold = mask_threshold;
    config.max_detections = max_detections;
    config.model_type = ModelType::SEGMENTATION;
    config.gpu_postprocess = true;
    config.gpu_preprocess = false; // no DALI pipeline needed for postprocess tests
    return config;
}

/// Intersection-over-union of two binary masks.
double mask_iou(const rfdetr::media::Mask &a, const rfdetr::media::Mask &b) {
    if (a.width != b.width || a.height != b.height || a.data.size() != b.data.size()) {
        return 0.0;
    }
    size_t intersection = 0;
    size_t union_count = 0;
    for (size_t i = 0; i < a.data.size(); ++i) {
        const bool lhs = a.data[i] != 0;
        const bool rhs = b.data[i] != 0;
        intersection += static_cast<size_t>(lhs && rhs);
        union_count += static_cast<size_t>(lhs || rhs);
    }
    if (union_count == 0) {
        return 1.0; // both empty: identical
    }
    return static_cast<double>(intersection) / static_cast<double>(union_count);
}

/// Runs both postprocessors over the same fixture and returns (cpu, gpu).
std::pair<SegOutputs, SegOutputs> run_both(const Fixture &fixture, const Config &config, int orig_w, int orig_h,
                                           const std::filesystem::path &labels) {
    rfdetr::gpu::GpuContext context(config.gpu_device_id);

    const float scale_w = static_cast<float>(orig_w) / static_cast<float>(config.resolution);
    const float scale_h = static_cast<float>(orig_h) / static_cast<float>(config.resolution);

    SegOutputs cpu;
    {
        auto backend = std::make_unique<MockBackend>();
        backend->set_outputs(fixture.tensors(), fixture.shapes());
        RFDETRInference inference(std::move(backend), labels, config);
        inference.run_inference({});
        inference.postprocess_segmentation_outputs(scale_w, scale_h, orig_h, orig_w, cpu.scores, cpu.class_ids,
                                                   cpu.boxes, cpu.masks);
    }

    SegOutputs gpu;
    {
        auto backend = std::make_unique<MockDeviceBackend>(context.stream());
        backend->set_outputs(fixture.tensors(), fixture.shapes());
        backend->upload();
        RFDETRInference inference(std::move(backend), labels, config);
        inference.postprocess_segmentation_outputs_gpu(scale_w, scale_h, orig_h, orig_w, gpu.scores, gpu.class_ids,
                                                       gpu.boxes, gpu.masks);
    }

    return {std::move(cpu), std::move(gpu)};
}

void expect_parity(const SegOutputs &cpu, const SegOutputs &gpu, double min_mask_iou = 0.999) {
    ASSERT_EQ(cpu.scores.size(), gpu.scores.size()) << "detection counts differ";
    ASSERT_EQ(cpu.class_ids.size(), gpu.class_ids.size());
    ASSERT_EQ(cpu.boxes.size(), gpu.boxes.size());
    ASSERT_EQ(cpu.masks.size(), gpu.masks.size());

    for (size_t i = 0; i < cpu.scores.size(); ++i) {
        SCOPED_TRACE("detection " + std::to_string(i));
        EXPECT_EQ(cpu.class_ids[i], gpu.class_ids[i]);
        EXPECT_NEAR(cpu.scores[i], gpu.scores[i], 1e-3F);
        EXPECT_NEAR(cpu.boxes[i].x_min, gpu.boxes[i].x_min, 1e-3F);
        EXPECT_NEAR(cpu.boxes[i].y_min, gpu.boxes[i].y_min, 1e-3F);
        EXPECT_NEAR(cpu.boxes[i].x_max, gpu.boxes[i].x_max, 1e-3F);
        EXPECT_NEAR(cpu.boxes[i].y_max, gpu.boxes[i].y_max, 1e-3F);

        EXPECT_EQ(cpu.masks[i].width, gpu.masks[i].width);
        EXPECT_EQ(cpu.masks[i].height, gpu.masks[i].height);
        // Report the offending instance rather than just a failed comparison, so
        // one bad mask out of a hundred is debuggable.
        const double iou = mask_iou(cpu.masks[i], gpu.masks[i]);
        EXPECT_GE(iou, min_mask_iou) << "mask IoU too low for detection " << i << " (class " << cpu.class_ids[i]
                                     << ", box [" << cpu.boxes[i].x_min << ", " << cpu.boxes[i].y_min << ", "
                                     << cpu.boxes[i].x_max << ", " << cpu.boxes[i].y_max << "])";
    }
}

} // namespace

TEST(GpuContextTest, ReportsDeviceAvailability) {
    // Must not throw or crash whether or not a device exists.
    const bool available = rfdetr::gpu::device_available();
    if (!available) {
        GTEST_SKIP() << "no CUDA device available";
    }
    EXPECT_FALSE(rfdetr::gpu::device_name(0).empty());
}

TEST(GpuContextTest, CreatesAndDestroysStream) {
    SKIP_WITHOUT_GPU();
    rfdetr::gpu::GpuContext context(0);
    EXPECT_NE(context.stream(), nullptr);
    EXPECT_EQ(context.device_id(), 0);
    EXPECT_NO_THROW(context.synchronize());
}

TEST(GpuContextTest, DeviceBufferGrowsMonotonically) {
    SKIP_WITHOUT_GPU();
    rfdetr::gpu::DeviceBuffer buffer;
    EXPECT_EQ(buffer.capacity(), 0U);
    buffer.reserve(1024);
    void *first = buffer.get();
    EXPECT_NE(first, nullptr);
    EXPECT_GE(buffer.capacity(), 1024U);

    // A smaller request must not reallocate.
    buffer.reserve(512);
    EXPECT_EQ(buffer.get(), first);
    EXPECT_GE(buffer.capacity(), 1024U);

    buffer.reserve(4096);
    EXPECT_GE(buffer.capacity(), 4096U);
}

TEST(GpuContextTest, HostDeviceRoundTrip) {
    SKIP_WITHOUT_GPU();
    rfdetr::gpu::GpuContext context(0);
    std::vector<float> source(1024);
    std::iota(source.begin(), source.end(), 1.0F);

    rfdetr::gpu::DeviceBuffer buffer(source.size() * sizeof(float));
    rfdetr::gpu::copy_h2d(buffer.get(), source.data(), source.size() * sizeof(float), context.stream());

    std::vector<float> readback(source.size(), 0.0F);
    rfdetr::gpu::copy_d2h(readback.data(), buffer.get(), readback.size() * sizeof(float), context.stream());
    rfdetr::gpu::stream_synchronize(context.stream());

    EXPECT_EQ(source, readback);
}

TEST(GpuSegPostprocessParity, MatchesCpuOnSparseDetections) {
    SKIP_WITHOUT_GPU();
    TempLabelFile labels("person\nbicycle\ncar\nmotorbike\naeroplane\n");
    const auto fixture = make_fixture(/*num_queries=*/300, /*num_classes=*/6, /*mask_h=*/108, /*mask_w=*/108,
                                      /*num_hits=*/12);
    const auto config = make_config(/*resolution=*/560, /*threshold=*/0.5F, /*mask_threshold=*/0.0F,
                                    /*max_detections=*/300);

    const auto [cpu, gpu] = run_both(fixture, config, /*orig_w=*/640, /*orig_h=*/480, labels.path());
    ASSERT_EQ(cpu.scores.size(), 12U) << "fixture did not produce the expected detections";
    expect_parity(cpu, gpu);
}

TEST(GpuSegPostprocessParity, MatchesCpuOnNonSquareLargeFrame) {
    SKIP_WITHOUT_GPU();
    TempLabelFile labels("person\nbicycle\ncar\n");
    const auto fixture = make_fixture(/*num_queries=*/300, /*num_classes=*/4, /*mask_h=*/108, /*mask_w=*/108,
                                      /*num_hits=*/8, /*seed=*/21);
    const auto config = make_config(560, 0.5F, 0.0F, 300);

    // 1080p: the resize is the expensive stage and the one most sensitive to the
    // (dst + 0.5) * scale - 0.5 sampling convention.
    const auto [cpu, gpu] = run_both(fixture, config, 1920, 1080, labels.path());
    ASSERT_FALSE(cpu.masks.empty());
    EXPECT_EQ(cpu.masks[0].width, 1920);
    EXPECT_EQ(cpu.masks[0].height, 1080);
    expect_parity(cpu, gpu);
}

TEST(GpuSegPostprocessParity, MatchesCpuOnDenseDetections) {
    SKIP_WITHOUT_GPU();
    // Dense fixture: more above-threshold candidates than the cap, so a
    // postprocessor that truncates in the wrong place — before ranking rather
    // than after — is caught. Sparse fixtures cannot distinguish the two.
    TempLabelFile labels("person\nbicycle\ncar\nmotorbike\naeroplane\nbus\ntrain\ntruck\n");
    const auto fixture = make_fixture(/*num_queries=*/300, /*num_classes=*/9, /*mask_h=*/64, /*mask_w=*/64,
                                      /*num_hits=*/300, /*seed=*/99);
    const auto config = make_config(560, 0.5F, 0.0F, /*max_detections=*/50);

    const auto [cpu, gpu] = run_both(fixture, config, 320, 240, labels.path());
    ASSERT_EQ(cpu.scores.size(), 50U) << "dense fixture must saturate max_detections";
    expect_parity(cpu, gpu);

    // The survivors must be the highest-scoring candidates, in descending order.
    EXPECT_TRUE(std::is_sorted(gpu.scores.begin(), gpu.scores.end(), std::greater<>()));
}

TEST(GpuSegPostprocessParity, MatchesCpuWithNonZeroMaskThreshold) {
    SKIP_WITHOUT_GPU();
    // mask_threshold is compared against the raw logit, not sigmoid(logit). A
    // kernel that applied a sigmoid first would agree with the CPU only at the
    // 0.0/0.5 default pair, and diverge here.
    TempLabelFile labels("person\nbicycle\ncar\n");
    const auto fixture = make_fixture(300, 4, 108, 108, 6, /*seed=*/5);
    const auto config = make_config(560, 0.5F, /*mask_threshold=*/0.75F, 300);

    const auto [cpu, gpu] = run_both(fixture, config, 800, 600, labels.path());
    ASSERT_FALSE(cpu.masks.empty());
    // Guard the guard: a threshold that admits everything or nothing would make
    // this test vacuous.
    const size_t set_pixels = rfdetr::media::count_nonzero(cpu.masks[0]);
    EXPECT_GT(set_pixels, 0U);
    EXPECT_LT(set_pixels, cpu.masks[0].data.size());
    expect_parity(cpu, gpu);
}

TEST(GpuSegPostprocessParity, HandlesNoDetections) {
    SKIP_WITHOUT_GPU();
    TempLabelFile labels("person\nbicycle\ncar\n");
    const auto fixture = make_fixture(300, 4, 108, 108, /*num_hits=*/0);
    const auto config = make_config(560, 0.5F, 0.0F, 300);

    const auto [cpu, gpu] = run_both(fixture, config, 640, 480, labels.path());
    EXPECT_TRUE(cpu.scores.empty());
    EXPECT_TRUE(gpu.scores.empty());
    EXPECT_TRUE(gpu.masks.empty());
}

TEST(GpuSegPostprocessParity, ReusesOneInstanceAcrossFrames) {
    SKIP_WITHOUT_GPU();
    // The video path calls run() repeatedly on one postprocessor. Device scratch
    // is reused, so a stale-count or stale-offset bug only shows up here.
    TempLabelFile labels("person\nbicycle\ncar\n");
    const auto config = make_config(560, 0.5F, 0.0F, 300);
    rfdetr::gpu::GpuContext context(0);

    auto backend = std::make_unique<MockDeviceBackend>(context.stream());
    const auto first = make_fixture(300, 4, 108, 108, /*num_hits=*/10, /*seed=*/3);
    backend->set_outputs(first.tensors(), first.shapes());
    backend->upload();
    RFDETRInference inference(std::move(backend), labels.path(), config);

    const float scale_w = 640.0F / 560.0F;
    const float scale_h = 480.0F / 560.0F;

    size_t previous = 0;
    for (int iteration = 0; iteration < 5; ++iteration) {
        std::vector<float> scores;
        std::vector<int> class_ids;
        std::vector<BoundingBox> boxes;
        std::vector<rfdetr::media::Mask> masks;
        inference.postprocess_segmentation_outputs_gpu(scale_w, scale_h, 480, 640, scores, class_ids, boxes, masks);
        EXPECT_EQ(scores.size(), 10U);
        EXPECT_EQ(masks.size(), 10U);
        for (const auto &mask : masks) {
            EXPECT_EQ(mask.data.size(), 640U * 480U);
        }
        if (iteration > 0) {
            EXPECT_EQ(scores.size(), previous) << "results changed across iterations";
        }
        previous = scores.size();
    }
}
