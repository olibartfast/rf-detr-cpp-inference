// Per-stage timing for the GPU pipeline, split the way the architecture splits
// it: preprocess, transfer (H2D/D2H), and segmentation postprocess. Still-image
// and video paths share these stages; video just re-runs the frame stages.
//
// The transfer + inference stage (H2D + TensorRT enqueue + D2H) needs a real
// engine and is measured on the Phase 4 exit gate, not here — this benchmark
// covers the two halves with CPU/GPU parity: preprocess and postprocess.
//
// GPU cases self-skip (State::SkipWithError) without a CUDA device, so
// -DBENCHMARKS=ON still builds and runs on a CPU-only machine.

#include "gpu_parity_fixtures.hpp"
#include "mock_backend.hpp"

#include <benchmark/benchmark.h>

#ifdef USE_DALI
#include "gpu/dali_preprocessor.hpp"
#include "gpu/gpu_context.hpp"
#endif
#ifdef USE_CUDA_PREPROCESS
#include "gpu/gpu_context.hpp"
#include "gpu/jpeg_decoder.hpp"
#include "gpu/rfdetr_preprocess.hpp"
#endif
#ifdef USE_CUDA_POSTPROCESS
#include "gpu_test_utils.hpp"
#endif

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

namespace {

#ifndef GPU_PARITY_DALI_DIR
#define GPU_PARITY_DALI_DIR "data/dali"
#endif

constexpr int kBenchResolution = 560;

rfdetr::media::Image bench_image(int width, int height) {
    // Same deterministic generator as the fixtures; geometry is what matters here.
    return gpu_parity::make_test_image(width, height, /*seed=*/42);
}

#if defined(USE_DALI) || defined(USE_CUDA_PREPROCESS)
std::vector<std::uint8_t> encode_jpeg(const rfdetr::media::Image &image) {
    const auto tmp = std::filesystem::temp_directory_path() / "bench_gpu_pipeline.jpg";
    if (!rfdetr::media::save_image(image, tmp)) {
        throw std::runtime_error("failed to write temp image");
    }
    std::ifstream file(tmp, std::ios::binary | std::ios::ate);
    const auto size = static_cast<size_t>(file.tellg());
    file.seekg(0, std::ios::beg);
    std::vector<std::uint8_t> bytes(size);
    file.read(reinterpret_cast<char *>(bytes.data()), static_cast<std::streamsize>(size));
    std::filesystem::remove(tmp);
    return bytes;
}
#endif

class TempLabelFile {
  public:
    TempLabelFile() : path_(std::filesystem::temp_directory_path() / "bench_gpu_pipeline_labels.txt") {
        std::ofstream file(path_);
        for (int i = 0; i < 9; ++i) {
            file << "class" << i << '\n';
        }
    }
    ~TempLabelFile() { std::filesystem::remove(path_); }
    [[nodiscard]] const std::filesystem::path &path() const { return path_; }

  private:
    std::filesystem::path path_;
};

// --- Preprocess stage --------------------------------------------------------

static void BM_CpuPreprocess(benchmark::State &state) {
    const int res = static_cast<int>(state.range(0));
    const auto image = bench_image(1280, 720);
    const std::array<float, 3> means{0.485F, 0.456F, 0.406F};
    const std::array<float, 3> stds{0.229F, 0.224F, 0.225F};
    std::vector<float> tensor(3 * static_cast<size_t>(res) * static_cast<size_t>(res));
    for (auto _ : state) {
        rfdetr::media::preprocess_bgr_image(image, tensor, res, means, stds);
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_CpuPreprocess)->Arg(432)->Arg(560)->Arg(576);

// Still-image CPU path: stb decode of a JPEG on disk, then the CPU preprocess —
// the work `BM_CudaPreprocessEncoded` replaces.
static void BM_CpuPreprocessEncoded(benchmark::State &state) {
    const int res = static_cast<int>(state.range(0));
    const auto path = std::filesystem::temp_directory_path() / "bench_cpu_preprocess_encoded.jpg";
    if (!rfdetr::media::save_image(bench_image(1280, 720), path)) {
        state.SkipWithError("failed to write temp image");
        return;
    }
    const std::array<float, 3> means{0.485F, 0.456F, 0.406F};
    const std::array<float, 3> stds{0.229F, 0.224F, 0.225F};
    std::vector<float> tensor(3 * static_cast<size_t>(res) * static_cast<size_t>(res));
    for (auto _ : state) {
        const auto image = rfdetr::media::load_image(path);
        rfdetr::media::preprocess_bgr_image(image, tensor, res, means, stds);
    }
    std::filesystem::remove(path);
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_CpuPreprocessEncoded)->Arg(432)->Arg(576);

#ifdef USE_DALI
static void BM_DaliPreprocess(benchmark::State &state) {
    if (!rfdetr::gpu::device_available()) {
        state.SkipWithError("no CUDA device available");
        return;
    }
    const auto image = bench_image(1280, 720);
    const auto bytes = encode_jpeg(image);
    const int res = static_cast<int>(state.range(0));
    const auto pipeline =
        std::filesystem::path(GPU_PARITY_DALI_DIR) / ("preprocess_encoded_" + std::to_string(res) + ".dali");

    rfdetr::gpu::GpuContext context(0);
    rfdetr::gpu::DaliPreprocessor preprocessor(pipeline, rfdetr::gpu::DaliPreprocessor::Source::EncodedImage, 0);
    const size_t tensor_bytes = 3 * static_cast<size_t>(res) * static_cast<size_t>(res) * sizeof(float);
    rfdetr::gpu::DeviceBuffer buffer(tensor_bytes);

    for (auto _ : state) {
        preprocessor.process_encoded(bytes, buffer.get(), tensor_bytes, context.stream());
        rfdetr::gpu::stream_synchronize(context.stream());
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_DaliPreprocess)->Arg(432)->Arg(576);

static void BM_DaliPreprocessFrame(benchmark::State &state) {
    if (!rfdetr::gpu::device_available()) {
        state.SkipWithError("no CUDA device available");
        return;
    }
    const auto image = bench_image(1280, 720);
    const int res = static_cast<int>(state.range(0));
    const auto pipeline =
        std::filesystem::path(GPU_PARITY_DALI_DIR) / ("preprocess_frame_" + std::to_string(res) + ".dali");

    rfdetr::gpu::GpuContext context(0);
    rfdetr::gpu::DaliPreprocessor preprocessor(pipeline, rfdetr::gpu::DaliPreprocessor::Source::BgrFrame, 0);
    const size_t tensor_bytes = 3 * static_cast<size_t>(res) * static_cast<size_t>(res) * sizeof(float);
    rfdetr::gpu::DeviceBuffer frame(image.bytes());
    rfdetr::gpu::DeviceBuffer buffer(tensor_bytes);

    for (auto _ : state) {
        rfdetr::gpu::copy_h2d(frame.get(), image.data(), image.bytes(), context.stream());
        preprocessor.process_frame(frame.get(), image.height, image.width, buffer.get(), tensor_bytes,
                                   context.stream());
        rfdetr::gpu::stream_synchronize(context.stream());
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_DaliPreprocessFrame)->Arg(432)->Arg(576);
#endif // USE_DALI

#ifdef USE_CUDA_PREPROCESS
// Video-frame path: H2D of the decoded BGR frame, then the fused kernel.
static void BM_CudaPreprocessFrame(benchmark::State &state) {
    if (!rfdetr::gpu::device_available()) {
        state.SkipWithError("no CUDA device available");
        return;
    }
    const auto image = bench_image(1280, 720);
    const int res = static_cast<int>(state.range(0));
    const std::array<float, 3> means{0.485F, 0.456F, 0.406F};
    const std::array<float, 3> stds{0.229F, 0.224F, 0.225F};

    rfdetr::gpu::GpuContext context(0);
    rfdetr::gpu::DeviceBuffer frame(image.bytes());
    rfdetr::gpu::DeviceBuffer buffer(3 * static_cast<size_t>(res) * static_cast<size_t>(res) * sizeof(float));

    for (auto _ : state) {
        rfdetr::gpu::copy_h2d(frame.get(), image.data(), image.bytes(), context.stream());
        rfdetr::gpu::preprocess_bgr_device(static_cast<const std::uint8_t *>(frame.get()), image.height, image.width,
                                           buffer.get(), res, means, stds, context.stream());
        rfdetr::gpu::stream_synchronize(context.stream());
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_CudaPreprocessFrame)->Arg(432)->Arg(576);

// The CPU preprocess as the TensorRT backend consumes it: the float tensor still
// has to reach the input binding, so the H2D copy is part of the CPU path's cost.
static void BM_CpuPreprocessUpload(benchmark::State &state) {
    if (!rfdetr::gpu::device_available()) {
        state.SkipWithError("no CUDA device available");
        return;
    }
    const auto image = bench_image(1280, 720);
    const int res = static_cast<int>(state.range(0));
    const std::array<float, 3> means{0.485F, 0.456F, 0.406F};
    const std::array<float, 3> stds{0.229F, 0.224F, 0.225F};
    std::vector<float> tensor(3 * static_cast<size_t>(res) * static_cast<size_t>(res));

    rfdetr::gpu::GpuContext context(0);
    rfdetr::gpu::DeviceBuffer buffer(tensor.size() * sizeof(float));

    for (auto _ : state) {
        rfdetr::media::preprocess_bgr_image(image, tensor, res, means, stds);
        rfdetr::gpu::copy_h2d(buffer.get(), tensor.data(), tensor.size() * sizeof(float), context.stream());
        rfdetr::gpu::stream_synchronize(context.stream());
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_CpuPreprocessUpload)->Arg(432)->Arg(576);

// Still-image path: nvJPEG decode of the encoded bytes, then the same kernel.
static void BM_CudaPreprocessEncoded(benchmark::State &state) {
    if (!rfdetr::gpu::device_available()) {
        state.SkipWithError("no CUDA device available");
        return;
    }
    const auto bytes = encode_jpeg(bench_image(1280, 720));
    const int res = static_cast<int>(state.range(0));
    const std::array<float, 3> means{0.485F, 0.456F, 0.406F};
    const std::array<float, 3> stds{0.229F, 0.224F, 0.225F};

    rfdetr::gpu::GpuContext context(0);
    rfdetr::gpu::JpegDecoder decoder;
    rfdetr::gpu::DeviceBuffer frame;
    rfdetr::gpu::DeviceBuffer buffer(3 * static_cast<size_t>(res) * static_cast<size_t>(res) * sizeof(float));

    for (auto _ : state) {
        const auto size = decoder.decode(bytes, frame, context.stream());
        rfdetr::gpu::preprocess_bgr_device(static_cast<const std::uint8_t *>(frame.get()), size.height, size.width,
                                           buffer.get(), res, means, stds, context.stream());
        rfdetr::gpu::stream_synchronize(context.stream());
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_CudaPreprocessEncoded)->Arg(432)->Arg(576);
#endif // USE_CUDA_PREPROCESS

// --- Transfer stage (proxy for H2D + D2H) ------------------------------------

#ifdef USE_CUDA_POSTPROCESS
static void BM_TransferRoundTrip(benchmark::State &state) {
    if (!rfdetr::gpu::device_available()) {
        state.SkipWithError("no CUDA device available");
        return;
    }
    // 3.7 MB: the preprocessed float tensor the CPU path would upload.
    const size_t bytes = 3 * 560 * 560 * sizeof(float);
    std::vector<float> src(bytes / sizeof(float), 1.0F);
    std::vector<float> dst(bytes / sizeof(float), 0.0F);

    rfdetr::gpu::GpuContext context(0);
    rfdetr::gpu::DeviceBuffer buffer(bytes);

    for (auto _ : state) {
        rfdetr::gpu::copy_h2d(buffer.get(), src.data(), bytes, context.stream());
        rfdetr::gpu::copy_d2h(dst.data(), buffer.get(), bytes, context.stream());
        rfdetr::gpu::stream_synchronize(context.stream());
    }
    state.SetBytesProcessed(static_cast<int64_t>(bytes) * state.iterations() * 2);
}
BENCHMARK(BM_TransferRoundTrip);
#endif // USE_CUDA_POSTPROCESS

// --- Postprocess stage -------------------------------------------------------

gpu_parity::SyntheticOutputs dense_outputs() {
    return gpu_parity::make_synthetic_outputs(/*num_queries=*/300, /*num_classes=*/9, /*mask_h=*/64, /*mask_w=*/64,
                                              /*num_hits=*/120, /*seed=*/7);
}

static void BM_CpuSegPostprocess(benchmark::State &state) {
    TempLabelFile labels;
    const auto outputs = dense_outputs();
    const int orig_w = 1920;
    const int orig_h = 1080;

    Config config;
    config.resolution = kBenchResolution;
    config.threshold = 0.5F;
    config.mask_threshold = 0.0F;
    config.max_detections = 300;
    config.model_type = ModelType::SEGMENTATION;

    auto backend = std::make_unique<MockBackend>();
    backend->set_outputs(outputs.tensors(), outputs.shapes());
    RFDETRInference inference(std::move(backend), labels.path(), config);
    inference.run_inference({});

    const float scale_w = static_cast<float>(orig_w) / static_cast<float>(kBenchResolution);
    const float scale_h = static_cast<float>(orig_h) / static_cast<float>(kBenchResolution);
    for (auto _ : state) {
        std::vector<float> scores;
        std::vector<int> class_ids;
        std::vector<BoundingBox> boxes;
        std::vector<rfdetr::media::Mask> masks;
        inference.postprocess_segmentation_outputs(scale_w, scale_h, orig_h, orig_w, scores, class_ids, boxes, masks);
        benchmark::DoNotOptimize(scores.size());
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_CpuSegPostprocess);

#ifdef USE_CUDA_POSTPROCESS
static void BM_GpuSegPostprocess(benchmark::State &state) {
    if (!rfdetr::gpu::device_available()) {
        state.SkipWithError("no CUDA device available");
        return;
    }
    TempLabelFile labels;
    const auto outputs = dense_outputs();
    const int orig_w = 1920;
    const int orig_h = 1080;

    Config config;
    config.resolution = kBenchResolution;
    config.threshold = 0.5F;
    config.mask_threshold = 0.0F;
    config.max_detections = 300;
    config.model_type = ModelType::SEGMENTATION;
    config.gpu_postprocess = true;

    rfdetr::gpu::GpuContext context(0);
    auto backend = std::make_unique<MockDeviceBackend>(context.stream());
    backend->set_outputs(outputs.tensors(), outputs.shapes());
    backend->upload();
    RFDETRInference inference(std::move(backend), labels.path(), config);

    const float scale_w = static_cast<float>(orig_w) / static_cast<float>(kBenchResolution);
    const float scale_h = static_cast<float>(orig_h) / static_cast<float>(kBenchResolution);
    for (auto _ : state) {
        std::vector<float> scores;
        std::vector<int> class_ids;
        std::vector<BoundingBox> boxes;
        std::vector<rfdetr::media::Mask> masks;
        inference.postprocess_segmentation_outputs_gpu(scale_w, scale_h, orig_h, orig_w, scores, class_ids, boxes,
                                                       masks);
        benchmark::DoNotOptimize(scores.size());
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_GpuSegPostprocess);
#endif // USE_CUDA_POSTPROCESS

} // namespace
