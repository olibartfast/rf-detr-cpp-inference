// End-to-end GPU parity gate: the same segmentation model run through the four
// pre/post combinations and compared against the CPU/CPU reference.
//
//   cpu-cpu        CPU preprocess + CPU postprocess   (reference)
//   gpupre-cpupost GPU preprocess + CPU postprocess   (isolates preprocess)
//   cpupre-gpupost CPU preprocess + CUDA postprocess  (isolates postprocess)
//   gpu-gpu        GPU preprocess + CUDA postprocess
//
// "GPU preprocess" is whichever preprocessor the build selected: the CUDA kernel
// + nvJPEG (USE_CUDA_PREPROCESS) or DALI (USE_DALI).
//
// CUDA postprocess is bit-identical to the CPU path, so cpupre-gpupost must
// match cpu-cpu to the tight tolerance. No GPU preprocessor produces a
// bit-identical tensor (the CUDA kernel is within ~1e-6, a GPU JPEG decode within
// the nvJPEG/stb gap), and the engine amplifies even 1e-6, so every
// preprocess-involving combination asserts the looser, documented bound.
//
// Needs a real TensorRT engine; skips otherwise. A DALI build also needs a
// checked-in .dali pipeline at the engine's resolution (432 or 576).

#include "rfdetr_inference.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#if (defined(USE_DALI) || defined(USE_CUDA_PREPROCESS)) && defined(USE_CUDA_POSTPROCESS)

namespace {

struct Results {
    std::vector<float> scores;
    std::vector<int> class_ids;
    std::vector<BoundingBox> boxes;
    std::vector<rfdetr::media::Mask> masks;
};

std::filesystem::path resolve_model() {
    if (const char *env_path = std::getenv("RFDETR_TEST_MODEL")) {
        return env_path;
    }
    const std::filesystem::path home = std::getenv("HOME") ? std::getenv("HOME") : "";
    for (const auto &stem :
         {home / "Downloads" / "rfdetr-seg-medium", std::filesystem::path("output") / "rfdetr-seg-medium",
          home / "Downloads" / "rfdetr-medium", std::filesystem::path("output") / "rfdetr-medium"}) {
        for (const char *ext : {".engine", ".trt", ".onnx"}) {
            std::filesystem::path candidate = stem;
            candidate += ext;
            if (std::filesystem::exists(candidate)) {
                return candidate;
            }
        }
    }
    return {};
}

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
    return union_count == 0 ? 1.0 : static_cast<double>(intersection) / static_cast<double>(union_count);
}

Config make_config(bool gpu_preprocess, bool gpu_postprocess) {
    Config config;
    config.resolution = 0; // auto-detect from the model
    config.model_type = ModelType::SEGMENTATION;
    config.gpu_preprocess = gpu_preprocess;
    config.gpu_postprocess = gpu_postprocess;
    return config;
}

Results run_combo(const std::filesystem::path &model, const std::filesystem::path &labels,
                  const std::filesystem::path &image, bool gpu_pre, bool gpu_post) {
    RFDETRInference inference(model, labels, make_config(gpu_pre, gpu_post));
    const int res = inference.get_resolution();

    int orig_h = 0;
    int orig_w = 0;
    if (gpu_pre) {
        inference.run_gpu_image(image, orig_h, orig_w);
        if (!gpu_post) {
            inference.fetch_device_outputs();
        }
    } else {
        auto input = inference.preprocess_image(image, orig_h, orig_w);
        inference.run_inference(input);
    }

    Results out;
    const float scale_w = static_cast<float>(orig_w) / static_cast<float>(res);
    const float scale_h = static_cast<float>(orig_h) / static_cast<float>(res);
    if (gpu_post) {
        inference.postprocess_segmentation_outputs_gpu(scale_w, scale_h, orig_h, orig_w, out.scores, out.class_ids,
                                                       out.boxes, out.masks);
    } else {
        inference.postprocess_segmentation_outputs(scale_w, scale_h, orig_h, orig_w, out.scores, out.class_ids,
                                                   out.boxes, out.masks);
    }
    return out;
}

void expect_same_count(const Results &a, const Results &b, const char *what) {
    ASSERT_EQ(a.scores.size(), b.scores.size()) << what << ": detection counts differ";
}

/// What one comparison may differ by. Box centres are bounded in pixels of the
/// original image.
struct Tolerance {
    double score;
    float centre_px;
    double mask_iou_min;
};

/// The input tensor is bit-identical (CPU preprocess on both sides), so only the
/// postprocessor differs and it must match the CPU path almost exactly.
constexpr Tolerance kSameInput{1e-3, 1.0F, 0.999};

/// Any GPU preprocessor, any input format. The engine amplifies input noise far
/// past its size: nudging one element of the CPU tensor by 1e-6 moved this
/// engine's logits by up to 8.3 (Phase 6 validation.md), so a tensor that
/// matches the CPU to 1e-6 (the CUDA kernel), or differs by the nvJPEG/stb
/// decoder gap, cannot be held to kSameInput. Measured worst cases on data/dog.jpg
/// (768x576, rfdetr-seg-medium at 432) are in the Phase 6 validation record.
Tolerance preprocess_tolerance(int orig_w, int orig_h) {
    return {0.06, 0.01F * static_cast<float>(std::max(orig_w, orig_h)), 0.95};
}

void expect_parity(const Results &ref, const Results &other, const char *what, const Tolerance &tol) {
    expect_same_count(ref, other, what);
    double max_score = 0.0;
    float max_centre = 0.0F;
    double min_iou = 1.0;
    for (size_t i = 0; i < ref.scores.size(); ++i) {
        SCOPED_TRACE(std::string(what) + " detection " + std::to_string(i));
        const float dx = std::abs((ref.boxes[i].x_min + ref.boxes[i].x_max) / 2.0F -
                                  (other.boxes[i].x_min + other.boxes[i].x_max) / 2.0F);
        const float dy = std::abs((ref.boxes[i].y_min + ref.boxes[i].y_max) / 2.0F -
                                  (other.boxes[i].y_min + other.boxes[i].y_max) / 2.0F);
        const double iou = mask_iou(ref.masks[i], other.masks[i]);
        max_score = std::max(max_score, static_cast<double>(std::abs(ref.scores[i] - other.scores[i])));
        max_centre = std::max({max_centre, dx, dy});
        min_iou = std::min(min_iou, iou);

        EXPECT_EQ(ref.class_ids[i], other.class_ids[i]);
        EXPECT_NEAR(ref.scores[i], other.scores[i], tol.score);
        EXPECT_LE(dx, tol.centre_px);
        EXPECT_LE(dy, tol.centre_px);
        EXPECT_GE(iou, tol.mask_iou_min) << what << ": mask IoU too low";
    }
    std::cout << "[gpu-parity] " << what << ": " << ref.scores.size()
              << " detections, max |score delta| = " << max_score << ", max centre delta = " << max_centre
              << " px, min mask IoU = " << min_iou << '\n';
}

#ifdef USE_DALI
bool dali_pipeline_exists(int resolution) {
    return std::filesystem::exists(std::filesystem::path("data/dali") /
                                   ("preprocess_encoded_" + std::to_string(resolution) + ".dali"));
}
#endif

} // namespace

TEST(GpuParityIntegration, FourCombinationsAgree) {
    const auto model = resolve_model();
    if (model.empty()) {
        GTEST_SKIP() << "no TensorRT engine/ONNX model found; set RFDETR_TEST_MODEL";
    }

    const std::filesystem::path labels = "data/coco-labels-91.txt";
    const std::filesystem::path image = "data/dog.jpg";
    ASSERT_TRUE(std::filesystem::exists(labels)) << "missing " << labels;
    ASSERT_TRUE(std::filesystem::exists(image)) << "missing " << image;

    // The reference (CPU/CPU) and the postprocess check need no DALI pipeline.
    const auto cpu_cpu = run_combo(model, labels, image, false, false);
    ASSERT_GT(cpu_cpu.scores.size(), 0U) << "reference run produced no detections";

    const auto cpupre_gpupost = run_combo(model, labels, image, false, true);
    expect_parity(cpu_cpu, cpupre_gpupost, "CUDA postprocess", kSameInput);

#ifdef USE_DALI
    // The DALI-preprocess combinations need a .dali pipeline matching the model
    // resolution; skip them rather than fail when none is checked in.
    {
        RFDETRInference probe(model, labels, make_config(false, false));
        if (!dali_pipeline_exists(probe.get_resolution())) {
            GTEST_SKIP() << "no .dali pipeline for resolution " << probe.get_resolution()
                         << "; the preprocess combinations are not runnable here";
        }
    }
#endif

    const auto gpupre_cpupost = run_combo(model, labels, image, true, false);
    const auto gpu_gpu = run_combo(model, labels, image, true, true);
    // The GPU-preprocessed tensor is not bit-identical to the CPU one, which this
    // engine amplifies; gpu-gpu shares gpupre-cpupost's tensor exactly, so the
    // postprocess comparison between them stays tight.
    const auto image_size = rfdetr::media::load_image(image);
    expect_parity(cpu_cpu, gpupre_cpupost, "GPU preprocess", preprocess_tolerance(image_size.width, image_size.height));
    expect_parity(gpupre_cpupost, gpu_gpu, "GPU preprocess + CUDA postprocess", kSameInput);
}

#ifdef USE_CUDA_PREPROCESS
// The non-JPEG route through run_gpu_image(): the header probe rejects a PNG, so
// it decodes with stb exactly as the CPU path does and only the kernel differs.
// The tensor then matches to ~1e-6 (the unit tests hold it to 1e-5), but that is
// still not bit-identical, so the end-to-end bound is the preprocess one.
TEST(GpuParityIntegration, PngFallbackMatchesCpu) {
    const auto model = resolve_model();
    if (model.empty()) {
        GTEST_SKIP() << "no TensorRT engine/ONNX model found; set RFDETR_TEST_MODEL";
    }

    const std::filesystem::path labels = "data/coco-labels-91.txt";
    const auto source = rfdetr::media::load_image("data/dog.jpg");
    ASSERT_FALSE(source.empty()) << "missing data/dog.jpg";
    const auto png = std::filesystem::temp_directory_path() / "gpu_parity_integration_dog.png";
    ASSERT_TRUE(rfdetr::media::save_image(source, png));

    const auto cpu_cpu = run_combo(model, labels, png, false, false);
    const auto gpupre_cpupost = run_combo(model, labels, png, true, false);
    std::filesystem::remove(png);

    ASSERT_GT(cpu_cpu.scores.size(), 0U) << "reference run produced no detections";
    expect_parity(cpu_cpu, gpupre_cpupost, "CUDA preprocess, PNG fallback",
                  preprocess_tolerance(source.width, source.height));
}
#endif

#endif // (USE_DALI || USE_CUDA_PREPROCESS) && USE_CUDA_POSTPROCESS
