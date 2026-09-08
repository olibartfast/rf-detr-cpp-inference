// End-to-end GPU parity gate: the same segmentation model run through the four
// pre/post combinations and compared against the CPU/CPU reference.
//
//   cpu-cpu        CPU preprocess + CPU postprocess   (reference)
//   gpupre-cpupost DALI preprocess + CPU postprocess  (isolates preprocess)
//   cpupre-gpupost CPU preprocess + CUDA postprocess  (isolates postprocess)
//   gpu-gpu        DALI preprocess + CUDA postprocess
//
// CUDA postprocess is bit-identical to the CPU path, so cpupre-gpupost must
// match cpu-cpu to the tight tolerance. DALI preprocess cannot bit-match the
// CPU tensor (nvJPEG vs stb decode), so the preprocess-involving combinations
// assert a looser, documented bound.
//
// Needs a real TensorRT engine at 432 or 576 (the resolutions with a checked-in
// .dali pipeline); skips otherwise. Compiled only when USE_GPU_PIPELINE is on.

#include "rfdetr_inference.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <vector>

#if defined(USE_DALI) && defined(USE_CUDA_POSTPROCESS)

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

void expect_parity(const Results &ref, const Results &other, const char *what, double score_tol, double mask_iou_min) {
    expect_same_count(ref, other, what);
    for (size_t i = 0; i < ref.scores.size(); ++i) {
        SCOPED_TRACE(std::string(what) + " detection " + std::to_string(i));
        EXPECT_EQ(ref.class_ids[i], other.class_ids[i]);
        EXPECT_NEAR(ref.scores[i], other.scores[i], score_tol);
        EXPECT_NEAR((ref.boxes[i].x_min + ref.boxes[i].x_max) / 2.0F,
                    (other.boxes[i].x_min + other.boxes[i].x_max) / 2.0F, 1.0F);
        EXPECT_NEAR((ref.boxes[i].y_min + ref.boxes[i].y_max) / 2.0F,
                    (other.boxes[i].y_min + other.boxes[i].y_max) / 2.0F, 1.0F);
        EXPECT_GE(mask_iou(ref.masks[i], other.masks[i]), mask_iou_min) << what << ": mask IoU too low";
    }
}

bool dali_pipeline_exists(int resolution) {
    return std::filesystem::exists(std::filesystem::path("data/dali") /
                                   ("preprocess_encoded_" + std::to_string(resolution) + ".dali"));
}

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
    expect_parity(cpu_cpu, cpupre_gpupost, "CUDA postprocess", /*score_tol=*/1e-3, /*mask_iou_min=*/0.999);

    // The DALI-preprocess combinations need a .dali pipeline matching the model
    // resolution; skip them rather than fail when none is checked in.
    {
        RFDETRInference probe(model, labels, make_config(false, false));
        if (!dali_pipeline_exists(probe.get_resolution())) {
            GTEST_SKIP() << "no .dali pipeline for resolution " << probe.get_resolution()
                         << "; the preprocess combinations are not runnable here";
        }
    }

    const auto gpupre_cpupost = run_combo(model, labels, image, true, false);
    const auto gpu_gpu = run_combo(model, labels, image, true, true);
    // DALI preprocess: nvJPEG vs stb decode means the tensor (and thus scores)
    // cannot bit-match the CPU path; assert the documented decode-aware bound.
    expect_parity(cpu_cpu, gpupre_cpupost, "DALI preprocess", /*score_tol=*/0.03, /*mask_iou_min=*/0.95);
    expect_parity(gpupre_cpupost, gpu_gpu, "DALI + CUDA postprocess", /*score_tol=*/1e-3, /*mask_iou_min=*/0.999);
}

#endif // USE_DALI && USE_CUDA_POSTPROCESS
