#pragma once

// Shared helpers for the GPU parity fixtures: the image/synthetic-output
// construction and the file formats the generator writes and the tests read.
//
// Everything here is CPU-only — the generator (gpu_parity_fixture_gen.cpp) and
// test_gpu_parity.cpp both include this header, so a fixture is reproduced
// bit-for-bit from its source by the same code path that produced it.

#include "mock_backend.hpp"
#include "rfdetr_inference.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace gpu_parity {

// The single resolution fixtures are stored at: 432 has a checked-in .dali
// pipeline (data/dali/), and the spec bounds storage to one resolution.
inline constexpr int kResolution = 432;
inline constexpr float kThreshold = 0.5F;
inline constexpr float kMaskThreshold = 0.0F;
inline constexpr int kMaxDetections = 300;

struct NaturalSpec {
    const char *name;
    int width;
    int height;
};

// small: upscale on both axes. wide/tall: independent scale_x / scale_y with one
// axis larger than res (a letterbox regression shows up as a horizontal/vertical
// shift, respectively).
inline constexpr std::array<NaturalSpec, 3> kNaturalFixtures = {
    {{"small", 256, 192}, {"wide", 1024, 288}, {"tall", 288, 1024}}};

// The dense fixture is synthetic-only (no image): enough above-threshold
// candidates to exceed the cap, which a truncating postprocessor would miss.
inline constexpr int kDenseQueries = 300;
inline constexpr int kDenseClasses = 9;
inline constexpr int kDenseMaskSize = 32;
inline constexpr int kDenseHits = 200;
inline constexpr uint32_t kDenseSeed = 99;
inline constexpr int kDenseOrigW = 640;
inline constexpr int kDenseOrigH = 480;

// Per-natural-fixture synthetic outputs for the end-to-end regression: small
// enough to commit, distinct enough to exercise the decode.
inline constexpr int kNaturalQueries = 12;
inline constexpr int kNaturalClasses = 4;
inline constexpr int kNaturalMaskSize = 32;
inline constexpr int kNaturalHits = 8;

struct SyntheticOutputs {
    int num_queries{0};
    int num_classes{0};
    int mask_h{0};
    int mask_w{0};
    std::vector<float> dets;   // Q*4
    std::vector<float> labels; // Q*C
    std::vector<float> masks;  // Q*H*W

    [[nodiscard]] std::vector<std::vector<float>> tensors() const { return {dets, labels, masks}; }
    [[nodiscard]] std::vector<std::vector<int64_t>> shapes() const {
        return {{1, num_queries, 4}, {1, num_queries, num_classes}, {1, num_queries, mask_h, mask_w}};
    }
};

struct Detection {
    int class_id{0};
    float score{0.0F};
    float x_min{0.0F};
    float y_min{0.0F};
    float x_max{0.0F};
    float y_max{0.0F};
};

struct Expected {
    int orig_w{0};
    int orig_h{0};
    std::vector<Detection> detections;
};

/// Deterministic smooth gradient + blob image. Low-frequency content keeps the
/// JPEG decode and resize agreement realistic: per-pixel noise would inflate the
/// nvJPEG-vs-stb decode difference and overstate the DALI-vs-CPU preprocess gap.
inline rfdetr::media::Image make_test_image(int width, int height, uint32_t seed) {
    rfdetr::media::Image image;
    image.resize(width, height);
    // Blob centres shift a little with the seed so fixtures differ, but the
    // content stays smooth (no high-frequency noise).
    const float cx1 = 0.30F + 0.05F * static_cast<float>(seed % 5);
    const float cy1 = 0.40F + 0.04F * static_cast<float>((seed / 5) % 5);
    const float cx2 = 0.70F - 0.05F * static_cast<float>(seed % 3);
    const float cy2 = 0.60F - 0.04F * static_cast<float>((seed / 3) % 3);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const float fx = static_cast<float>(x) / static_cast<float>(width);
            const float fy = static_cast<float>(y) / static_cast<float>(height);
            const float blob1 = std::exp(-180.0F * ((fx - cx1) * (fx - cx1) + (fy - cy1) * (fy - cy1)));
            const float blob2 = std::exp(-180.0F * ((fx - cx2) * (fx - cx2) + (fy - cy2) * (fy - cy2)));
            const size_t idx = (static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x)) * 3;
            image.bgr[idx + 0] = static_cast<uint8_t>(std::clamp(40.0F + 180.0F * fx + 120.0F * blob1, 0.0F, 255.0F));
            image.bgr[idx + 1] = static_cast<uint8_t>(std::clamp(40.0F + 180.0F * fy + 120.0F * blob2, 0.0F, 255.0F));
            image.bgr[idx + 2] = static_cast<uint8_t>(
                std::clamp(120.0F + 120.0F * (fx + fy) * 0.5F + 80.0F * (blob1 + blob2), 0.0F, 255.0F));
        }
    }
    return image;
}

/// Synthetic segmentation model outputs, mirroring test_gpu_postprocess.cpp's
/// fixture builder. `num_hits` queries get an above-threshold class score; the
/// rest stay at the background level.
inline SyntheticOutputs make_synthetic_outputs(int num_queries, int num_classes, int mask_h, int mask_w, int num_hits,
                                               uint32_t seed) {
    SyntheticOutputs out{num_queries, num_classes, mask_h, mask_w, {}, {}, {}};
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> unit(0.0F, 1.0F);

    out.dets.resize(static_cast<size_t>(num_queries) * 4);
    for (int q = 0; q < num_queries; ++q) {
        out.dets[static_cast<size_t>(q) * 4 + 0] = 0.15F + 0.7F * unit(rng);
        out.dets[static_cast<size_t>(q) * 4 + 1] = 0.15F + 0.7F * unit(rng);
        out.dets[static_cast<size_t>(q) * 4 + 2] = 0.05F + 0.25F * unit(rng);
        out.dets[static_cast<size_t>(q) * 4 + 3] = 0.05F + 0.25F * unit(rng);
    }

    // Background (logit 0) stays low so it never wins; the class index shifts by
    // one on the way out, so logit 1 becomes class 0.
    out.labels.assign(static_cast<size_t>(num_queries) * static_cast<size_t>(num_classes), -8.0F);
    for (int q = 0; q < num_hits && q < num_queries; ++q) {
        const int class_slot = 1 + (q % (num_classes - 1));
        out.labels[static_cast<size_t>(q) * static_cast<size_t>(num_classes) + static_cast<size_t>(class_slot)] =
            2.0F + 0.01F * static_cast<float>(num_hits - q);
    }

    out.masks.resize(static_cast<size_t>(num_queries) * static_cast<size_t>(mask_h) * static_cast<size_t>(mask_w));
    for (int q = 0; q < num_queries; ++q) {
        for (int y = 0; y < mask_h; ++y) {
            for (int x = 0; x < mask_w; ++x) {
                const float fx = static_cast<float>(x) / static_cast<float>(mask_w);
                const float fy = static_cast<float>(y) / static_cast<float>(mask_h);
                const float blob = std::exp(-12.0F * ((fx - 0.5F) * (fx - 0.5F) + (fy - 0.5F) * (fy - 0.5F)));
                const float value = 4.0F * blob - 1.5F + 0.5F * (fx - fy) + 0.1F * static_cast<float>(q % 5);
                out.masks[(static_cast<size_t>(q) * static_cast<size_t>(mask_h) + static_cast<size_t>(y)) *
                              static_cast<size_t>(mask_w) +
                          static_cast<size_t>(x)] = value;
            }
        }
    }
    return out;
}

// --- Binary IO (raw float32, little-endian) ---------------------------------

inline void write_floats(const std::filesystem::path &path, const std::vector<float> &data) {
    std::ofstream file(path, std::ios::binary);
    file.write(reinterpret_cast<const char *>(data.data()), static_cast<std::streamsize>(data.size() * sizeof(float)));
}

inline std::vector<float> read_floats(const std::filesystem::path &path, size_t count) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("missing fixture file: " + path.string());
    }
    std::vector<float> data(count);
    file.read(reinterpret_cast<char *>(data.data()), static_cast<std::streamsize>(count * sizeof(float)));
    if (!file) {
        throw std::runtime_error("short read on fixture file: " + path.string());
    }
    return data;
}

inline std::vector<float> read_preprocessed(const std::filesystem::path &path) {
    return read_floats(path, 3 * static_cast<size_t>(kResolution) * static_cast<size_t>(kResolution));
}

/// outputs.bin: four int32 header fields (Q, C, H, W) then dets, labels, masks.
inline void write_outputs(const std::filesystem::path &path, const SyntheticOutputs &outputs) {
    std::ofstream file(path, std::ios::binary);
    const std::array<std::int32_t, 4> header{outputs.num_queries, outputs.num_classes, outputs.mask_h, outputs.mask_w};
    file.write(reinterpret_cast<const char *>(header.data()), static_cast<std::streamsize>(header.size() * 4));
    for (const auto *tensor : {&outputs.dets, &outputs.labels, &outputs.masks}) {
        file.write(reinterpret_cast<const char *>(tensor->data()),
                   static_cast<std::streamsize>(tensor->size() * sizeof(float)));
    }
}

inline SyntheticOutputs read_outputs(const std::filesystem::path &path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("missing fixture file: " + path.string());
    }
    std::array<std::int32_t, 4> header{};
    file.read(reinterpret_cast<char *>(header.data()), static_cast<std::streamsize>(header.size() * 4));
    SyntheticOutputs out{header[0], header[1], header[2], header[3], {}, {}, {}};
    const size_t dets_count = static_cast<size_t>(out.num_queries) * 4;
    const size_t labels_count = static_cast<size_t>(out.num_queries) * static_cast<size_t>(out.num_classes);
    const size_t masks_count =
        static_cast<size_t>(out.num_queries) * static_cast<size_t>(out.mask_h) * static_cast<size_t>(out.mask_w);
    out.dets.resize(dets_count);
    out.labels.resize(labels_count);
    out.masks.resize(masks_count);
    for (auto *tensor : {&out.dets, &out.labels, &out.masks}) {
        file.read(reinterpret_cast<char *>(tensor->data()),
                  static_cast<std::streamsize>(tensor->size() * sizeof(float)));
    }
    if (!file) {
        throw std::runtime_error("short read on fixture file: " + path.string());
    }
    return out;
}

// --- Expected detections (text) ---------------------------------------------

inline void write_expected(const std::filesystem::path &path, const Expected &expected) {
    std::ofstream file(path);
    file << "orig_w " << expected.orig_w << "\n";
    file << "orig_h " << expected.orig_h << "\n";
    file << "count " << expected.detections.size() << "\n";
    file << std::setprecision(9);
    for (const auto &d : expected.detections) {
        file << d.class_id << ' ' << d.score << ' ' << d.x_min << ' ' << d.y_min << ' ' << d.x_max << ' ' << d.y_max
             << '\n';
    }
}

inline Expected read_expected(const std::filesystem::path &path) {
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("missing fixture file: " + path.string());
    }
    Expected expected;
    std::string key;
    file >> key >> expected.orig_w;
    file >> key >> expected.orig_h;
    size_t count = 0;
    file >> key >> count;
    expected.detections.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        Detection d;
        file >> d.class_id >> d.score >> d.x_min >> d.y_min >> d.x_max >> d.y_max;
        expected.detections.push_back(d);
    }
    return expected;
}

/// CPU preprocess of `image` at the stored resolution, byte-identical to the
/// generator's golden tensor (Config's default means/stds).
inline std::vector<float> cpu_preprocess(const rfdetr::media::Image &image) {
    const std::array<float, 3> means{0.485F, 0.456F, 0.406F};
    const std::array<float, 3> stds{0.229F, 0.224F, 0.225F};
    std::vector<float> tensor(3 * static_cast<size_t>(kResolution) * static_cast<size_t>(kResolution));
    rfdetr::media::preprocess_bgr_image(image, tensor, kResolution, means, stds);
    return tensor;
}

/// Runs the CPU segmentation postprocess over `outputs` and returns detections,
/// the same path the generator used to produce `<name>.expected.txt`.
inline std::vector<Detection> cpu_decode(const SyntheticOutputs &outputs, const std::filesystem::path &labels,
                                         int orig_w, int orig_h) {
    Config config;
    config.resolution = kResolution;
    config.threshold = kThreshold;
    config.mask_threshold = kMaskThreshold;
    config.max_detections = kMaxDetections;
    config.model_type = ModelType::SEGMENTATION;

    auto backend = std::make_unique<MockBackend>();
    backend->set_outputs(outputs.tensors(), outputs.shapes());
    RFDETRInference inference(std::move(backend), labels, config);
    inference.run_inference({});

    std::vector<float> scores;
    std::vector<int> class_ids;
    std::vector<BoundingBox> boxes;
    std::vector<rfdetr::media::Mask> masks;
    const float scale_w = static_cast<float>(orig_w) / static_cast<float>(kResolution);
    const float scale_h = static_cast<float>(orig_h) / static_cast<float>(kResolution);
    inference.postprocess_segmentation_outputs(scale_w, scale_h, orig_h, orig_w, scores, class_ids, boxes, masks);

    std::vector<Detection> detections;
    detections.reserve(scores.size());
    for (size_t i = 0; i < scores.size(); ++i) {
        detections.push_back({class_ids[i], scores[i], boxes[i].x_min, boxes[i].y_min, boxes[i].x_max, boxes[i].y_max});
    }
    return detections;
}

} // namespace gpu_parity
