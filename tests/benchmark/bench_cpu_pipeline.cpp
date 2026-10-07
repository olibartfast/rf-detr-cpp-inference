// CPU decode-path benchmarks: preprocessing, class-layout scoring, top-k selection, and
// annotation drawing. All input data is synthesized in-process with a seeded std::mt19937 so
// this binary needs no fixture files and runs from any working directory. See
// bench_gpu_pipeline.cpp for the GPU-side counterparts and bench_preprocessing.cpp for the
// smaller primitive-level cases these build on.

#include "media.hpp"
#include "processing_utils.hpp"

#include <algorithm>
#include <array>
#include <benchmark/benchmark.h>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <span>
#include <vector>

namespace {

// Competing implementation of select_topk_multiclass's selection, kept local to this benchmark
// (never in src/): a bounded max-size-k heap instead of the shipped std::iota + full-size
// std::partial_sort. `result` doubles as scratch storage and output, reused across calls -
// but select_topk_multiclass itself returns by value and allocates a fresh n-sized index
// vector on every call, so BM_SelectTopkShipped cannot reuse a buffer the way this arm does.
// The comparison below therefore measures algorithm plus allocation for the shipped function,
// not the selection algorithm alone.
void select_topk_bounded_heap(std::span<const float> scores, size_t num_select, std::vector<size_t> &result) {
    const size_t count = std::min(num_select, scores.size());
    result.clear();
    if (count == 0) {
        return;
    }
    result.reserve(count);

    // Mirrors select_topk_multiclass's rank key: NaN ranks ahead of every finite score.
    const auto rank_key = [scores](size_t index) {
        const float score = scores[index];
        return std::isnan(score) ? std::numeric_limits<float>::infinity() : score;
    };
    // Shipped ordering: descending score, then ascending flattened index.
    const auto better = [&rank_key](size_t lhs, size_t rhs) {
        const float lhs_key = rank_key(lhs);
        const float rhs_key = rank_key(rhs);
        return lhs_key != rhs_key ? lhs_key > rhs_key : lhs < rhs;
    };
    // std::make_heap's max-under-comparator lands at front(): passing `better` directly makes
    // that "maximum" the *worst* kept candidate (better items are heap-smaller), which is what we
    // want to compare newcomers against and evict.
    for (size_t index = 0; index < scores.size(); ++index) {
        if (result.size() < count) {
            result.push_back(index);
            if (result.size() == count) {
                std::make_heap(result.begin(), result.end(), better);
            }
        } else if (better(index, result.front())) {
            std::pop_heap(result.begin(), result.end(), better);
            result.back() = index;
            std::push_heap(result.begin(), result.end(), better);
        }
    }
    std::sort(result.begin(), result.end(), better);
}

rfdetr::media::Image synthesize_image(int width, int height, unsigned seed) {
    rfdetr::media::Image image;
    image.resize(width, height);
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> pixel_dist(0, 255);
    for (auto &byte : image.bgr) {
        byte = static_cast<uint8_t>(pixel_dist(rng));
    }
    return image;
}

std::vector<float> synthesize_scores(size_t count, unsigned seed) {
    std::vector<float> scores(count);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (auto &value : scores) {
        value = dist(rng);
    }
    return scores;
}

// Shared score grid for both top-k arms: same seed, same content, no NaN. A comparison between
// two implementations means nothing if they are not timed on the same input, so both
// BM_SelectTopkShipped and BM_SelectTopkBoundedHeap must build their timed data through this one
// helper rather than synthesizing independently.
std::vector<float> synthesize_topk_scores(int num_queries, int num_classes) {
    return synthesize_scores(static_cast<size_t>(num_queries) * static_cast<size_t>(num_classes), /*seed=*/13);
}

// --- Preprocessing ------------------------------------------------------------

void BM_PreprocessBgrImage(benchmark::State &state) {
    const int res = static_cast<int>(state.range(0));
    constexpr int kWidth = 768;
    constexpr int kHeight = 576;

    const rfdetr::media::Image image = synthesize_image(kWidth, kHeight, /*seed=*/7);
    const std::array<float, 3> means{0.485f, 0.456f, 0.406f};
    const std::array<float, 3> stds{0.229f, 0.224f, 0.225f};
    std::vector<float> output(3 * static_cast<size_t>(res) * static_cast<size_t>(res));

    for (auto _ : state) {
        rfdetr::media::preprocess_bgr_image(image, output, res, means, stds);
        benchmark::DoNotOptimize(output.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(output.size()));
}
BENCHMARK(BM_PreprocessBgrImage)->Arg(432)->Arg(576)->Arg(640);

// --- Class-layout scoring ------------------------------------------------------

void BM_BuildForegroundScores(benchmark::State &state) {
    const auto num_queries = static_cast<int>(state.range(0));
    const auto num_classes = static_cast<int>(state.range(1));
    // Shipped default (Config::background_class_id{0}, see src/rfdetr_inference.hpp): background
    // is logit 0, so slot_for_foreground_column shifts every foreground column by +1. That is the
    // offset read pattern production actually performs; slot num_classes - 1 would shift nothing
    // and measure a layout the application never takes.
    const int background_slot = 0;

    const auto logits = synthesize_scores(static_cast<size_t>(num_queries) * static_cast<size_t>(num_classes),
                                          /*seed=*/11);

    std::vector<float> scores; // Reused across iterations, as the shipped caller does.
    for (auto _ : state) {
        rfdetr::processing::build_foreground_scores(logits, num_queries, num_classes, background_slot, scores);
        benchmark::DoNotOptimize(scores.data());
        benchmark::ClobberMemory();
    }
    const int num_foreground = rfdetr::processing::foreground_class_count(num_classes, background_slot);
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(num_queries) *
                            static_cast<int64_t>(num_foreground));
}
BENCHMARK(BM_BuildForegroundScores)->Args({100, 91})->Args({300, 91});

// --- Top-k selection ------------------------------------------------------------

void BM_SelectTopkShipped(benchmark::State &state) {
    const auto num_queries = static_cast<int>(state.range(0));
    const auto num_classes = static_cast<int>(state.range(1));
    constexpr size_t kNumSelect = 300;

    const auto scores = synthesize_topk_scores(num_queries, num_classes);

    for (auto _ : state) {
        auto order = rfdetr::processing::select_topk_multiclass(scores, kNumSelect);
        benchmark::DoNotOptimize(order.data());
        benchmark::ClobberMemory();
    }
    // Reports the size of the grid actually scanned, not just the kNumSelect kept, since the
    // selection work is a function of the former.
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(scores.size()));
}
BENCHMARK(BM_SelectTopkShipped)->Args({100, 91})->Args({300, 91});

void BM_SelectTopkBoundedHeap(benchmark::State &state) {
    const auto num_queries = static_cast<int>(state.range(0));
    const auto num_classes = static_cast<int>(state.range(1));
    constexpr size_t kNumSelect = 300;

    // Correctness check, run once and untimed: NaN must rank ahead of every finite score under
    // both implementations. This uses its own explicitly NaN-containing input, built separately
    // from the shared timed grid below, so the NaN branch is verified without ever entering
    // either arm's timed measurement.
    {
        auto nan_scores = synthesize_topk_scores(num_queries, num_classes);
        if (!nan_scores.empty()) {
            nan_scores[nan_scores.size() / 2] = std::numeric_limits<float>::quiet_NaN();
        }
        std::vector<size_t> nan_result;
        select_topk_bounded_heap(nan_scores, kNumSelect, nan_result);
        const auto expected = rfdetr::processing::select_topk_multiclass(nan_scores, kNumSelect);
        if (nan_result != expected) {
            state.SkipWithError("select_topk_bounded_heap does not reproduce select_topk_multiclass's ordering");
            return;
        }
    }

    // Same seed, same content, same kNumSelect as BM_SelectTopkShipped (via
    // synthesize_topk_scores) and no NaN, so the two arms are timed on identical input.
    const auto scores = synthesize_topk_scores(num_queries, num_classes);
    std::vector<size_t> result;
    for (auto _ : state) {
        select_topk_bounded_heap(scores, kNumSelect, result);
        benchmark::DoNotOptimize(result.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(scores.size()));
}
BENCHMARK(BM_SelectTopkBoundedHeap)->Args({100, 91})->Args({300, 91});

// --- Annotation drawing ---------------------------------------------------------

void BM_DrawDetections(benchmark::State &state) {
    const int num_boxes = static_cast<int>(state.range(0));
    constexpr int kWidth = 768;
    constexpr int kHeight = 576;

    rfdetr::media::Image image = synthesize_image(kWidth, kHeight, /*seed=*/23);

    std::mt19937 rng(29);
    std::uniform_real_distribution<float> x_dist(0.0f, static_cast<float>(kWidth - 21));
    std::uniform_real_distribution<float> y_dist(0.0f, static_cast<float>(kHeight - 21));
    std::uniform_int_distribution<int> class_dist(0, 79);

    std::vector<BoundingBox> boxes(static_cast<size_t>(num_boxes));
    std::vector<int> class_ids(static_cast<size_t>(num_boxes));
    for (int i = 0; i < num_boxes; ++i) {
        const float x0 = x_dist(rng);
        const float y0 = y_dist(rng);
        boxes[static_cast<size_t>(i)] = BoundingBox{x0, y0, x0 + 20.0f, y0 + 20.0f};
        class_ids[static_cast<size_t>(i)] = class_dist(rng);
    }

    // draw_detections only writes pixels covered by the given boxes; it never reads existing
    // buffer content first, so its cost does not depend on what was drawn on a prior iteration.
    // No per-iteration restore (and no PauseTiming/ResumeTiming, whose overhead Google Benchmark
    // itself documents as high relative to a cheap per-iteration body) is needed: drawing
    // repeatedly into one reused image measures the same cost a fresh buffer would.
    for (auto _ : state) {
        rfdetr::media::draw_detections(image, boxes, class_ids);
        benchmark::DoNotOptimize(image.bgr.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(num_boxes));
}
BENCHMARK(BM_DrawDetections)->Arg(1)->Arg(10)->Arg(100);

} // namespace
