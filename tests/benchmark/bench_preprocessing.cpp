#include "processing_utils.hpp"

#include <benchmark/benchmark.h>
#include <random>
#include <vector>

static void BM_Sigmoid(benchmark::State &state) {
    float x = 1.5f;
    for (auto _ : state) {
        benchmark::DoNotOptimize(rfdetr::processing::sigmoid(x));
    }
}
BENCHMARK(BM_Sigmoid);

static void BM_CxCyWhToXyxy(benchmark::State &state) {
    for (auto _ : state) {
        benchmark::DoNotOptimize(rfdetr::processing::cxcywh_to_xyxy(50.0f, 50.0f, 20.0f, 10.0f));
    }
}
BENCHMARK(BM_CxCyWhToXyxy);

static void BM_NormalizeImage(benchmark::State &state) {
    const auto res = static_cast<size_t>(state.range(0));
    const size_t channel_size = res * res;
    std::vector<float> data(3 * channel_size);

    // Fill with realistic pixel values [0, 1]
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (auto &v : data) {
        v = dist(rng);
    }

    std::array<float, 3> means = {0.485f, 0.456f, 0.406f};
    std::array<float, 3> stds = {0.229f, 0.224f, 0.225f};

    for (auto _ : state) {
        rfdetr::processing::normalize_image(data, channel_size, means, stds);
    }
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(3 * channel_size));
}
BENCHMARK(BM_NormalizeImage)->Arg(224)->Arg(560);

BENCHMARK_MAIN();
