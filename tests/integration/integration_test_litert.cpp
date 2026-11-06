#include "backends/litert_backend.hpp"
#include "backends/litert_contract.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <gtest/gtest.h>

namespace {
std::vector<float> read_floats(const std::filesystem::path &path, size_t count) {
    std::vector<float> data(count);
    std::ifstream stream(path, std::ios::binary);
    if (!stream.read(reinterpret_cast<char *>(data.data()), static_cast<std::streamsize>(count * sizeof(float))) ||
        stream.peek() != std::char_traits<char>::eof()) {
        throw std::runtime_error("Invalid LiteRT reference tensor: " + path.string());
    }
    return data;
}

TEST(LiteRTIntegration, MatchesPythonInterpreterAndPreservesModelOnFailedReload) {
    const char *model = std::getenv("RFDETR_LITERT_MODEL");
    const char *reference = std::getenv("RFDETR_LITERT_REFERENCE");
    if (model == nullptr || reference == nullptr) {
        GTEST_SKIP() << "Set RFDETR_LITERT_MODEL and RFDETR_LITERT_REFERENCE (input.bin, output_0.bin, ...)";
    }
    rfdetr::backend::LiteRTBackend backend;
    const auto shape = backend.initialize(model, {1, 3, 0, 0});
    const auto input = read_floats(std::filesystem::path(reference) / "input.bin",
                                   rfdetr::backend::litert_detail::element_count(shape));
    EXPECT_FALSE(backend.supports_device_io());
    EXPECT_THROW(backend.get_output_data(0, nullptr, 0), std::runtime_error);
    EXPECT_THROW(backend.run_inference({}, shape), std::runtime_error);
    EXPECT_THROW(backend.run_inference(input, {1, 3, 1, 1}), std::runtime_error);
    for (int iteration = 0; iteration < 2; ++iteration) {
        ASSERT_EQ(backend.run_inference(input, shape).size(), backend.get_output_count());
        for (size_t i = 0; i < backend.get_output_count(); ++i) {
            const auto count = rfdetr::backend::litert_detail::element_count(backend.get_output_shape(i));
            const auto expected =
                read_floats(std::filesystem::path(reference) / ("output_" + std::to_string(i) + ".bin"), count);
            std::vector<float> actual(count);
            backend.get_output_data(i, actual.data(), count);
            const float tolerance = i == 0 ? 1e-5F : 1e-3F;
            float max_error = 0;
            for (size_t j = 0; j < count; ++j) {
                ASSERT_TRUE(std::isfinite(actual[j]));
                ASSERT_TRUE(std::isfinite(expected[j]));
                max_error = std::max(max_error, std::abs(actual[j] - expected[j]));
            }
            EXPECT_LE(max_error, tolerance) << "output " << i;
            RecordProperty("max_abs_error_output_" + std::to_string(i), std::to_string(max_error));
            EXPECT_THROW(backend.get_output_data(i, actual.data(), count - 1), std::runtime_error);
        }
        EXPECT_THROW(backend.initialize("/nonexistent/rfdetr.tflite", {1, 3, 0, 0}), std::runtime_error);
    }
    EXPECT_THROW((void)backend.get_output_shape(backend.get_output_count()), std::out_of_range);
    EXPECT_EQ(backend.initialize(model, {1, 3, 0, 0}), shape);
    EXPECT_THROW(backend.get_output_data(0, nullptr, 0), std::runtime_error);
    EXPECT_EQ(backend.run_inference(input, shape).size(), backend.get_output_count());
}
} // namespace
