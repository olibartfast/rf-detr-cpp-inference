#include "backends/litert_contract.hpp"

#ifdef USE_LITERT
#include "backends/inference_backend.hpp"
#endif

#include <gtest/gtest.h>
#include <limits>
#include <stdexcept>

namespace {
using rfdetr::backend::litert_detail::element_count;
using rfdetr::backend::litert_detail::validate_input;
using rfdetr::backend::litert_detail::validate_outputs;

TEST(LiteRTContract, AcceptsFixedNchwAndAutoResolution) {
    EXPECT_NO_THROW(validate_input({1, 3, 384, 384}, {1, 3, 0, 0}));
    EXPECT_NO_THROW(validate_input({1, 3, 384, 384}, {1, 3, 384, 384}));
    EXPECT_EQ(element_count({1, 3, 384, 384}), 442368U);
}

TEST(LiteRTContract, RejectsUnsupportedInputLayoutsAndResolutionMismatch) {
    EXPECT_THROW(validate_input({1, 384, 384, 3}, {1, 3, 0, 0}), std::runtime_error);
    EXPECT_THROW(validate_input({2, 3, 384, 384}, {1, 3, 0, 0}), std::runtime_error);
    EXPECT_THROW(validate_input({1, 3, 384, 512}, {1, 3, 0, 0}), std::runtime_error);
    EXPECT_THROW(validate_input({1, 3, 384, 384}, {1, 3, 640, 640}), std::runtime_error);
    EXPECT_THROW(validate_input({1, 3, 384, 384}, {}), std::runtime_error);
}

TEST(LiteRTContract, RejectsDynamicEmptyAndOverflowingTensors) {
    EXPECT_THROW(element_count({}), std::runtime_error);
    EXPECT_THROW(element_count({1, -1, 4}), std::runtime_error);
    EXPECT_THROW(element_count({1, 0, 4}), std::runtime_error);
    EXPECT_THROW(element_count({std::numeric_limits<int64_t>::max(), 4}), std::runtime_error);
}

TEST(LiteRTContract, AcceptsAllThreeTasksWithVariableQueryCounts) {
    EXPECT_NO_THROW(validate_outputs({{1, 300, 4}, {1, 300, 91}}));
    EXPECT_NO_THROW(validate_outputs({{1, 100, 4}, {1, 100, 2}, {1, 100, 34, 8}}));
    EXPECT_NO_THROW(validate_outputs({{1, 100, 4}, {1, 100, 2}, {1, 100, 96, 96}}));
}

TEST(LiteRTContract, RejectsWrongOrderQueryMismatchAndInvalidMasks) {
    EXPECT_THROW(validate_outputs({{1, 300, 91}, {1, 300, 4}}), std::runtime_error);
    EXPECT_THROW(validate_outputs({{1, 300, 4}, {1, 100, 91}}), std::runtime_error);
    EXPECT_THROW(validate_outputs({{1, 300, 4}, {1, 300, 91}, {1, 96, 96, 300}}), std::runtime_error);
    EXPECT_THROW(validate_outputs({{1, 300, 4}}), std::runtime_error);
    EXPECT_THROW(validate_outputs({{1, 300, 4}, {1, 300, 91}, {1, 300, 0, 96}}), std::runtime_error);
}
} // namespace

#ifdef USE_LITERT
TEST(LiteRTBackend, RejectsUninitializedUseAndFailedLoadLeavesItEmpty) {
    auto backend = rfdetr::backend::create_backend();
    EXPECT_EQ(backend->get_backend_name(), "LiteRT");
    EXPECT_FALSE(backend->supports_device_io());
    EXPECT_EQ(backend->get_output_count(), 0U);
    EXPECT_THROW(backend->run_inference({}, {1, 3, 384, 384}), std::runtime_error);
    EXPECT_THROW((void)backend->get_output_shape(0), std::runtime_error);
    EXPECT_THROW(backend->get_output_data(0, nullptr, 0), std::runtime_error);
    EXPECT_THROW(backend->initialize("/nonexistent/rfdetr.tflite", {1, 3, 0, 0}), std::runtime_error);
    EXPECT_EQ(backend->get_output_count(), 0U);
}
#endif
