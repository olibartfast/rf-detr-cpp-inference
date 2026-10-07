#include "cli_options.hpp"

#include <cmath>
#include <gtest/gtest.h>
#include <span>
#include <vector>

namespace {

rfdetr::cli::ParseOutcome run(std::initializer_list<const char *> extra) {
    std::vector<const char *> argv = {"app", "m.onnx", "in.jpg", "labels.txt"};
    for (const char *a : extra) {
        argv.push_back(a);
    }
    return rfdetr::cli::parse_cli(std::span<const char *const>(argv.data(), argv.size()));
}

} // namespace

using rfdetr::cli::ParseStatus;

// --- Usage / defaults / basic argument plumbing ---------------------------

TEST(CliOptionsBasics, TooFewArgsShowsUsage) {
    std::vector<const char *> argv = {"app", "m.onnx", "in.jpg"};
    const auto outcome = rfdetr::cli::parse_cli(std::span<const char *const>(argv.data(), argv.size()));
    EXPECT_EQ(outcome.status, ParseStatus::ShowUsage);
}

TEST(CliOptionsBasics, DefaultsAreUnset) {
    const auto outcome = run({});
    ASSERT_EQ(outcome.status, ParseStatus::Ok);
    const auto &opts = outcome.options;
    EXPECT_EQ(opts.dali_pipeline_dir, std::filesystem::path("data/dali"));
    EXPECT_FALSE(opts.segmentation);
    EXPECT_FALSE(opts.keypoint);
    EXPECT_FALSE(opts.display);
    EXPECT_FALSE(opts.gpu_preprocess);
    EXPECT_FALSE(opts.gpu_postprocess);
    EXPECT_FALSE(opts.resolution.has_value());
    EXPECT_FALSE(opts.max_detections.has_value());
    EXPECT_FALSE(opts.threshold.has_value());
    EXPECT_FALSE(opts.mask_threshold.has_value());
    EXPECT_FALSE(opts.background_class_id_given);
    EXPECT_FALSE(opts.background_class_id.has_value());
    EXPECT_FALSE(opts.keypoint_counts.has_value());
    EXPECT_FALSE(opts.output_path.has_value());
}

TEST(CliOptionsBasics, PositionalPathsAreTakenFromArgs1To3) {
    const auto outcome = run({});
    ASSERT_EQ(outcome.status, ParseStatus::Ok);
    EXPECT_EQ(outcome.options.model_path, std::filesystem::path("m.onnx"));
    EXPECT_EQ(outcome.options.input_path, std::filesystem::path("in.jpg"));
    EXPECT_EQ(outcome.options.label_path, std::filesystem::path("labels.txt"));
}

TEST(CliOptionsBasics, UnknownFlagIsIgnored) {
    const auto outcome = run({"--bogus"});
    EXPECT_EQ(outcome.status, ParseStatus::Ok);
}

TEST(CliOptionsBasics, TrailingValueFlagWithNoValueIsIgnored) {
    const auto outcome = run({"--threshold"});
    ASSERT_EQ(outcome.status, ParseStatus::Ok);
    EXPECT_FALSE(outcome.options.threshold.has_value());
}

TEST(CliOptionsBasics, SimpleBooleanFlagsSetTheirFields) {
    const auto outcome = run({"--segmentation", "--keypoint", "--display"});
    ASSERT_EQ(outcome.status, ParseStatus::Ok);
    EXPECT_TRUE(outcome.options.segmentation);
    EXPECT_TRUE(outcome.options.keypoint);
    EXPECT_TRUE(outcome.options.display);
}

#if defined(USE_CUDA_PREPROCESS) || defined(USE_DALI)
TEST(CliOptionsBasics, GpuPreprocessFlagSetsField) {
    const auto outcome = run({"--gpu-preprocess"});
    ASSERT_EQ(outcome.status, ParseStatus::Ok);
    EXPECT_TRUE(outcome.options.gpu_preprocess);
}
#endif

#if defined(USE_CUDA_POSTPROCESS)
TEST(CliOptionsBasics, GpuPostprocessFlagSetsField) {
    const auto outcome = run({"--segmentation", "--gpu-postprocess"});
    ASSERT_EQ(outcome.status, ParseStatus::Ok);
    EXPECT_TRUE(outcome.options.gpu_postprocess);
}
#endif

TEST(CliOptionsBasics, OutputAndDaliPipelineDirSetPaths) {
    const auto outcome = run({"--output", "out.jpg", "--dali-pipeline-dir", "mydir"});
    ASSERT_EQ(outcome.status, ParseStatus::Ok);
    ASSERT_TRUE(outcome.options.output_path.has_value());
    EXPECT_EQ(*outcome.options.output_path, std::filesystem::path("out.jpg"));
    EXPECT_EQ(outcome.options.dali_pipeline_dir, std::filesystem::path("mydir"));
}

// --- --threshold ------------------------------------------------------------

TEST(CliOptionsThreshold, LeadingSpaceOutOfRange) {
    const auto outcome = run({"--threshold", " 5"});
    ASSERT_EQ(outcome.status, ParseStatus::Error);
    EXPECT_EQ(outcome.error, "Error: --threshold must be in [0, 1], got 5");
}

TEST(CliOptionsThreshold, LeadingPlusOutOfRange) {
    const auto outcome = run({"--threshold", "+5"});
    ASSERT_EQ(outcome.status, ParseStatus::Error);
    EXPECT_EQ(outcome.error, "Error: --threshold must be in [0, 1], got 5");
}

TEST(CliOptionsThreshold, HexLiteralOutOfRange) {
    const auto outcome = run({"--threshold", "0x10"});
    ASSERT_EQ(outcome.status, ParseStatus::Error);
    EXPECT_EQ(outcome.error, "Error: --threshold must be in [0, 1], got 16");
}

TEST(CliOptionsThreshold, ScientificNotationOk) {
    const auto outcome = run({"--threshold", "1e-1"});
    ASSERT_EQ(outcome.status, ParseStatus::Ok);
    ASSERT_TRUE(outcome.options.threshold.has_value());
    EXPECT_FLOAT_EQ(*outcome.options.threshold, 0.1f);
}

TEST(CliOptionsThreshold, InfOutOfRange) {
    const auto outcome = run({"--threshold", "inf"});
    ASSERT_EQ(outcome.status, ParseStatus::Error);
    EXPECT_EQ(outcome.error, "Error: --threshold must be in [0, 1], got inf");
}

TEST(CliOptionsThreshold, NanIsOk) {
    const auto outcome = run({"--threshold", "nan"});
    ASSERT_EQ(outcome.status, ParseStatus::Ok);
    ASSERT_TRUE(outcome.options.threshold.has_value());
    EXPECT_TRUE(std::isnan(*outcome.options.threshold));
}

TEST(CliOptionsThreshold, EmptyStringIsNotANumber) {
    const auto outcome = run({"--threshold", ""});
    ASSERT_EQ(outcome.status, ParseStatus::Error);
    EXPECT_EQ(outcome.error, "Error: --threshold expects a number, got ''");
}

TEST(CliOptionsThreshold, TrailingSpaceIsNotFullyConsumed) {
    const auto outcome = run({"--threshold", "5 "});
    ASSERT_EQ(outcome.status, ParseStatus::Error);
    EXPECT_EQ(outcome.error, "Error: --threshold expects a number, got '5 '");
}

TEST(CliOptionsThreshold, NegativeZeroIsOk) {
    const auto outcome = run({"--threshold", "-0"});
    ASSERT_EQ(outcome.status, ParseStatus::Ok);
    ASSERT_TRUE(outcome.options.threshold.has_value());
}

TEST(CliOptionsThreshold, HalfIsOk) {
    const auto outcome = run({"--threshold", "0.5"});
    ASSERT_EQ(outcome.status, ParseStatus::Ok);
    ASSERT_TRUE(outcome.options.threshold.has_value());
    EXPECT_FLOAT_EQ(*outcome.options.threshold, 0.5f);
}

TEST(CliOptionsThreshold, AboveOneIsOutOfRange) {
    const auto outcome = run({"--threshold", "1.5"});
    ASSERT_EQ(outcome.status, ParseStatus::Error);
    EXPECT_EQ(outcome.error, "Error: --threshold must be in [0, 1], got 1.5");
}

// --- --max-detections --------------------------------------------------------

TEST(CliOptionsMaxDetections, LeadingSpaceOk) {
    const auto outcome = run({"--max-detections", " 5"});
    ASSERT_EQ(outcome.status, ParseStatus::Ok);
    ASSERT_TRUE(outcome.options.max_detections.has_value());
    EXPECT_EQ(*outcome.options.max_detections, 5);
}

TEST(CliOptionsMaxDetections, LeadingPlusOk) {
    const auto outcome = run({"--max-detections", "+5"});
    ASSERT_EQ(outcome.status, ParseStatus::Ok);
    ASSERT_TRUE(outcome.options.max_detections.has_value());
    EXPECT_EQ(*outcome.options.max_detections, 5);
}

TEST(CliOptionsMaxDetections, NotFullyConsumedValuesAreErrors) {
    for (const char *value : {"0x10", "1e-1", "inf", "nan", "", "5 ", "0.5"}) {
        const auto outcome = run({"--max-detections", value});
        ASSERT_EQ(outcome.status, ParseStatus::Error) << "value='" << value << "'";
        EXPECT_EQ(outcome.error, std::string("Error: --max-detections expects an integer, got '") + value + "'")
            << "value='" << value << "'";
    }
}

TEST(CliOptionsMaxDetections, ZeroOrNegativeZeroIsNotPositive) {
    for (const char *value : {"-0", "0"}) {
        const auto outcome = run({"--max-detections", value});
        ASSERT_EQ(outcome.status, ParseStatus::Error) << "value='" << value << "'";
        EXPECT_EQ(outcome.error, "Error: --max-detections must be positive");
    }
}

// --- --resolution -------------------------------------------------------------

TEST(CliOptionsResolution, LeadingSpaceOk) {
    const auto outcome = run({"--resolution", " 5"});
    ASSERT_EQ(outcome.status, ParseStatus::Ok);
    ASSERT_TRUE(outcome.options.resolution.has_value());
    EXPECT_EQ(*outcome.options.resolution, 5);
}

TEST(CliOptionsResolution, LeadingPlusOk) {
    const auto outcome = run({"--resolution", "+5"});
    ASSERT_EQ(outcome.status, ParseStatus::Ok);
    ASSERT_TRUE(outcome.options.resolution.has_value());
    EXPECT_EQ(*outcome.options.resolution, 5);
}

TEST(CliOptionsResolution, NonPositiveValuesAreRejected) {
    for (const char *value : {"-0", "-3"}) {
        const auto outcome = run({"--resolution", value});
        ASSERT_EQ(outcome.status, ParseStatus::Error) << "value='" << value << "'";
        EXPECT_EQ(outcome.error, "Error: --resolution must be positive; omit it to auto-detect from the model");
    }
}

TEST(CliOptionsResolution, NonIntegerValueIsAnError) {
    const auto outcome = run({"--resolution", "0.5"});
    ASSERT_EQ(outcome.status, ParseStatus::Error);
    EXPECT_EQ(outcome.error, "Error: --resolution expects an integer, got '0.5'");
}

// --- --gpu-preprocess / --gpu-postprocess guard checks -----------------------

TEST(CliOptionsGpuFlags, GpuPostprocessWithoutSegmentationIsAnError) {
    const auto outcome = run({"--gpu-postprocess"});
    ASSERT_EQ(outcome.status, ParseStatus::Error);
    EXPECT_EQ(outcome.error, "Error: --gpu-postprocess applies to segmentation only; add --segmentation");
}

#if !defined(USE_CUDA_POSTPROCESS)
TEST(CliOptionsGpuFlags, GpuPostprocessRequiresCudaPostprocessBuild) {
    const auto outcome = run({"--segmentation", "--gpu-postprocess"});
    ASSERT_EQ(outcome.status, ParseStatus::Error);
    EXPECT_EQ(outcome.error, "Error: --gpu-postprocess requires a build with -DUSE_CUDA_POSTPROCESS=ON");
}
#endif

#if !defined(USE_CUDA_PREPROCESS) && !defined(USE_DALI)
TEST(CliOptionsGpuFlags, GpuPreprocessRequiresCudaPreprocessOrDaliBuild) {
    const auto outcome = run({"--gpu-preprocess"});
    ASSERT_EQ(outcome.status, ParseStatus::Error);
    EXPECT_EQ(outcome.error, "Error: --gpu-preprocess requires a build with -DUSE_CUDA_PREPROCESS=ON or -DUSE_DALI=ON");
}
#endif

// --- --background-class-id -----------------------------------------------------

TEST(CliOptionsBackgroundClassId, NonIntegerValueIsAnError) {
    const auto outcome = run({"--background-class-id", "1x"});
    ASSERT_EQ(outcome.status, ParseStatus::Error);
    EXPECT_EQ(outcome.error, "Error: --background-class-id expects an integer, got '1x'");
}

TEST(CliOptionsBackgroundClassId, NoneKeepsEverySlot) {
    const auto outcome = run({"--background-class-id", "none"});
    ASSERT_EQ(outcome.status, ParseStatus::Ok);
    EXPECT_TRUE(outcome.options.background_class_id_given);
    EXPECT_FALSE(outcome.options.background_class_id.has_value());
}

TEST(CliOptionsBackgroundClassId, NegativeOneIsOk) {
    const auto outcome = run({"--background-class-id", "-1"});
    ASSERT_EQ(outcome.status, ParseStatus::Ok);
    EXPECT_TRUE(outcome.options.background_class_id_given);
    ASSERT_TRUE(outcome.options.background_class_id.has_value());
    EXPECT_EQ(*outcome.options.background_class_id, -1);
}

// --- --keypoint-counts ----------------------------------------------------------

TEST(CliOptionsKeypointCounts, LeadingCommaIsAnError) {
    const auto outcome = run({"--keypoint-counts", ",1"});
    ASSERT_EQ(outcome.status, ParseStatus::Error);
    EXPECT_EQ(outcome.error, "Error: --keypoint-counts expects comma-separated integers, got ',1'");
}

TEST(CliOptionsKeypointCounts, TrailingCommaIsAnError) {
    const auto outcome = run({"--keypoint-counts", "1,"});
    ASSERT_EQ(outcome.status, ParseStatus::Error);
    EXPECT_EQ(outcome.error, "Error: --keypoint-counts expects comma-separated integers, got '1,'");
}

TEST(CliOptionsKeypointCounts, SpacesAndSignsAreOk) {
    const auto outcome = run({"--keypoint-counts", " 1,+2"});
    ASSERT_EQ(outcome.status, ParseStatus::Ok);
    ASSERT_TRUE(outcome.options.keypoint_counts.has_value());
    EXPECT_EQ(*outcome.options.keypoint_counts, (std::vector<int>{1, 2}));
}

// --- --mask-threshold -------------------------------------------------------------

TEST(CliOptionsMaskThreshold, NegativeValueIsOk) {
    const auto outcome = run({"--mask-threshold", "-0.5"});
    ASSERT_EQ(outcome.status, ParseStatus::Ok);
    ASSERT_TRUE(outcome.options.mask_threshold.has_value());
    EXPECT_FLOAT_EQ(*outcome.options.mask_threshold, -0.5f);
}

// --- repeated flags: last one wins ------------------------------------------

TEST(CliOptionsRepeatedFlags, RepeatedKeypointCountsReplacesRatherThanAppends) {
    const auto outcome = run({"--keypoint-counts", "1,2", "--keypoint-counts", "3"});
    ASSERT_EQ(outcome.status, ParseStatus::Ok);
    ASSERT_TRUE(outcome.options.keypoint_counts.has_value());
    EXPECT_EQ(*outcome.options.keypoint_counts, (std::vector<int>{3}));
}

TEST(CliOptionsRepeatedFlags, RepeatedScalarFlagLastOneWins) {
    const auto outcome = run({"--threshold", "0.2", "--threshold", "0.7"});
    ASSERT_EQ(outcome.status, ParseStatus::Ok);
    ASSERT_TRUE(outcome.options.threshold.has_value());
    EXPECT_FLOAT_EQ(*outcome.options.threshold, 0.7f);
}

TEST(CliOptionsRepeatedFlags, RepeatedBackgroundClassIdLastOneWins) {
    const auto outcome = run({"--background-class-id", "3", "--background-class-id", "none"});
    ASSERT_EQ(outcome.status, ParseStatus::Ok);
    EXPECT_TRUE(outcome.options.background_class_id_given);
    EXPECT_FALSE(outcome.options.background_class_id.has_value());
}

// --- error precedence: scan errors before range checks, first error wins ---
// Measured against the baseline binary (build-baseline/inference_app) from
// /home/oli/repos/rf-detr-cpp-inference with
// data/models/rfdetr-nano-1101.onnx, data/dog.jpg, data/coco-labels-91.txt.

TEST(CliOptionsErrorPrecedence, ScanErrorBeatsLaterRangeCheck) {
    const auto outcome = run({"--threshold", "5", "--resolution", "abc"});
    ASSERT_EQ(outcome.status, ParseStatus::Error);
    EXPECT_EQ(outcome.error, "Error: --resolution expects an integer, got 'abc'");
}

TEST(CliOptionsErrorPrecedence, RangeChecksRunInThresholdResolutionMaxDetectionsOrder) {
    const auto outcome = run({"--resolution", "0", "--max-detections", "0", "--threshold", "2"});
    ASSERT_EQ(outcome.status, ParseStatus::Error);
    EXPECT_EQ(outcome.error, "Error: --threshold must be in [0, 1], got 2");
}

// --- misc parse-failure wording ---------------------------------------------

TEST(CliOptionsEdgeCases, MaskThresholdNonNumericIsAnError) {
    const auto outcome = run({"--mask-threshold", "x"});
    ASSERT_EQ(outcome.status, ParseStatus::Error);
    EXPECT_EQ(outcome.error, "Error: --mask-threshold expects a number, got 'x'");
}

TEST(CliOptionsEdgeCases, KeypointCountsNonNumericIsAnError) {
    const auto outcome = run({"--keypoint-counts", "a"});
    ASSERT_EQ(outcome.status, ParseStatus::Error);
    EXPECT_EQ(outcome.error, "Error: --keypoint-counts expects comma-separated integers, got 'a'");
}

// A value flag swallows the very next token, even if that token looks like
// another flag; there is no lookahead to tell the two apart.
TEST(CliOptionsEdgeCases, ValueFlagSwallowsNextTokenEvenIfItLooksLikeAFlag) {
    const auto outcome = run({"--threshold", "--segmentation"});
    ASSERT_EQ(outcome.status, ParseStatus::Error);
    EXPECT_EQ(outcome.error, "Error: --threshold expects a number, got '--segmentation'");
}

TEST(CliOptionsEdgeCases, OutputSwallowsNextTokenLeavingSegmentationUnset) {
    const auto outcome = run({"--output", "--segmentation"});
    ASSERT_EQ(outcome.status, ParseStatus::Ok);
    ASSERT_TRUE(outcome.options.output_path.has_value());
    EXPECT_EQ(*outcome.options.output_path, std::filesystem::path("--segmentation"));
    EXPECT_FALSE(outcome.options.segmentation);
}

// std::stoi/std::stof throw std::out_of_range for values outside int/float
// range; the parser folds that into the same "expects a ..." message as a
// malformed value, rather than a distinct overflow message.
TEST(CliOptionsEdgeCases, MaxDetectionsOutOfIntRangeIsAnError) {
    const auto outcome = run({"--max-detections", "99999999999"});
    ASSERT_EQ(outcome.status, ParseStatus::Error);
    EXPECT_EQ(outcome.error, "Error: --max-detections expects an integer, got '99999999999'");
}

TEST(CliOptionsEdgeCases, ThresholdOutOfFloatRangeIsAnError) {
    const auto outcome = run({"--threshold", "1e40"});
    ASSERT_EQ(outcome.status, ParseStatus::Error);
    EXPECT_EQ(outcome.error, "Error: --threshold expects a number, got '1e40'");
}
