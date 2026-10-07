#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace rfdetr::cli {

struct CliOptions {
    std::filesystem::path model_path;
    std::filesystem::path input_path;
    std::filesystem::path label_path;
    bool segmentation = false;
    bool keypoint = false;
    bool display = false;
    bool gpu_preprocess = false;
    bool gpu_postprocess = false;
    std::filesystem::path dali_pipeline_dir = "data/dali";
    std::optional<int> resolution;
    std::optional<int> max_detections;
    std::optional<float> threshold;
    std::optional<float> mask_threshold;
    bool background_class_id_given = false;
    std::optional<int> background_class_id;
    std::optional<std::vector<int>> keypoint_counts;
    std::optional<std::filesystem::path> output_path;
};

enum class ParseStatus : std::uint8_t { Ok, ShowUsage, Error };

struct ParseOutcome {
    ParseStatus status = ParseStatus::Error;
    CliOptions options; // meaningful only when status == ParseStatus::Ok
    std::string error;  // full message without trailing newline
};

[[nodiscard]] ParseOutcome parse_cli(std::span<const char *const> args); // args spans argv[0..argc)

} // namespace rfdetr::cli
