#include "cli_options.hpp"

#include <array>
#include <cstring>
#include <sstream>

namespace rfdetr::cli {

namespace {

// Numeric options report a bad value against the flag it belongs to: std::stoi/std::stof throw on a
// typo, and an uncaught exception here would abort before the usage text can help.
std::optional<std::string> parse_int_option(const char *flag, const char *value, std::optional<int> &out) {
    size_t consumed = 0;
    int parsed = 0;
    try {
        parsed = std::stoi(value, &consumed);
    } catch (const std::exception &) {
        std::ostringstream oss;
        oss << "Error: " << flag << " expects an integer, got '" << value << "'";
        return oss.str();
    }
    if (consumed != std::strlen(value)) {
        std::ostringstream oss;
        oss << "Error: " << flag << " expects an integer, got '" << value << "'";
        return oss.str();
    }
    out = parsed;
    return std::nullopt;
}

std::optional<std::string> parse_float_option(const char *flag, const char *value, std::optional<float> &out) {
    size_t consumed = 0;
    float parsed = 0.0F;
    try {
        parsed = std::stof(value, &consumed);
    } catch (const std::exception &) {
        std::ostringstream oss;
        oss << "Error: " << flag << " expects a number, got '" << value << "'";
        return oss.str();
    }
    if (consumed != std::strlen(value)) {
        std::ostringstream oss;
        oss << "Error: " << flag << " expects a number, got '" << value << "'";
        return oss.str();
    }
    out = parsed;
    return std::nullopt;
}

std::optional<std::string> parse_int_list(const char *flag, const char *value, std::vector<int> &out) {
    out.clear();
    const std::string input(value);
    size_t start = 0;
    while (true) {
        const size_t comma = input.find(',', start);
        const std::string token = input.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
        if (token.empty()) {
            std::ostringstream oss;
            oss << "Error: " << flag << " expects comma-separated integers, got '" << value << "'";
            return oss.str();
        }
        size_t consumed = 0;
        int parsed = 0;
        try {
            parsed = std::stoi(token, &consumed);
        } catch (const std::exception &) {
            std::ostringstream oss;
            oss << "Error: " << flag << " expects comma-separated integers, got '" << value << "'";
            return oss.str();
        }
        if (consumed != token.size()) {
            std::ostringstream oss;
            oss << "Error: " << flag << " expects comma-separated integers, got '" << value << "'";
            return oss.str();
        }
        out.push_back(parsed);
        if (comma == std::string::npos) {
            break;
        }
        start = comma + 1;
    }
    return std::nullopt;
}

// --- Flag dispatch tables ---------------------------------------------------
// Flattening the flag chain into small lookup tables keeps each branch a
// one-line table entry instead of a growing if/else-if ladder, so adding a
// flag never deepens the nesting the rest of the parser sits at.

struct BoolFlag {
    const char *name;
    bool CliOptions::*member;
};

constexpr std::array<BoolFlag, 5> kBoolFlags = {{
    {"--segmentation", &CliOptions::segmentation},
    {"--keypoint", &CliOptions::keypoint},
    {"--display", &CliOptions::display},
    {"--gpu-preprocess", &CliOptions::gpu_preprocess},
    {"--gpu-postprocess", &CliOptions::gpu_postprocess},
}};

using ValueHandler = std::optional<std::string> (*)(const char *, CliOptions &);

std::optional<std::string> handle_dali_pipeline_dir(const char *value, CliOptions &opts) {
    opts.dali_pipeline_dir = value;
    return std::nullopt;
}

std::optional<std::string> handle_threshold(const char *value, CliOptions &opts) {
    return parse_float_option("--threshold", value, opts.threshold);
}

std::optional<std::string> handle_resolution(const char *value, CliOptions &opts) {
    return parse_int_option("--resolution", value, opts.resolution);
}

std::optional<std::string> handle_max_detections(const char *value, CliOptions &opts) {
    return parse_int_option("--max-detections", value, opts.max_detections);
}

std::optional<std::string> handle_background_class_id(const char *value, CliOptions &opts) {
    opts.background_class_id_given = true;
    if (std::strcmp(value, "none") == 0) {
        opts.background_class_id.reset();
        return std::nullopt;
    }
    return parse_int_option("--background-class-id", value, opts.background_class_id);
}

std::optional<std::string> handle_mask_threshold(const char *value, CliOptions &opts) {
    return parse_float_option("--mask-threshold", value, opts.mask_threshold);
}

std::optional<std::string> handle_output(const char *value, CliOptions &opts) {
    opts.output_path = value;
    return std::nullopt;
}

std::optional<std::string> handle_keypoint_counts(const char *value, CliOptions &opts) {
    std::vector<int> counts;
    if (auto err = parse_int_list("--keypoint-counts", value, counts)) {
        return err;
    }
    opts.keypoint_counts = std::move(counts);
    return std::nullopt;
}

struct ValueFlag {
    const char *name;
    ValueHandler handler;
};

constexpr std::array<ValueFlag, 8> kValueFlags = {{
    {"--dali-pipeline-dir", &handle_dali_pipeline_dir},
    {"--threshold", &handle_threshold},
    {"--resolution", &handle_resolution},
    {"--max-detections", &handle_max_detections},
    {"--background-class-id", &handle_background_class_id},
    {"--mask-threshold", &handle_mask_threshold},
    {"--output", &handle_output},
    {"--keypoint-counts", &handle_keypoint_counts},
}};

// Unknown tokens are ignored, and a value flag in the last position is
// ignored (there is no following value to consume).
std::optional<std::string> scan_args(std::span<const char *const> args, CliOptions &opts) {
    for (size_t i = 4; i < args.size(); ++i) {
        bool matched = false;
        for (const auto &flag : kBoolFlags) {
            if (std::strcmp(args[i], flag.name) == 0) {
                opts.*(flag.member) = true;
                matched = true;
                break;
            }
        }
        if (matched) {
            continue;
        }
        if (i + 1 >= args.size()) {
            continue;
        }
        for (const auto &flag : kValueFlags) {
            if (std::strcmp(args[i], flag.name) == 0) {
                if (auto err = flag.handler(args[++i], opts)) {
                    return err;
                }
                break;
            }
        }
    }
    return std::nullopt;
}

std::optional<std::string> validate(const CliOptions &opts) {
    if (opts.threshold && (*opts.threshold < 0.0f || *opts.threshold > 1.0f)) {
        std::ostringstream oss;
        oss << "Error: --threshold must be in [0, 1], got " << *opts.threshold;
        return oss.str();
    }
    if (opts.resolution && *opts.resolution <= 0) {
        return "Error: --resolution must be positive; omit it to auto-detect from the model";
    }
    if (opts.max_detections && *opts.max_detections <= 0) {
        return "Error: --max-detections must be positive";
    }
    if (opts.gpu_postprocess && !opts.segmentation) {
        return "Error: --gpu-postprocess applies to segmentation only; add --segmentation";
    }

#if !defined(USE_CUDA_PREPROCESS) && !defined(USE_DALI)
    if (opts.gpu_preprocess) {
        return "Error: --gpu-preprocess requires a build with -DUSE_CUDA_PREPROCESS=ON or -DUSE_DALI=ON";
    }
#endif
#if !defined(USE_CUDA_POSTPROCESS)
    if (opts.gpu_postprocess) {
        return "Error: --gpu-postprocess requires a build with -DUSE_CUDA_POSTPROCESS=ON";
    }
#endif

    return std::nullopt;
}

} // anonymous namespace

ParseOutcome parse_cli(std::span<const char *const> args) {
    ParseOutcome outcome;

    if (args.size() < 4) {
        outcome.status = ParseStatus::ShowUsage;
        return outcome;
    }

    CliOptions opts;
    opts.model_path = args[1];
    opts.input_path = args[2];
    opts.label_path = args[3];

    if (auto err = scan_args(args, opts)) {
        outcome.status = ParseStatus::Error;
        outcome.error = *err;
        return outcome;
    }

    if (auto err = validate(opts)) {
        outcome.status = ParseStatus::Error;
        outcome.error = *err;
        return outcome;
    }

    outcome.status = ParseStatus::Ok;
    outcome.options = std::move(opts);
    return outcome;
}

} // namespace rfdetr::cli
