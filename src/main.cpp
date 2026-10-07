#include "cli_options.hpp"
#include "rfdetr_inference.hpp"
#include "video_pipeline.hpp"

#include <algorithm>
#include <iostream>
#include <span>
#include <unordered_set>

namespace {

bool is_video_file(const std::filesystem::path &path) {
    static const std::unordered_set<std::string> video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"};
    std::string ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return std::tolower(c); });
    return video_exts.contains(ext);
}

// Usage text is specialized to the backend compiled into this binary: only one exists at a time, so
// showing the model container it actually accepts is more useful than listing all three.
#if defined(USE_TENSORRT)
constexpr const char *kExampleModel = "./model.engine";
constexpr const char *kBackendDescription = "TensorRT — .engine/.trt, or .onnx to build an engine";
constexpr const char *kBackendBuildFlags = "-DUSE_ONNX_RUNTIME=OFF -DUSE_TENSORRT=ON";
#elif defined(USE_EXECUTORCH)
constexpr const char *kExampleModel = "./model.pte";
constexpr const char *kBackendDescription = "ExecuTorch — .pte exported by rfdetr 1.9.0+";
constexpr const char *kBackendBuildFlags = "-DUSE_ONNX_RUNTIME=OFF -DUSE_EXECUTORCH=ON -DEXECUTORCH_ROOTDIR=<prefix>";
#else
constexpr const char *kExampleModel = "./model.onnx";
constexpr const char *kBackendDescription = "ONNX Runtime — .onnx";
constexpr const char *kBackendBuildFlags = "-DUSE_ONNX_RUNTIME=ON";
#endif

void print_usage(const char *program) {
    std::cerr << "Usage: " << program
              << " <path_to_model> <path_to_image_or_video> <path_to_coco_labels> [--segmentation|--keypoint] "
                 "[--threshold <val>] [--resolution <px>] [--max-detections <n>] [--mask-threshold <val>] "
                 "[--background-class-id <n|none>] [--keypoint-counts <n[,n...]>] [--output <path>] "
                 "[--display] [--gpu-preprocess] [--gpu-postprocess] [--dali-pipeline-dir <dir>]"
              << '\n';
    std::cerr << "Examples:" << '\n';
    std::cerr << "  Detection:    " << program << " " << kExampleModel << " ./image.jpg ./coco_labels.txt" << '\n';
    std::cerr << "  Segmentation: " << program << " " << kExampleModel
              << " ./image.jpg ./coco_labels.txt --segmentation" << '\n';
    std::cerr << "  Keypoint:     " << program << " " << kExampleModel << " ./image.jpg ./coco_labels.txt --keypoint"
              << '\n';
    std::cerr << "  Video:        " << program << " " << kExampleModel << " ./video.mp4 ./coco_labels.txt" << '\n';
    std::cerr << "  Video+display:" << program << " " << kExampleModel << " ./video.mp4 ./coco_labels.txt --display"
              << '\n';
    std::cerr << "  Tuned:        " << program << " " << kExampleModel
              << " ./image.jpg ./coco_labels.txt --threshold 0.7 --max-detections 100" << '\n';
    std::cerr << "  GPU pipeline: " << program
              << " ./model.engine ./image.jpg ./coco_labels.txt --segmentation --gpu-preprocess --gpu-postprocess"
              << '\n';
    std::cerr << '\n';
    std::cerr << "Note: exactly one backend is selected at compile time; this binary was built with" << '\n';
    std::cerr << "      " << kBackendDescription << '\n';
    std::cerr << "      Rebuild with " << kBackendBuildFlags << " to select it explicitly." << '\n';
    std::cerr << "      --background-class-id selects the exported logit slot holding background" << '\n';
    std::cerr << "      (default 0 = background-first, as the shipped RF-DETR exports are;" << '\n';
    std::cerr << "      negative counts from the end, 'none' keeps every slot)." << '\n';
    std::cerr << "      --keypoint-counts sets num_keypoints_per_class as comma-separated counts" << '\n';
    std::cerr << "      (default 0,17 = background-first COCO; pass 17 for an active-first export)." << '\n';
    std::cerr << "      --gpu-preprocess needs -DUSE_CUDA_PREPROCESS=ON (or the DALI alternative,\n";
    std::cerr << "      -DUSE_DALI=ON, which also reads --dali-pipeline-dir); --gpu-postprocess needs\n";
    std::cerr << "      -DUSE_CUDA_POSTPROCESS=ON; both require the TensorRT backend." << '\n';
}

Config build_config(const rfdetr::cli::CliOptions &opts) {
    Config config;
    config.resolution = opts.resolution.value_or(0); // 0 = auto-detect from model
    if (opts.keypoint) {
        config.model_type = ModelType::KEYPOINT;
    } else {
        config.model_type = opts.segmentation ? ModelType::SEGMENTATION : ModelType::DETECTION;
    }
    config.gpu_preprocess = opts.gpu_preprocess;
    config.gpu_postprocess = opts.gpu_postprocess;
    config.dali_pipeline_dir = opts.dali_pipeline_dir;
    if (opts.threshold) {
        config.threshold = *opts.threshold;
    }
    if (opts.max_detections) {
        config.max_detections = *opts.max_detections;
    }
    if (opts.mask_threshold) {
        config.mask_threshold = *opts.mask_threshold;
    }
    if (opts.background_class_id_given) {
        config.background_class_id = opts.background_class_id;
    }
    if (opts.keypoint_counts) {
        config.keypoint_counts = *opts.keypoint_counts;
    }
    return config;
}

void run_video(const rfdetr::cli::CliOptions &opts, Config config) {
    // Probe model to resolve auto-detected resolution
    RFDETRInference probe(opts.model_path, opts.label_path, config);
    config.resolution = probe.get_resolution();

    rfdetr::video::VideoPipelineConfig vconfig;
    vconfig.video_path = opts.input_path;
    vconfig.model_path = opts.model_path;
    vconfig.label_path = opts.label_path;
    vconfig.output_path = opts.output_path.value_or("output_video.mp4");
    vconfig.inference_config = config;
    vconfig.ring_buffer_size = 8;
    vconfig.display = opts.display;

    rfdetr::video::VideoPipeline pipeline(vconfig);
    const size_t total = pipeline.run();
    std::cout << "Processed " << total << " frames. Output: " << vconfig.output_path.string() << '\n';
}

const char *result_type_name(bool use_keypoint, bool use_segmentation) {
    if (use_keypoint) {
        return "Keypoint";
    }
    if (use_segmentation) {
        return "Segmentation";
    }
    return "Detection";
}

void print_results(const rfdetr::cli::CliOptions &opts, const Config &config, const RFDETRInference &inference,
                   const std::vector<BoundingBox> &boxes, const std::vector<int> &class_ids,
                   const std::vector<float> &scores, const std::vector<std::vector<KeypointResult>> &keypoints,
                   const std::vector<rfdetr::media::Mask> &masks) {
    const char *result_type = result_type_name(opts.keypoint, opts.segmentation);
    std::cout << "\n--- " << result_type << " Results ---" << '\n';
    std::cout << "Found " << boxes.size() << " " << (opts.segmentation ? "instances" : "detections")
              << " above threshold " << config.threshold << '\n';
    for (size_t i = 0; i < boxes.size(); ++i) {
        std::cout << (opts.segmentation ? "Instance " : "Detection ") << i << ":" << '\n';
        std::cout << "  Box: [" << boxes[i].x_min << ", " << boxes[i].y_min << ", " << boxes[i].x_max << ", "
                  << boxes[i].y_max << "]" << '\n';
        std::cout << "  Class: " << inference.get_label_name(class_ids[i]) << " (Score: " << scores[i] << ")" << '\n';
        if (opts.keypoint && i < keypoints.size()) {
            std::cout << "  Keypoints: " << keypoints[i].size() << '\n';
            for (size_t k = 0; k < keypoints[i].size(); ++k) {
                const auto &kp = keypoints[i][k];
                std::string kp_name = (k < config.keypoint_names.size()) ? config.keypoint_names[k] : std::to_string(k);
                std::cout << "    " << kp_name << " (" << kp.x << ", " << kp.y << ") findability=" << kp.findability
                          << " visibility=" << kp.visibility << '\n';
            }
        }
        if (opts.segmentation && i < masks.size()) {
            const auto mask_pixels = rfdetr::media::count_nonzero(masks[i]);
            std::cout << "  Mask pixels: " << mask_pixels << '\n';
        }
    }
}

void run_image(const rfdetr::cli::CliOptions &opts, const Config &config) {
    // --- Single image inference (existing logic) ---
    RFDETRInference inference(opts.model_path, opts.label_path, config);

    int orig_h = 0;
    int orig_w = 0;

    // Both are always false in a CPU-only build: gpu_*_active() reports
    // whether the path is compiled in, enabled, and backed by a device.
    const bool gpu_pre = inference.gpu_preprocess_active();
    const bool gpu_post = inference.gpu_postprocess_active();

#if defined(USE_CUDA_POSTPROCESS) || defined(USE_CUDA_PREPROCESS) || defined(USE_DALI)
    if (gpu_pre) {
        // Preprocess and infer entirely on the device; nothing but the
        // compressed image bytes is copied to the GPU.
        inference.run_gpu_image(opts.input_path, orig_h, orig_w);
    } else
#endif
    {
        std::vector<float> input_data = inference.preprocess_image(opts.input_path, orig_h, orig_w);
        inference.run_inference(input_data);
    }

    std::vector<float> scores;
    std::vector<int> class_ids;
    std::vector<BoundingBox> boxes;
    std::vector<rfdetr::media::Mask> masks;
    std::vector<std::vector<KeypointResult>> keypoints;
    const float scale_w = static_cast<float>(orig_w) / static_cast<float>(inference.get_resolution());
    const float scale_h = static_cast<float>(orig_h) / static_cast<float>(inference.get_resolution());

#if defined(USE_CUDA_POSTPROCESS) || defined(USE_CUDA_PREPROCESS) || defined(USE_DALI)
    // A device-side inference leaves the outputs on the GPU. The CUDA
    // postprocessor reads them there; every CPU postprocessor needs them
    // pulled into the host cache first.
    if (gpu_pre && !gpu_post) {
        inference.fetch_device_outputs();
    }
    if (gpu_post) {
        inference.postprocess_segmentation_outputs_gpu(scale_w, scale_h, orig_h, orig_w, scores, class_ids, boxes,
                                                       masks);
    } else
#endif
        if (opts.keypoint) {
        inference.postprocess_keypoint_outputs(scale_w, scale_h, orig_h, orig_w, scores, class_ids, boxes, keypoints);
    } else if (opts.segmentation) {
        inference.postprocess_segmentation_outputs(scale_w, scale_h, orig_h, orig_w, scores, class_ids, boxes, masks);
    } else {
        inference.postprocess_outputs(scale_w, scale_h, scores, class_ids, boxes);
    }
    // Both are unused in a CPU-only build, where they are always false.
    (void)gpu_pre;
    (void)gpu_post;

    rfdetr::media::Image image = rfdetr::media::load_image(opts.input_path);
    if (image.empty()) {
        throw std::runtime_error("Could not load image for drawing: " + opts.input_path.string());
    }

    if (opts.keypoint) {
        inference.draw_keypoints(image, boxes, class_ids, scores, keypoints);
    } else if (opts.segmentation) {
        inference.draw_segmentation_masks(image, boxes, class_ids, scores, masks);
    } else {
        inference.draw_detections(image, boxes, class_ids, scores);
    }

    const std::filesystem::path output_path = opts.output_path.value_or("output_image.jpg");
    if (const auto saved_path = inference.save_output_image(image, output_path)) {
        std::cout << "Output image saved to: " << saved_path->string() << '\n';
    } else {
        throw std::runtime_error("Could not save output image to " + output_path.string());
    }

    print_results(opts, config, inference, boxes, class_ids, scores, keypoints, masks);
}

} // anonymous namespace

int main(int argc, const char *argv[]) {
    const auto outcome = rfdetr::cli::parse_cli(std::span<const char *const>(argv, static_cast<size_t>(argc)));

    if (outcome.status == rfdetr::cli::ParseStatus::ShowUsage) {
        print_usage(argv[0]);
        return 1;
    }

    if (outcome.status == rfdetr::cli::ParseStatus::Error) {
        std::cerr << outcome.error << '\n';
        return 1;
    }

    const auto &opts = outcome.options;

    try {
        const Config config = build_config(opts);
        if (is_video_file(opts.input_path)) {
            run_video(opts, config);
        } else {
            run_image(opts, config);
        }
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }

    return 0;
}
