#include "rfdetr_inference.hpp"
#include "video_pipeline.hpp"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <optional>
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

// Numeric options report a bad value against the flag it belongs to: std::stoi/std::stof throw on a
// typo, and an uncaught exception here would abort before the usage text can help.
bool parse_int_option(const char *flag, const char *value, std::optional<int> &out) {
    try {
        size_t consumed = 0;
        const int parsed = std::stoi(value, &consumed);
        if (consumed == std::strlen(value)) {
            out = parsed;
            return true;
        }
    } catch (const std::exception &) {
        // fall through to the shared error message
    }
    std::cerr << "Error: " << flag << " expects an integer, got '" << value << "'" << std::endl;
    return false;
}

bool parse_float_option(const char *flag, const char *value, std::optional<float> &out) {
    try {
        size_t consumed = 0;
        const float parsed = std::stof(value, &consumed);
        if (consumed == std::strlen(value)) {
            out = parsed;
            return true;
        }
    } catch (const std::exception &) {
        // fall through to the shared error message
    }
    std::cerr << "Error: " << flag << " expects a number, got '" << value << "'" << std::endl;
    return false;
}

} // anonymous namespace

int main(int argc, const char *argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0]
                  << " <path_to_model> <path_to_image_or_video> <path_to_coco_labels> [--segmentation|--keypoint] "
                     "[--threshold <val>] [--resolution <px>] [--max-detections <n>] [--mask-threshold <val>] "
                     "[--background-class-id <n|none>] "
                     "[--display] [--gpu-preprocess] [--gpu-postprocess] [--dali-pipeline-dir <dir>]"
                  << std::endl;
        std::cerr << "Examples:" << std::endl;
        std::cerr << "  Detection:    " << argv[0] << " " << kExampleModel << " ./image.jpg ./coco_labels.txt"
                  << std::endl;
        std::cerr << "  Segmentation: " << argv[0] << " " << kExampleModel
                  << " ./image.jpg ./coco_labels.txt --segmentation" << std::endl;
        std::cerr << "  Keypoint:     " << argv[0] << " " << kExampleModel
                  << " ./image.jpg ./coco_labels.txt --keypoint" << std::endl;
        std::cerr << "  Video:        " << argv[0] << " " << kExampleModel << " ./video.mp4 ./coco_labels.txt"
                  << std::endl;
        std::cerr << "  Video+display:" << argv[0] << " " << kExampleModel << " ./video.mp4 ./coco_labels.txt --display"
                  << std::endl;
        std::cerr << "  Tuned:        " << argv[0] << " " << kExampleModel
                  << " ./image.jpg ./coco_labels.txt --threshold 0.7 --max-detections 100" << std::endl;
        std::cerr << "  GPU pipeline: " << argv[0]
                  << " ./model.engine ./image.jpg ./coco_labels.txt --segmentation --gpu-preprocess --gpu-postprocess"
                  << std::endl;
        std::cerr << std::endl;
        std::cerr << "Note: exactly one backend is selected at compile time; this binary was built with" << std::endl;
        std::cerr << "      " << kBackendDescription << std::endl;
        std::cerr << "      Rebuild with " << kBackendBuildFlags << " to select it explicitly." << std::endl;
        std::cerr << "      --background-class-id selects the exported logit slot holding background" << std::endl;
        std::cerr << "      (default 0 = background-first, as the shipped RF-DETR exports are;" << std::endl;
        std::cerr << "      negative counts from the end, 'none' keeps every slot)." << std::endl;
        std::cerr << "      --gpu-preprocess needs -DUSE_DALI=ON, --gpu-postprocess needs" << std::endl;
        std::cerr << "      -DUSE_CUDA_POSTPROCESS=ON; both require the TensorRT backend." << std::endl;
        return 1;
    }

    const std::filesystem::path model_path = argv[1];
    const std::filesystem::path input_path = argv[2];
    const std::filesystem::path label_file_path = argv[3];

    // Parse optional arguments
    bool use_segmentation = false;
    bool use_keypoint = false;
    bool display = false;
    bool gpu_preprocess = false;
    bool gpu_postprocess = false;
    std::filesystem::path dali_pipeline_dir = "data/dali";
    // Unset means "leave the Config default alone" — a plain sentinel value would be ambiguous for
    // --mask-threshold, whose argument is a logit and may legitimately be negative or zero.
    std::optional<int> resolution;
    std::optional<int> max_detections;
    std::optional<float> threshold;
    std::optional<float> mask_threshold;
    // Two levels of "unset": no flag at all leaves the Config default, while
    // --background-class-id none is an explicit request to keep every logit slot.
    bool background_class_id_given = false;
    std::optional<int> background_class_id;

    for (int i = 4; i < argc; ++i) {
        if (std::strcmp(argv[i], "--segmentation") == 0) {
            use_segmentation = true;
        } else if (std::strcmp(argv[i], "--keypoint") == 0) {
            use_keypoint = true;
        } else if (std::strcmp(argv[i], "--display") == 0) {
            display = true;
        } else if (std::strcmp(argv[i], "--gpu-preprocess") == 0) {
            gpu_preprocess = true;
        } else if (std::strcmp(argv[i], "--gpu-postprocess") == 0) {
            gpu_postprocess = true;
        } else if (std::strcmp(argv[i], "--dali-pipeline-dir") == 0 && i + 1 < argc) {
            dali_pipeline_dir = argv[++i];
        } else if (std::strcmp(argv[i], "--threshold") == 0 && i + 1 < argc) {
            if (!parse_float_option("--threshold", argv[++i], threshold)) {
                return 1;
            }
        } else if (std::strcmp(argv[i], "--resolution") == 0 && i + 1 < argc) {
            if (!parse_int_option("--resolution", argv[++i], resolution)) {
                return 1;
            }
        } else if (std::strcmp(argv[i], "--max-detections") == 0 && i + 1 < argc) {
            if (!parse_int_option("--max-detections", argv[++i], max_detections)) {
                return 1;
            }
        } else if (std::strcmp(argv[i], "--background-class-id") == 0 && i + 1 < argc) {
            const char *value = argv[++i];
            background_class_id_given = true;
            if (std::strcmp(value, "none") == 0) {
                background_class_id.reset();
            } else if (!parse_int_option("--background-class-id", value, background_class_id)) {
                return 1;
            }
        } else if (std::strcmp(argv[i], "--mask-threshold") == 0 && i + 1 < argc) {
            if (!parse_float_option("--mask-threshold", argv[++i], mask_threshold)) {
                return 1;
            }
        }
    }

    if (threshold && (*threshold < 0.0f || *threshold > 1.0f)) {
        std::cerr << "Error: --threshold must be in [0, 1], got " << *threshold << std::endl;
        return 1;
    }
    if (resolution && *resolution <= 0) {
        std::cerr << "Error: --resolution must be positive; omit it to auto-detect from the model" << std::endl;
        return 1;
    }
    if (max_detections && *max_detections <= 0) {
        std::cerr << "Error: --max-detections must be positive" << std::endl;
        return 1;
    }
    if (gpu_postprocess && !use_segmentation) {
        std::cerr << "Error: --gpu-postprocess applies to segmentation only; add --segmentation" << std::endl;
        return 1;
    }

#if !defined(USE_DALI)
    if (gpu_preprocess) {
        std::cerr << "Error: --gpu-preprocess requires a build with -DUSE_DALI=ON" << std::endl;
        return 1;
    }
#endif
#if !defined(USE_CUDA_POSTPROCESS)
    if (gpu_postprocess) {
        std::cerr << "Error: --gpu-postprocess requires a build with -DUSE_CUDA_POSTPROCESS=ON" << std::endl;
        return 1;
    }
#endif
    try {
        Config config;
        config.resolution = resolution.value_or(0); // 0 = auto-detect from model
        if (use_keypoint) {
            config.model_type = ModelType::KEYPOINT;
        } else {
            config.model_type = use_segmentation ? ModelType::SEGMENTATION : ModelType::DETECTION;
        }
        config.gpu_preprocess = gpu_preprocess;
        config.gpu_postprocess = gpu_postprocess;
        config.dali_pipeline_dir = dali_pipeline_dir;
        if (threshold) {
            config.threshold = *threshold;
        }
        if (max_detections) {
            config.max_detections = *max_detections;
        }
        if (mask_threshold) {
            config.mask_threshold = *mask_threshold;
        }
        if (background_class_id_given) {
            config.background_class_id = background_class_id;
        }

        if (is_video_file(input_path)) {
            // --- Video pipeline ---
            // Probe model to resolve auto-detected resolution
            RFDETRInference probe(model_path, label_file_path, config);
            config.resolution = probe.get_resolution();

            rfdetr::video::VideoPipelineConfig vconfig;
            vconfig.video_path = input_path;
            vconfig.model_path = model_path;
            vconfig.label_path = label_file_path;
            vconfig.output_path = "output_video.mp4";
            vconfig.inference_config = config;
            vconfig.ring_buffer_size = 8;
            vconfig.display = display;

            rfdetr::video::VideoPipeline pipeline(vconfig);
            const size_t total = pipeline.run();
            std::cout << "Processed " << total << " frames. Output: " << vconfig.output_path.string() << std::endl;
        } else {
            // --- Single image inference (existing logic) ---
            RFDETRInference inference(model_path, label_file_path, config);

            int orig_h = 0;
            int orig_w = 0;

            // Both are always false in a CPU-only build: gpu_*_active() reports
            // whether the path is compiled in, enabled, and backed by a device.
            const bool gpu_pre = inference.gpu_preprocess_active();
            const bool gpu_post = inference.gpu_postprocess_active();

#if defined(USE_CUDA_POSTPROCESS) || defined(USE_DALI)
            if (gpu_pre) {
                // Preprocess and infer entirely on the device; nothing but the
                // compressed image bytes is copied to the GPU.
                inference.run_gpu_image(input_path, orig_h, orig_w);
            } else
#endif
            {
                std::vector<float> input_data = inference.preprocess_image(input_path, orig_h, orig_w);
                inference.run_inference(input_data);
            }

            std::vector<float> scores;
            std::vector<int> class_ids;
            std::vector<BoundingBox> boxes;
            std::vector<rfdetr::media::Mask> masks;
            std::vector<std::vector<KeypointResult>> keypoints;
            const float scale_w = static_cast<float>(orig_w) / static_cast<float>(inference.get_resolution());
            const float scale_h = static_cast<float>(orig_h) / static_cast<float>(inference.get_resolution());

#if defined(USE_CUDA_POSTPROCESS) || defined(USE_DALI)
            // A device-side inference leaves the outputs on the GPU. The CUDA
            // postprocessor reads them there; every CPU postprocessor needs them
            // pulled into the host cache first.
            if (gpu_pre && !gpu_post) {
                inference.fetch_device_outputs();
            }
            if (gpu_post) {
                inference.postprocess_segmentation_outputs_gpu(scale_w, scale_h, orig_h, orig_w, scores, class_ids,
                                                               boxes, masks);
            } else
#endif
                if (use_keypoint) {
                inference.postprocess_keypoint_outputs(scale_w, scale_h, orig_h, orig_w, scores, class_ids, boxes,
                                                       keypoints);
            } else if (use_segmentation) {
                inference.postprocess_segmentation_outputs(scale_w, scale_h, orig_h, orig_w, scores, class_ids, boxes,
                                                           masks);
            } else {
                inference.postprocess_outputs(scale_w, scale_h, scores, class_ids, boxes);
            }
            // Both are unused in a CPU-only build, where they are always false.
            (void)gpu_pre;
            (void)gpu_post;

            rfdetr::media::Image image = rfdetr::media::load_image(input_path);
            if (image.empty()) {
                throw std::runtime_error("Could not load image for drawing: " + input_path.string());
            }

            if (use_keypoint) {
                inference.draw_keypoints(image, boxes, class_ids, scores, keypoints);
            } else if (use_segmentation) {
                inference.draw_segmentation_masks(image, boxes, class_ids, scores, masks);
            } else {
                inference.draw_detections(image, boxes, class_ids, scores);
            }

            const std::filesystem::path output_path = "output_image.jpg";
            if (const auto saved_path = inference.save_output_image(image, output_path)) {
                std::cout << "Output image saved to: " << saved_path->string() << std::endl;
            } else {
                throw std::runtime_error("Could not save output image to " + output_path.string());
            }

            const std::string result_type =
                use_keypoint ? "Keypoint" : (use_segmentation ? "Segmentation" : "Detection");
            std::cout << "\n--- " << result_type << " Results ---" << std::endl;
            std::cout << "Found " << boxes.size() << " " << (use_segmentation ? "instances" : "detections")
                      << " above threshold " << config.threshold << std::endl;
            for (size_t i = 0; i < boxes.size(); ++i) {
                std::cout << (use_segmentation ? "Instance " : "Detection ") << i << ":" << std::endl;
                std::cout << "  Box: [" << boxes[i].x_min << ", " << boxes[i].y_min << ", " << boxes[i].x_max << ", "
                          << boxes[i].y_max << "]" << std::endl;
                std::cout << "  Class: " << inference.get_label_name(class_ids[i]) << " (Score: " << scores[i] << ")"
                          << std::endl;
                if (use_keypoint && i < keypoints.size()) {
                    std::cout << "  Keypoints: " << keypoints[i].size() << std::endl;
                    for (size_t k = 0; k < keypoints[i].size(); ++k) {
                        const auto &kp = keypoints[i][k];
                        std::string kp_name =
                            (k < config.keypoint_names.size()) ? config.keypoint_names[k] : std::to_string(k);
                        std::cout << "    " << kp_name << " (" << kp.x << ", " << kp.y
                                  << ") findability=" << kp.findability << " visibility=" << kp.visibility << std::endl;
                    }
                }
                if (use_segmentation && i < masks.size()) {
                    const auto mask_pixels = rfdetr::media::count_nonzero(masks[i]);
                    std::cout << "  Mask pixels: " << mask_pixels << std::endl;
                }
            }
        }
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
