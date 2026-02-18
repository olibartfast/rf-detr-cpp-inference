#include "rfdetr_inference.hpp"
#include "video_pipeline.hpp"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <unordered_set>

namespace {

bool is_video_file(const std::filesystem::path &path) {
    static const std::unordered_set<std::string> video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"};
    std::string ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return std::tolower(c); });
    return video_exts.contains(ext);
}

} // anonymous namespace

int main(int argc, const char *argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0]
                  << " <path_to_model> <path_to_image_or_video> <path_to_coco_labels> [--segmentation] [--display]"
                  << std::endl;
        std::cerr << "Examples:" << std::endl;
        std::cerr << "  Detection:    " << argv[0] << " ./model.onnx ./image.jpg ./coco_labels.txt" << std::endl;
        std::cerr << "  Segmentation: " << argv[0] << " ./model.onnx ./image.jpg ./coco_labels.txt --segmentation"
                  << std::endl;
        std::cerr << "  Video:        " << argv[0] << " ./model.onnx ./video.mp4 ./coco_labels.txt" << std::endl;
        std::cerr << "  Video+display:" << argv[0] << " ./model.onnx ./video.mp4 ./coco_labels.txt --display"
                  << std::endl;
        std::cerr << std::endl;
        std::cerr << "Note: Backend (ONNX Runtime or TensorRT) is selected at compile time." << std::endl;
        std::cerr << "      Build with -DUSE_ONNX_RUNTIME=ON or -DUSE_TENSORRT=ON" << std::endl;
        return 1;
    }

    const std::filesystem::path model_path = argv[1];
    const std::filesystem::path input_path = argv[2];
    const std::filesystem::path label_file_path = argv[3];

    // Parse optional arguments
    bool use_segmentation = false;
    bool display = false;

    for (int i = 4; i < argc; ++i) {
        if (std::strcmp(argv[i], "--segmentation") == 0) {
            use_segmentation = true;
        } else if (std::strcmp(argv[i], "--display") == 0) {
            display = true;
        }
    }

    try {
        Config config;
        config.resolution = 0; // 0 = auto-detect from model
        config.model_type = use_segmentation ? ModelType::SEGMENTATION : ModelType::DETECTION;
        config.max_detections = 300;
        config.mask_threshold = 0.0F;

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
            std::vector<float> input_data = inference.preprocess_image(input_path, orig_h, orig_w);

            inference.run_inference(input_data);

            std::vector<float> scores;
            std::vector<int> class_ids;
            std::vector<std::vector<float>> boxes;
            std::vector<cv::Mat> masks;
            const float scale_w = static_cast<float>(orig_w) / static_cast<float>(inference.get_resolution());
            const float scale_h = static_cast<float>(orig_h) / static_cast<float>(inference.get_resolution());

            if (use_segmentation) {
                inference.postprocess_segmentation_outputs(scale_w, scale_h, orig_h, orig_w, scores, class_ids, boxes,
                                                           masks);
            } else {
                inference.postprocess_outputs(scale_w, scale_h, scores, class_ids, boxes);
            }

            cv::Mat image = cv::imread(input_path.string(), cv::IMREAD_COLOR);
            if (image.empty()) {
                throw std::runtime_error("Could not load image for drawing: " + input_path.string());
            }

            if (use_segmentation) {
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

            std::cout << "\n--- " << (use_segmentation ? "Segmentation" : "Detection") << " Results ---" << std::endl;
            std::cout << "Found " << boxes.size() << " " << (use_segmentation ? "instances" : "detections")
                      << " above threshold " << config.threshold << std::endl;
            for (size_t i = 0; i < boxes.size(); ++i) {
                std::cout << (use_segmentation ? "Instance " : "Detection ") << i << ":" << std::endl;
                std::cout << "  Box: [" << boxes[i][0] << ", " << boxes[i][1] << ", " << boxes[i][2] << ", "
                          << boxes[i][3] << "]" << std::endl;
                std::cout << "  Class: " << inference.get_coco_labels()[static_cast<size_t>(class_ids[i])]
                          << " (Score: " << scores[i] << ")" << std::endl;
                if (use_segmentation && i < masks.size()) {
                    const int mask_pixels = cv::countNonZero(masks[i]);
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
