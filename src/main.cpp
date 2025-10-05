#include "rfdetr_inference.hpp"
#include <iostream>
#include <cstring>

int main(int argc, char* argv[]) {
    if (argc < 4 || argc > 5) {
        std::cerr << "Usage: " << argv[0] << " <path_to_onnx_model> <path_to_image> <path_to_coco_labels> [--segmentation]" << std::endl;
        std::cerr << "Example (detection): " << argv[0] << " ./model.onnx ./image.jpg ./coco_labels.txt" << std::endl;
        std::cerr << "Example (segmentation): " << argv[0] << " ./model.onnx ./image.jpg ./coco_labels.txt --segmentation" << std::endl;
        return 1;
    }

    const std::filesystem::path model_path = argv[1];
    const std::filesystem::path image_path = argv[2];
    const std::filesystem::path label_file_path = argv[3];
    
    // Check if segmentation mode is enabled
    bool use_segmentation = false;
    if (argc == 5 && std::strcmp(argv[4], "--segmentation") == 0) {
        use_segmentation = true;
    }

    try {
        // Initialize the inference pipeline with a configurable resolution
        Config config;
        config.resolution = 0; // 0 = auto-detect from model
        config.model_type = use_segmentation ? ModelType::SEGMENTATION : ModelType::DETECTION;
        config.max_detections = 300;
        config.mask_threshold = 0.0f;
        RFDETRInference inference(model_path, label_file_path, config);

        // Preprocess the image
        int orig_h, orig_w;
        std::vector<float> input_data = inference.preprocess_image(image_path, orig_h, orig_w);

        // Run inference
        std::vector<Ort::Value> output_tensors = inference.run_inference(input_data);

        // Post-process the outputs
        std::vector<float> scores;
        std::vector<int> class_ids;
        std::vector<std::vector<float>> boxes;
        std::vector<cv::Mat> masks;
        const float scale_w = static_cast<float>(orig_w) / inference.get_resolution();
        const float scale_h = static_cast<float>(orig_h) / inference.get_resolution();
        
        if (use_segmentation) {
            inference.postprocess_segmentation_outputs(output_tensors, scale_w, scale_h, orig_h, orig_w, 
                                                       scores, class_ids, boxes, masks);
        } else {
            inference.postprocess_outputs(output_tensors, scale_w, scale_h, scores, class_ids, boxes);
        }

        // Load the original image for drawing
        cv::Mat image = cv::imread(image_path.string(), cv::IMREAD_COLOR);
        if (image.empty()) {
            throw std::runtime_error("Could not load image for drawing: " + image_path.string());
        }

        // Draw detections or segmentation masks
        if (use_segmentation) {
            inference.draw_segmentation_masks(image, boxes, class_ids, scores, masks);
        } else {
            inference.draw_detections(image, boxes, class_ids, scores);
        }

        // Save the output image
        const std::filesystem::path output_path = "output_image.jpg";
        if (const auto saved_path = inference.save_output_image(image, output_path)) {
            std::cout << "Output image saved to: " << saved_path->string() << std::endl;
        } else {
            throw std::runtime_error("Could not save output image to " + output_path.string());
        }

        // Print results
        std::cout << "\n--- " << (use_segmentation ? "Segmentation" : "Detection") << " Results ---" << std::endl;
        std::cout << "Found " << boxes.size() << " " << (use_segmentation ? "instances" : "detections") 
                  << " above threshold " << config.threshold << std::endl;
        for (size_t i = 0; i < boxes.size(); ++i) {
            std::cout << (use_segmentation ? "Instance " : "Detection ") << i << ":" << std::endl;
            std::cout << "  Box: [" << boxes[i][0] << ", " << boxes[i][1] << ", "
                      << boxes[i][2] << ", " << boxes[i][3] << "]" << std::endl;
            std::cout << "  Class: " << inference.get_coco_labels()[class_ids[i]]
                      << " (Score: " << scores[i] << ")" << std::endl;
            if (use_segmentation && i < masks.size()) {
                const int mask_pixels = cv::countNonZero(masks[i]);
                std::cout << "  Mask pixels: " << mask_pixels << std::endl;
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}