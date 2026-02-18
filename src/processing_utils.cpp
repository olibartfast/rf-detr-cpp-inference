#include "processing_utils.hpp"

#include <cmath>
#include <cstring>
#include <opencv2/imgproc.hpp>

namespace rfdetr::processing {

float sigmoid(float x) noexcept { return 1.0f / (1.0f + std::exp(-x)); }

void normalize_image(std::span<float> data, size_t channel_size, std::span<const float, 3> means,
                     std::span<const float, 3> stds) {
    for (size_t c = 0; c < 3; ++c) {
        const float mean = means[c];
        const float std = stds[c];
        for (size_t i = 0; i < channel_size; ++i) {
            data[c * channel_size + i] = (data[c * channel_size + i] - mean) / std;
        }
    }
}

BoundingBox cxcywh_to_xyxy(float cx, float cy, float w, float h) noexcept {
    return {cx - w / 2.0f, cy - h / 2.0f, cx + w / 2.0f, cy + h / 2.0f};
}

BoundingBox scale_box(const BoundingBox &box, float scale_w, float scale_h) noexcept {
    return {box.x_min * scale_w, box.y_min * scale_h, box.x_max * scale_w, box.y_max * scale_h};
}

cv::Scalar get_color_for_class(int class_id) noexcept {
    const int hue = (class_id * 137) % 180;
    cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar(hue, 200, 200));
    cv::Mat bgr;
    cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
    return {static_cast<double>(bgr.at<cv::Vec3b>(0, 0)[0]), static_cast<double>(bgr.at<cv::Vec3b>(0, 0)[1]),
            static_cast<double>(bgr.at<cv::Vec3b>(0, 0)[2])};
}

void preprocess_frame(const cv::Mat &bgr_frame, std::span<float> output, int resolution,
                      std::span<const float, 3> means, std::span<const float, 3> stds) {
    const auto res = static_cast<size_t>(resolution);

    cv::Mat resized;
    cv::resize(bgr_frame, resized, cv::Size(resolution, resolution), 0, 0, cv::INTER_LINEAR);
    cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
    resized.convertTo(resized, CV_32F, 1.0 / 255.0);

    std::vector<cv::Mat> channels;
    cv::split(resized, channels);

    float *ptr = output.data();
    for (size_t c = 0; c < 3; ++c) {
        std::memcpy(ptr, channels[c].data, res * res * sizeof(float));
        ptr += res * res;
    }

    normalize_image(output.subspan(0, 3 * res * res), res * res, means, stds);
}

} // namespace rfdetr::processing
