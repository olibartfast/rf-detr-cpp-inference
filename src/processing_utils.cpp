#include "processing_utils.hpp"

#include <cmath>
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

} // namespace rfdetr::processing
