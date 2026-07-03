#include "media.hpp"

#include "processing_utils.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <font8x8_basic.h>
#include <stdexcept>
#include <string>

#ifdef USE_OPENCV
#include <opencv2/imgcodecs.hpp>
#else
// clang-format off: the STB *_IMPLEMENTATION macros must each immediately
// precede their corresponding header so the implementation is emitted exactly
// once in this TU.
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>
// clang-format on
#endif

namespace rfdetr::media {
namespace {

[[nodiscard]] uint8_t clamp_to_byte(float value) noexcept {
    return static_cast<uint8_t>(std::clamp(value, 0.0f, 255.0f));
}

void set_pixel(Image &image, int x, int y, Color color) noexcept {
    if (x < 0 || y < 0 || x >= image.width || y >= image.height) {
        return;
    }
    const size_t offset = (static_cast<size_t>(y) * static_cast<size_t>(image.width) + static_cast<size_t>(x)) * 3;
    image.bgr[offset + 0] = color.b;
    image.bgr[offset + 1] = color.g;
    image.bgr[offset + 2] = color.r;
}

void blend_pixel(Image &image, int x, int y, Color color, float alpha) noexcept {
    if (x < 0 || y < 0 || x >= image.width || y >= image.height) {
        return;
    }
    const size_t offset = (static_cast<size_t>(y) * static_cast<size_t>(image.width) + static_cast<size_t>(x)) * 3;
    image.bgr[offset + 0] =
        clamp_to_byte(static_cast<float>(image.bgr[offset + 0]) * (1.0f - alpha) + static_cast<float>(color.b) * alpha);
    image.bgr[offset + 1] =
        clamp_to_byte(static_cast<float>(image.bgr[offset + 1]) * (1.0f - alpha) + static_cast<float>(color.g) * alpha);
    image.bgr[offset + 2] =
        clamp_to_byte(static_cast<float>(image.bgr[offset + 2]) * (1.0f - alpha) + static_cast<float>(color.r) * alpha);
}

void draw_line(Image &image, int x0, int y0, int x1, int y1, Color color, int thickness) noexcept {
    const int dx = std::abs(x1 - x0);
    const int sx = x0 < x1 ? 1 : -1;
    const int dy = -std::abs(y1 - y0);
    const int sy = y0 < y1 ? 1 : -1;
    int err = dx + dy;

    while (true) {
        for (int yy = -thickness / 2; yy <= thickness / 2; ++yy) {
            for (int xx = -thickness / 2; xx <= thickness / 2; ++xx) {
                set_pixel(image, x0 + xx, y0 + yy, color);
            }
        }
        if (x0 == x1 && y0 == y1) {
            break;
        }
        const int e2 = 2 * err;
        if (e2 >= dy) {
            err += dy;
            x0 += sx;
        }
        if (e2 <= dx) {
            err += dx;
            y0 += sy;
        }
    }
}

void draw_rect(Image &image, const std::vector<float> &box, Color color, int thickness) noexcept {
    const int x0 = static_cast<int>(std::round(box[0]));
    const int y0 = static_cast<int>(std::round(box[1]));
    const int x1 = static_cast<int>(std::round(box[2]));
    const int y1 = static_cast<int>(std::round(box[3]));
    for (int t = 0; t < thickness; ++t) {
        draw_line(image, x0, y0 + t, x1, y0 + t, color, 1);
        draw_line(image, x0, y1 - t, x1, y1 - t, color, 1);
        draw_line(image, x0 + t, y0, x0 + t, y1, color, 1);
        draw_line(image, x1 - t, y0, x1 - t, y1, color, 1);
    }
}

void draw_circle(Image &image, int cx, int cy, int radius, Color color) noexcept {
    const int r2 = radius * radius;
    for (int y = -radius; y <= radius; ++y) {
        for (int x = -radius; x <= radius; ++x) {
            if (x * x + y * y <= r2) {
                set_pixel(image, cx + x, cy + y, color);
            }
        }
    }
}

} // namespace

Image load_image(const std::filesystem::path &path) {
#ifdef USE_OPENCV
    cv::Mat mat = cv::imread(path.string(), cv::IMREAD_COLOR); // always 3-channel BGR
    if (mat.empty()) {
        throw std::runtime_error("Could not load image from: " + path.string());
    }

    Image image;
    image.resize(mat.cols, mat.rows);
    if (mat.isContinuous()) {
        std::memcpy(image.data(), mat.data, image.bgr.size());
    } else {
        for (int r = 0; r < mat.rows; ++r) {
            std::memcpy(image.data() + static_cast<size_t>(r) * static_cast<size_t>(mat.cols) * 3, mat.ptr<uint8_t>(r),
                        static_cast<size_t>(mat.cols) * 3);
        }
    }
    return image;
#else
    int width = 0;
    int height = 0;
    int channels = 0;
    unsigned char *rgb = stbi_load(path.string().c_str(), &width, &height, &channels, 3);
    if (rgb == nullptr) {
        throw std::runtime_error("Could not load image from: " + path.string());
    }

    Image image;
    image.resize(width, height);
    const size_t pixels = static_cast<size_t>(width) * static_cast<size_t>(height);
    for (size_t i = 0; i < pixels; ++i) {
        image.bgr[i * 3 + 0] = rgb[i * 3 + 2];
        image.bgr[i * 3 + 1] = rgb[i * 3 + 1];
        image.bgr[i * 3 + 2] = rgb[i * 3 + 0];
    }
    stbi_image_free(rgb);
    return image;
#endif
}

bool save_image(const Image &image, const std::filesystem::path &path) {
    if (image.empty()) {
        return false;
    }

#ifdef USE_OPENCV
    // Wrap the contiguous BGR buffer without copying.
    cv::Mat mat(image.height, image.width, CV_8UC3, const_cast<uint8_t *>(image.data()));
    std::string ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    if (ext == ".png") {
        const std::vector<int> params{cv::IMWRITE_PNG_COMPRESSION, 3};
        return cv::imwrite(path.string(), mat, params);
    }
    const std::vector<int> params{cv::IMWRITE_JPEG_QUALITY, 95};
    return cv::imwrite(path.string(), mat, params);
#else
    std::vector<uint8_t> rgb(image.bgr.size());
    const size_t pixels = static_cast<size_t>(image.width) * static_cast<size_t>(image.height);
    for (size_t i = 0; i < pixels; ++i) {
        rgb[i * 3 + 0] = image.bgr[i * 3 + 2];
        rgb[i * 3 + 1] = image.bgr[i * 3 + 1];
        rgb[i * 3 + 2] = image.bgr[i * 3 + 0];
    }

    std::string ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    if (ext == ".png") {
        return stbi_write_png(path.string().c_str(), image.width, image.height, 3, rgb.data(), image.width * 3) != 0;
    }
    return stbi_write_jpg(path.string().c_str(), image.width, image.height, 3, rgb.data(), 95) != 0;
#endif
}

size_t count_nonzero(const Mask &mask) noexcept {
    return static_cast<size_t>(
        std::count_if(mask.data.begin(), mask.data.end(), [](uint8_t value) { return value != 0; }));
}

void preprocess_bgr_image(const Image &image, std::span<float> output, int resolution, std::span<const float, 3> means,
                          std::span<const float, 3> stds) {
    if (image.empty()) {
        throw std::runtime_error("Input image is empty");
    }
    const auto res = static_cast<size_t>(resolution);
    if (output.size() < 3 * res * res) {
        throw std::runtime_error("Output tensor is too small for requested resolution");
    }

    const float scale_x = static_cast<float>(image.width) / static_cast<float>(resolution);
    const float scale_y = static_cast<float>(image.height) / static_cast<float>(resolution);
    const size_t channel_size = res * res;
    const size_t img_w = static_cast<size_t>(image.width);
    const auto bgr_at = [&](int yy, int xx, int cc) -> float {
        const size_t idx = (static_cast<size_t>(yy) * img_w + static_cast<size_t>(xx)) * 3U + static_cast<size_t>(cc);
        return static_cast<float>(image.bgr[idx]);
    };

    for (int y = 0; y < resolution; ++y) {
        const float src_y = (static_cast<float>(y) + 0.5f) * scale_y - 0.5f;
        const int y0 = std::clamp(static_cast<int>(std::floor(src_y)), 0, image.height - 1);
        const int y1 = std::min(y0 + 1, image.height - 1);
        const float wy = src_y - static_cast<float>(y0);
        for (int x = 0; x < resolution; ++x) {
            const float src_x = (static_cast<float>(x) + 0.5f) * scale_x - 0.5f;
            const int x0 = std::clamp(static_cast<int>(std::floor(src_x)), 0, image.width - 1);
            const int x1 = std::min(x0 + 1, image.width - 1);
            const float wx = src_x - static_cast<float>(x0);
            const size_t dst = static_cast<size_t>(y) * res + static_cast<size_t>(x);

            float bgr[3]{};
            for (int c = 0; c < 3; ++c) {
                const float p00 = bgr_at(y0, x0, c);
                const float p01 = bgr_at(y0, x1, c);
                const float p10 = bgr_at(y1, x0, c);
                const float p11 = bgr_at(y1, x1, c);
                bgr[c] = (p00 * (1.0f - wx) + p01 * wx) * (1.0f - wy) + (p10 * (1.0f - wx) + p11 * wx) * wy;
            }

            output[dst] = bgr[2] / 255.0f;
            output[channel_size + dst] = bgr[1] / 255.0f;
            output[2 * channel_size + dst] = bgr[0] / 255.0f;
        }
    }

    rfdetr::processing::normalize_image(output.subspan(0, 3 * channel_size), channel_size, means, stds);
}

Mask resize_threshold_mask(std::span<const float> mask, int mask_width, int mask_height, int out_width, int out_height,
                           float threshold) {
    Mask out;
    out.width = out_width;
    out.height = out_height;
    out.data.resize(static_cast<size_t>(out_width) * static_cast<size_t>(out_height));

    const float scale_x = static_cast<float>(mask_width) / static_cast<float>(out_width);
    const float scale_y = static_cast<float>(mask_height) / static_cast<float>(out_height);
    const size_t mw = static_cast<size_t>(mask_width);
    const auto m = [&](int yy, int xx) -> float {
        return mask[static_cast<size_t>(yy) * mw + static_cast<size_t>(xx)];
    };
    for (int y = 0; y < out_height; ++y) {
        const float src_y = (static_cast<float>(y) + 0.5f) * scale_y - 0.5f;
        const int y0 = std::clamp(static_cast<int>(std::floor(src_y)), 0, mask_height - 1);
        const int y1 = std::min(y0 + 1, mask_height - 1);
        const float wy = src_y - static_cast<float>(y0);
        for (int x = 0; x < out_width; ++x) {
            const float src_x = (static_cast<float>(x) + 0.5f) * scale_x - 0.5f;
            const int x0 = std::clamp(static_cast<int>(std::floor(src_x)), 0, mask_width - 1);
            const int x1 = std::min(x0 + 1, mask_width - 1);
            const float wx = src_x - static_cast<float>(x0);
            const float p00 = m(y0, x0);
            const float p01 = m(y0, x1);
            const float p10 = m(y1, x0);
            const float p11 = m(y1, x1);
            const float value = (p00 * (1.0f - wx) + p01 * wx) * (1.0f - wy) + (p10 * (1.0f - wx) + p11 * wx) * wy;
            out.data[static_cast<size_t>(y) * static_cast<size_t>(out_width) + static_cast<size_t>(x)] =
                value > threshold ? 255 : 0;
        }
    }
    return out;
}

Color get_color_for_class(int class_id) noexcept {
    const float hue = static_cast<float>((class_id * 137) % 360) / 60.0f;
    const int sector = static_cast<int>(std::floor(hue));
    const float x = 200.0f * (1.0f - std::abs(std::fmod(hue, 2.0f) - 1.0f));

    float r = 0.0f;
    float g = 0.0f;
    float b = 0.0f;
    switch (sector) {
    case 0:
        r = 200.0f;
        g = x;
        break;
    case 1:
        r = x;
        g = 200.0f;
        break;
    case 2:
        g = 200.0f;
        b = x;
        break;
    case 3:
        g = x;
        b = 200.0f;
        break;
    case 4:
        r = x;
        b = 200.0f;
        break;
    default:
        r = 200.0f;
        b = x;
        break;
    }
    return {clamp_to_byte(b), clamp_to_byte(g), clamp_to_byte(r)};
}

void draw_detections(Image &image, std::span<const std::vector<float>> boxes, std::span<const int> class_ids) {
    for (size_t i = 0; i < boxes.size(); ++i) {
        draw_rect(image, boxes[i], get_color_for_class(class_ids[i]), 2);
    }
}

void draw_segmentation_masks(Image &image, std::span<const std::vector<float>> boxes, std::span<const int> class_ids,
                             std::span<const Mask> masks) {
    for (size_t i = 0; i < boxes.size(); ++i) {
        const Color color = get_color_for_class(class_ids[i]);
        if (masks[i].width == image.width && masks[i].height == image.height) {
            const size_t w = static_cast<size_t>(image.width);
            for (int y = 0; y < image.height; ++y) {
                for (int x = 0; x < image.width; ++x) {
                    if (masks[i].data[static_cast<size_t>(y) * w + static_cast<size_t>(x)] != 0) {
                        blend_pixel(image, x, y, color, 0.5f);
                    }
                }
            }
        }
        draw_rect(image, boxes[i], color, 2);
    }
}

void draw_keypoints(Image &image, std::span<const std::vector<float>> boxes, std::span<const int> class_ids,
                    std::span<const std::vector<KeypointResult>> keypoints,
                    std::span<const std::pair<int, int>> skeleton, Color keypoint_color) {
    const int min_dim = std::max(1, std::min(image.width, image.height));
    const int line_thickness = std::max(2, min_dim / 900);
    const int radius = std::max(4, min_dim / 500);

    for (size_t i = 0; i < boxes.size(); ++i) {
        const Color color = get_color_for_class(class_ids[i]);
        draw_rect(image, boxes[i], color, 2);
        const auto &kps = keypoints[i];
        const auto is_drawable = [&image](const KeypointResult &kp) {
            return std::isfinite(kp.x) && std::isfinite(kp.y) && kp.x >= 0.0f && kp.y >= 0.0f &&
                   kp.x < static_cast<float>(image.width) && kp.y < static_cast<float>(image.height) &&
                   kp.findability > 0.05f && kp.visibility > 0.01f;
        };
        for (const auto &[idx1, idx2] : skeleton) {
            if (idx1 < static_cast<int>(kps.size()) && idx2 < static_cast<int>(kps.size())) {
                const auto &kp1 = kps[static_cast<size_t>(idx1)];
                const auto &kp2 = kps[static_cast<size_t>(idx2)];
                if (is_drawable(kp1) && is_drawable(kp2)) {
                    draw_line(image, static_cast<int>(std::round(kp1.x)), static_cast<int>(std::round(kp1.y)),
                              static_cast<int>(std::round(kp2.x)), static_cast<int>(std::round(kp2.y)), color,
                              line_thickness);
                }
            }
        }
        for (const auto &kp : kps) {
            if (is_drawable(kp)) {
                draw_circle(image, static_cast<int>(std::round(kp.x)), static_cast<int>(std::round(kp.y)), radius,
                            keypoint_color);
            }
        }
    }
}

int text_width(std::string_view text, int scale) noexcept {
    const auto n = static_cast<int>(text.size());
    const auto sc = std::max(1, scale);
    return (n > 0 ? (n * kFontGlyphW - 1) : 0) * sc;
}

void draw_text(Image &image, std::string_view text, int x, int y, Color color, int scale) {
    const auto sc = std::max(1, scale);
    int pen_x = x;
    for (char ch : text) {
        const auto c = static_cast<unsigned char>(ch);
        const auto &glyph = font8x8_basic[c < 128 ? c : static_cast<int>('?')];
        for (int gy = 0; gy < kFontGlyphH; ++gy) {
            unsigned char row = glyph[static_cast<size_t>(gy)];
            for (int gx = 0; gx < kFontGlyphW; ++gx) {
                if ((row & (1 << gx)) == 0) {
                    continue;
                }
                for (int dy = 0; dy < sc; ++dy) {
                    for (int dx = 0; dx < sc; ++dx) {
                        set_pixel(image, pen_x + gx * sc + dx, y + gy * sc + dy, color);
                    }
                }
            }
        }
        pen_x += kFontGlyphW * sc;
    }
}

void draw_labeled_box(Image &image, const std::vector<float> &box, Color box_color, std::string_view label,
                      Color text_color, Color bg_color, int thickness, int font_scale) {
    draw_rect(image, box, box_color, thickness);
    if (label.empty()) {
        return;
    }
    const auto sc = std::max(1, font_scale);
    const int text_w = text_width(label, sc);
    const int text_h = kFontGlyphH * sc;
    int x0 = static_cast<int>(std::round(box[0]));
    int y_top = static_cast<int>(std::round(box[1]));
    // Place label just above the box; if it would clip the top, drop it below.
    int label_y = y_top - text_h - 2;
    if (label_y < 0) {
        label_y = y_top + 2;
    }
    int label_x = x0;
    if (label_x + text_w > image.width) {
        label_x = image.width - text_w - 2;
    }
    if (label_x < 0) {
        label_x = 0;
    }
    // Filled background rectangle for contrast.
    for (int yy = label_y - 1; yy <= label_y + text_h; ++yy) {
        for (int xx = label_x - 1; xx <= label_x + text_w + 1; ++xx) {
            set_pixel(image, xx, yy, bg_color);
        }
    }
    draw_text(image, label, label_x, label_y, text_color, sc);
}

} // namespace rfdetr::media
