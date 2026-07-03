#pragma once

#include "rfdetr_types.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

namespace rfdetr::media {

struct Color {
    uint8_t b{0};
    uint8_t g{0};
    uint8_t r{0};

    [[nodiscard]] bool operator==(const Color &) const noexcept = default;
};

struct Image {
    int width{0};
    int height{0};
    std::vector<uint8_t> bgr;

    [[nodiscard]] bool empty() const noexcept { return width <= 0 || height <= 0 || bgr.empty(); }
    [[nodiscard]] size_t bytes() const noexcept { return bgr.size(); }
    [[nodiscard]] uint8_t *data() noexcept { return bgr.data(); }
    [[nodiscard]] const uint8_t *data() const noexcept { return bgr.data(); }

    void resize(int new_width, int new_height) {
        width = new_width;
        height = new_height;
        bgr.resize(static_cast<size_t>(width) * static_cast<size_t>(height) * 3);
    }
};

struct Mask {
    int width{0};
    int height{0};
    std::vector<uint8_t> data;
};

[[nodiscard]] Image load_image(const std::filesystem::path &path);
[[nodiscard]] bool save_image(const Image &image, const std::filesystem::path &path);
[[nodiscard]] size_t count_nonzero(const Mask &mask) noexcept;

void preprocess_bgr_image(const Image &image, std::span<float> output, int resolution, std::span<const float, 3> means,
                          std::span<const float, 3> stds);

[[nodiscard]] Mask resize_threshold_mask(std::span<const float> mask, int mask_width, int mask_height, int out_width,
                                         int out_height, float threshold);

[[nodiscard]] Color get_color_for_class(int class_id) noexcept;
void draw_detections(Image &image, std::span<const std::vector<float>> boxes, std::span<const int> class_ids);
void draw_segmentation_masks(Image &image, std::span<const std::vector<float>> boxes, std::span<const int> class_ids,
                             std::span<const Mask> masks);
void draw_keypoints(Image &image, std::span<const std::vector<float>> boxes, std::span<const int> class_ids,
                    std::span<const std::vector<KeypointResult>> keypoints,
                    std::span<const std::pair<int, int>> skeleton, Color keypoint_color);

/// 8x8 bitmap font glyph metrics.
inline constexpr int kFontGlyphW = 8;
inline constexpr int kFontGlyphH = 8;

/// Measure the rendered width of `text` at the given pixel `scale` (height = 8*scale).
[[nodiscard]] int text_width(std::string_view text, int scale = 1) noexcept;

/// Draw `text` with the 8x8 bitmap font, scaled by `scale` (1 = 8x8 px).
void draw_text(Image &image, std::string_view text, int x, int y, Color color, int scale = 1);

/// Draw a labeled box: rectangle outline + filled label background + text.
/// Useful for detection/segmentation annotations where the label string matters.
void draw_labeled_box(Image &image, const std::vector<float> &box, Color box_color, std::string_view label,
                      Color text_color = {255, 255, 255}, Color bg_color = {0, 0, 0}, int thickness = 2,
                      int font_scale = 1);

} // namespace rfdetr::media
