#pragma once

#include "media.hpp"

#include <filesystem>
#include <memory>

namespace rfdetr::media {

/// Video writer: encodes BGR24 Image frames into a video file (MP4). The
/// backend (FFmpeg or OpenCV VideoWriter) is selected at compile time via the
/// CMake `USE_OPENCV` option.
class VideoWriter {
  public:
    VideoWriter(const std::filesystem::path &path, int width, int height, double fps);
    ~VideoWriter();

    VideoWriter(const VideoWriter &) = delete;
    VideoWriter &operator=(const VideoWriter &) = delete;
    VideoWriter(VideoWriter &&) = delete;
    VideoWriter &operator=(VideoWriter &&) = delete;

    /// Encode one BGR24 frame. `frame` must match the writer's width/height.
    bool write(const Image &frame);

    [[nodiscard]] int width() const noexcept;
    [[nodiscard]] int height() const noexcept;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace rfdetr::media
