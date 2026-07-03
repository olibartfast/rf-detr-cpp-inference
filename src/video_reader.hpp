#pragma once

#include "media.hpp"

#include <filesystem>
#include <memory>

namespace rfdetr::media {

/// Video reader: decodes a container into BGR24 Image frames, one at a time
/// (pull model). The backend (FFmpeg or OpenCV VideoCapture) is selected at
/// compile time via the CMake `USE_OPENCV` option.
class VideoReader {
  public:
    explicit VideoReader(const std::filesystem::path &path);
    ~VideoReader();

    VideoReader(const VideoReader &) = delete;
    VideoReader &operator=(const VideoReader &) = delete;
    VideoReader(VideoReader &&) = delete;
    VideoReader &operator=(VideoReader &&) = delete;

    /// Decode the next frame into `out` (BGR24). Returns false at end of stream.
    bool read(Image &out);

    [[nodiscard]] int width() const noexcept;
    [[nodiscard]] int height() const noexcept;
    /// Stream frame rate as a double; falls back to 25.0 if unknown.
    [[nodiscard]] double fps() const noexcept;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace rfdetr::media
