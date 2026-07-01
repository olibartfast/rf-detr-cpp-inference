#pragma once

#include "media.hpp"

#include <filesystem>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
}

#include <memory>

namespace rfdetr::media {

/// FFmpeg-backed video reader: demuxes + decodes a container into BGR24 Image
/// frames. One frame at a time, on demand (pull model).
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

    [[nodiscard]] int width() const noexcept { return width_; }
    [[nodiscard]] int height() const noexcept { return height_; }
    /// Stream frame rate as a double; falls back to 25.0 if unknown.
    [[nodiscard]] double fps() const noexcept { return fps_; }

  private:
    void open_input(const std::filesystem::path &path);

    AVFormatContext *fmt_ctx_{nullptr};
    const AVStream *video_stream_{nullptr};
    AVCodecContext *dec_ctx_{nullptr};
    SwsContext *sws_{nullptr};

    AVFrame *frame_{nullptr};
    AVPacket *packet_{nullptr};

    int width_{0};
    int height_{0};
    double fps_{25.0};
    int stream_index_{-1};
    bool eof_{false};
};

} // namespace rfdetr::media
