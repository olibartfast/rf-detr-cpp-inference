#pragma once

#include "media.hpp"

#include <filesystem>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
}

namespace rfdetr::media {

/// FFmpeg-backed video writer: encodes BGR24 Image frames into an MP4 file
/// (libx264 when available, otherwise MPEG-4 Part 2 fallback).
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

    [[nodiscard]] int width() const noexcept { return width_; }
    [[nodiscard]] int height() const noexcept { return height_; }

  private:
    void open(const std::filesystem::path &path, double fps);
    void encode_and_write(const AVFrame *frame);
    void flush();

    AVFormatContext *fmt_ctx_{nullptr};
    AVCodecContext *enc_ctx_{nullptr};
    AVStream *stream_{nullptr};
    SwsContext *sws_{nullptr};
    AVFrame *yuv_frame_{nullptr};
    AVPacket *packet_{nullptr};

    int width_{0};
    int height_{0};
    int64_t pts_{0};
    bool header_written_{false};
};

} // namespace rfdetr::media
