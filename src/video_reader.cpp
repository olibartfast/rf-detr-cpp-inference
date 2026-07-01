#include "video_reader.hpp"

#include <stdexcept>
#include <string>

extern "C" {
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
#include <libavutil/pixdesc.h>
}

namespace rfdetr::media {

namespace {

void check(int err, const std::string &what) {
    if (err < 0) {
        char buf[AV_ERROR_MAX_STRING_SIZE] = {};
        av_strerror(err, buf, sizeof(buf));
        throw std::runtime_error(what + ": " + std::string(buf));
    }
}

} // namespace

VideoReader::VideoReader(const std::filesystem::path &path) {
    frame_ = av_frame_alloc();
    packet_ = av_packet_alloc();
    if (frame_ == nullptr || packet_ == nullptr) {
        throw std::runtime_error("VideoReader: av_frame/av_packet alloc failed");
    }
    open_input(path);
}

VideoReader::~VideoReader() {
    if (sws_ != nullptr) {
        sws_freeContext(sws_);
    }
    if (dec_ctx_ != nullptr) {
        avcodec_free_context(&dec_ctx_);
    }
    if (fmt_ctx_ != nullptr) {
        avformat_close_input(&fmt_ctx_);
    }
    if (frame_ != nullptr) {
        av_frame_free(&frame_);
    }
    if (packet_ != nullptr) {
        av_packet_free(&packet_);
    }
}

void VideoReader::open_input(const std::filesystem::path &path) {
    if (!std::filesystem::exists(path)) {
        throw std::runtime_error("Video file does not exist: " + path.string());
    }

    int err = avformat_open_input(&fmt_ctx_, path.string().c_str(), nullptr, nullptr);
    check(err, "VideoReader: avformat_open_input failed for " + path.string());

    err = avformat_find_stream_info(fmt_ctx_, nullptr);
    if (err < 0) {
        check(err, "VideoReader: avformat_find_stream_info failed");
    }

    stream_index_ = av_find_best_stream(fmt_ctx_, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    if (stream_index_ < 0) {
        throw std::runtime_error("VideoReader: no video stream found in " + path.string());
    }
    video_stream_ = fmt_ctx_->streams[stream_index_];

    const AVCodec *codec = avcodec_find_decoder(video_stream_->codecpar->codec_id);
    if (codec == nullptr) {
        throw std::runtime_error("VideoReader: unsupported codec id " +
                                 std::to_string(static_cast<int>(video_stream_->codecpar->codec_id)));
    }

    dec_ctx_ = avcodec_alloc_context3(codec);
    if (dec_ctx_ == nullptr) {
        throw std::runtime_error("VideoReader: avcodec_alloc_context3 failed");
    }
    err = avcodec_parameters_to_context(dec_ctx_, video_stream_->codecpar);
    check(err, "VideoReader: avcodec_parameters_to_context failed");

    err = avcodec_open2(dec_ctx_, codec, nullptr);
    check(err, "VideoReader: avcodec_open2 failed");

    width_ = dec_ctx_->width;
    height_ = dec_ctx_->height;
    if (width_ <= 0 || height_ <= 0) {
        throw std::runtime_error("VideoReader: invalid dimensions " + std::to_string(width_) + "x" +
                                 std::to_string(height_));
    }

    if (video_stream_->avg_frame_rate.den != 0 && video_stream_->avg_frame_rate.num != 0) {
        fps_ = static_cast<double>(video_stream_->avg_frame_rate.num) /
               static_cast<double>(video_stream_->avg_frame_rate.den);
    } else if (video_stream_->r_frame_rate.den != 0 && video_stream_->r_frame_rate.num != 0) {
        fps_ =
            static_cast<double>(video_stream_->r_frame_rate.num) / static_cast<double>(video_stream_->r_frame_rate.den);
    }

    sws_ = sws_getContext(width_, height_, dec_ctx_->pix_fmt, width_, height_, AV_PIX_FMT_BGR24, SWS_BILINEAR, nullptr,
                          nullptr, nullptr);
    if (sws_ == nullptr) {
        throw std::runtime_error("VideoReader: sws_getContext failed");
    }
}

bool VideoReader::read(Image &out) {
    if (eof_) {
        return false;
    }

    while (true) {
        bool draining = false;
        int err = av_read_frame(fmt_ctx_, packet_);
        if (err < 0) {
            // End of file or read error: flush the decoder by sending a null packet.
            draining = true;
            err = avcodec_send_packet(dec_ctx_, nullptr);
            if (err < 0 && err != AVERROR(EAGAIN) && err != AVERROR_EOF) {
                check(err, "VideoReader: flush avcodec_send_packet failed");
            }
        } else if (packet_->stream_index != stream_index_) {
            av_packet_unref(packet_);
            continue;
        } else {
            err = avcodec_send_packet(dec_ctx_, packet_);
            if (err < 0 && err != AVERROR(EAGAIN) && err != AVERROR_EOF) {
                av_packet_unref(packet_);
                check(err, "VideoReader: avcodec_send_packet failed");
            }
        }

        // Drain all available decoded frames for this packet (or the flush).
        while (true) {
            err = avcodec_receive_frame(dec_ctx_, frame_);
            if (err == AVERROR(EAGAIN) || err == AVERROR_EOF) {
                break;
            }
            if (err < 0) {
                check(err, "VideoReader: avcodec_receive_frame failed");
            }

            out.resize(width_, height_);
            uint8_t *dst_data[1] = {out.data()};
            int dst_linesize[1] = {width_ * 3};
            sws_scale(sws_, frame_->data, frame_->linesize, 0, height_, dst_data, dst_linesize);
            av_packet_unref(packet_);
            return true;
        }

        av_packet_unref(packet_);

        if (draining) {
            // Flush fully consumed: no more frames will come.
            eof_ = true;
            return false;
        }
    }
}

} // namespace rfdetr::media
