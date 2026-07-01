#include "video_writer.hpp"

#include <stdexcept>
#include <string>

extern "C" {
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
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

const AVCodec *pick_encoder() {
    // Prefer libx264 (H.264); fall back to the default MPEG-4 Part 2 encoder.
    const AVCodec *enc = avcodec_find_encoder_by_name("libx264");
    if (enc != nullptr) {
        return enc;
    }
    enc = avcodec_find_encoder(AV_CODEC_ID_H264);
    if (enc != nullptr) {
        return enc;
    }
    return avcodec_find_encoder(AV_CODEC_ID_MPEG4);
}

} // namespace

VideoWriter::VideoWriter(const std::filesystem::path &path, int width, int height, double fps)
    : width_(width), height_(height) {
    if (width_ <= 0 || height_ <= 0) {
        throw std::runtime_error("VideoWriter: invalid dimensions " + std::to_string(width_) + "x" +
                                 std::to_string(height_));
    }
    if (fps <= 0.0) {
        fps = 25.0;
    }
    yuv_frame_ = av_frame_alloc();
    packet_ = av_packet_alloc();
    if (yuv_frame_ == nullptr || packet_ == nullptr) {
        throw std::runtime_error("VideoWriter: av_frame/av_packet alloc failed");
    }
    open(path, fps);
}

VideoWriter::~VideoWriter() {
    try {
        flush();
    } catch (...) {
        // Destructors must not throw; ignore flush failures during teardown.
    }
    if (header_written_) {
        if (fmt_ctx_ != nullptr) {
            av_write_trailer(fmt_ctx_);
        }
    }
    if (fmt_ctx_ != nullptr && fmt_ctx_->pb != nullptr) {
        avio_closep(&fmt_ctx_->pb);
    }
    if (yuv_frame_ != nullptr) {
        av_frame_free(&yuv_frame_);
    }
    if (packet_ != nullptr) {
        av_packet_free(&packet_);
    }
    if (sws_ != nullptr) {
        sws_freeContext(sws_);
    }
    if (enc_ctx_ != nullptr) {
        avcodec_free_context(&enc_ctx_);
    }
    if (fmt_ctx_ != nullptr) {
        avformat_free_context(fmt_ctx_);
    }
}

void VideoWriter::open(const std::filesystem::path &path, double fps) {
    const AVCodec *codec = pick_encoder();
    if (codec == nullptr) {
        throw std::runtime_error("VideoWriter: no H.264/MPEG-4 encoder available");
    }

    int err = avformat_alloc_output_context2(&fmt_ctx_, nullptr, nullptr, path.string().c_str());
    check(err, "VideoWriter: avformat_alloc_output_context2 failed");
    if (fmt_ctx_ == nullptr) {
        throw std::runtime_error("VideoWriter: could not infer output format");
    }

    stream_ = avformat_new_stream(fmt_ctx_, codec);
    if (stream_ == nullptr) {
        throw std::runtime_error("VideoWriter: avformat_new_stream failed");
    }

    enc_ctx_ = avcodec_alloc_context3(codec);
    if (enc_ctx_ == nullptr) {
        throw std::runtime_error("VideoWriter: avcodec_alloc_context3 failed");
    }

    enc_ctx_->width = width_;
    enc_ctx_->height = height_;
    enc_ctx_->pix_fmt = AV_PIX_FMT_YUV420P;
    const auto fps_r = av_d2q(fps, 1 << 16);
    enc_ctx_->time_base = av_inv_q(fps_r);
    enc_ctx_->framerate = fps_r;
    enc_ctx_->gop_size = std::max(12, static_cast<int>(fps) * 2);

    if (fmt_ctx_->oformat->flags & AVFMT_GLOBALHEADER) {
        enc_ctx_->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
    }

    // Quality: CRF for libx264, qmax for everything else.
    if (std::string(codec->name) == "libx264") {
        av_opt_set(enc_ctx_->priv_data, "crf", "23", 0);
        av_opt_set(enc_ctx_->priv_data, "preset", "veryfast", 0);
    } else {
        enc_ctx_->qmax = 5;
    }

    err = avcodec_open2(enc_ctx_, codec, nullptr);
    check(err, "VideoWriter: avcodec_open2 failed");

    err = avcodec_parameters_from_context(stream_->codecpar, enc_ctx_);
    check(err, "VideoWriter: avcodec_parameters_from_context failed");
    stream_->time_base = enc_ctx_->time_base;

    sws_ = sws_getContext(width_, height_, AV_PIX_FMT_BGR24, width_, height_, AV_PIX_FMT_YUV420P, SWS_BILINEAR, nullptr,
                          nullptr, nullptr);
    if (sws_ == nullptr) {
        throw std::runtime_error("VideoWriter: sws_getContext failed");
    }

    yuv_frame_->format = AV_PIX_FMT_YUV420P;
    yuv_frame_->width = width_;
    yuv_frame_->height = height_;
    err = av_frame_get_buffer(yuv_frame_, 0);
    check(err, "VideoWriter: av_frame_get_buffer failed");

    err = avio_open(&fmt_ctx_->pb, path.string().c_str(), AVIO_FLAG_WRITE);
    check(err, "VideoWriter: avio_open failed for " + path.string());

    err = avformat_write_header(fmt_ctx_, nullptr);
    check(err, "VideoWriter: avformat_write_header failed");
    header_written_ = true;
}

void VideoWriter::encode_and_write(const AVFrame *frame) {
    int err = avcodec_send_frame(enc_ctx_, frame);
    if (err < 0 && err != AVERROR(EAGAIN)) {
        check(err, "VideoWriter: avcodec_send_frame failed");
    }
    while (true) {
        err = avcodec_receive_packet(enc_ctx_, packet_);
        if (err == AVERROR(EAGAIN) || err == AVERROR_EOF) {
            break;
        }
        if (err < 0) {
            check(err, "VideoWriter: avcodec_receive_packet failed");
        }
        av_packet_rescale_ts(packet_, enc_ctx_->time_base, stream_->time_base);
        packet_->stream_index = stream_->index;
        err = av_interleaved_write_frame(fmt_ctx_, packet_);
        if (err < 0) {
            char buf[AV_ERROR_MAX_STRING_SIZE] = {};
            av_strerror(err, buf, sizeof(buf));
            av_packet_unref(packet_);
            throw std::runtime_error(std::string("VideoWriter: av_interleaved_write_frame failed: ") + buf);
        }
        av_packet_unref(packet_);
    }
}

bool VideoWriter::write(const Image &frame) {
    if (frame.width != width_ || frame.height != height_) {
        throw std::runtime_error("VideoWriter: frame size " + std::to_string(frame.width) + "x" +
                                 std::to_string(frame.height) + " does not match writer " + std::to_string(width_) +
                                 "x" + std::to_string(height_));
    }
    if (frame.bgr.size() != static_cast<size_t>(width_) * static_cast<size_t>(height_) * 3) {
        throw std::runtime_error("VideoWriter: frame buffer size mismatch");
    }

    int err = av_frame_make_writable(yuv_frame_);
    check(err, "VideoWriter: av_frame_make_writable failed");

    const uint8_t *src_data[1] = {frame.data()};
    const int src_linesize[1] = {width_ * 3};
    sws_scale(sws_, src_data, src_linesize, 0, height_, yuv_frame_->data, yuv_frame_->linesize);
    yuv_frame_->pts = pts_++;
    encode_and_write(yuv_frame_);
    return true;
}

void VideoWriter::flush() {
    if (!header_written_ || enc_ctx_ == nullptr) {
        return;
    }
    encode_and_write(nullptr);
}

} // namespace rfdetr::media
