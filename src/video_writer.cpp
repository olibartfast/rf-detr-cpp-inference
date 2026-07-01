#include "video_writer.hpp"

#include <stdexcept>
#include <string>

#ifdef USE_OPENCV
#include <opencv2/videoio.hpp>
#else
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
#include <libswscale/swscale.h>
}
#endif

namespace rfdetr::media {

namespace {

constexpr int kChannels = 3;

} // namespace

#ifdef USE_OPENCV

struct VideoWriter::Impl {
    cv::VideoWriter writer;
    int width{0};
    int height{0};

    Impl(const std::filesystem::path &path, int w, int h, double fps) : width(w), height(h) {
        if (width <= 0 || height <= 0) {
            throw std::runtime_error("VideoWriter: invalid dimensions " + std::to_string(width) + "x" +
                                     std::to_string(height));
        }
        if (fps <= 0.0) {
            fps = 25.0;
        }
        const int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
        if (!writer.open(path.string(), fourcc, fps, cv::Size(width, height), true)) {
            throw std::runtime_error("VideoWriter: could not open " + path.string());
        }
    }

    bool write(const Image &frame) {
        if (frame.width != width || frame.height != height) {
            throw std::runtime_error("VideoWriter: frame size " + std::to_string(frame.width) + "x" +
                                     std::to_string(frame.height) + " does not match writer " + std::to_string(width) +
                                     "x" + std::to_string(height));
        }
        // Wrap the contiguous BGR buffer without copying. VideoWriter treats it as read-only.
        cv::Mat mat(height, width, CV_8UC3, const_cast<uint8_t *>(frame.data()));
        writer.write(mat);
        return true;
    }
};

#else // FFmpeg backend

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

struct VideoWriter::Impl {
    AVFormatContext *fmt_ctx{nullptr};
    AVCodecContext *enc_ctx{nullptr};
    AVStream *stream{nullptr};
    SwsContext *sws{nullptr};
    AVFrame *yuv_frame{nullptr};
    AVPacket *packet{nullptr};
    int width{0};
    int height{0};
    int64_t pts{0};
    bool header_written{false};

    Impl(const std::filesystem::path &path, int w, int h, double fps) : width(w), height(h) {
        if (width <= 0 || height <= 0) {
            throw std::runtime_error("VideoWriter: invalid dimensions " + std::to_string(width) + "x" +
                                     std::to_string(height));
        }
        if (fps <= 0.0) {
            fps = 25.0;
        }
        yuv_frame = av_frame_alloc();
        packet = av_packet_alloc();
        if (yuv_frame == nullptr || packet == nullptr) {
            throw std::runtime_error("VideoWriter: av_frame/av_packet alloc failed");
        }
        open(path, fps);
    }

    ~Impl() {
        try {
            flush();
        } catch (...) {
            // Destructors must not throw; ignore flush failures during teardown.
        }
        if (header_written && fmt_ctx != nullptr) {
            av_write_trailer(fmt_ctx);
        }
        if (fmt_ctx != nullptr && fmt_ctx->pb != nullptr) {
            avio_closep(&fmt_ctx->pb);
        }
        if (yuv_frame != nullptr) {
            av_frame_free(&yuv_frame);
        }
        if (packet != nullptr) {
            av_packet_free(&packet);
        }
        if (sws != nullptr) {
            sws_freeContext(sws);
        }
        if (enc_ctx != nullptr) {
            avcodec_free_context(&enc_ctx);
        }
        if (fmt_ctx != nullptr) {
            avformat_free_context(fmt_ctx);
        }
    }

    void open(const std::filesystem::path &path, double fps) {
        const AVCodec *codec = pick_encoder();
        if (codec == nullptr) {
            throw std::runtime_error("VideoWriter: no H.264/MPEG-4 encoder available");
        }

        int err = avformat_alloc_output_context2(&fmt_ctx, nullptr, nullptr, path.string().c_str());
        check(err, "VideoWriter: avformat_alloc_output_context2 failed");
        if (fmt_ctx == nullptr) {
            throw std::runtime_error("VideoWriter: could not infer output format");
        }

        stream = avformat_new_stream(fmt_ctx, codec);
        if (stream == nullptr) {
            throw std::runtime_error("VideoWriter: avformat_new_stream failed");
        }

        enc_ctx = avcodec_alloc_context3(codec);
        if (enc_ctx == nullptr) {
            throw std::runtime_error("VideoWriter: avcodec_alloc_context3 failed");
        }

        enc_ctx->width = width;
        enc_ctx->height = height;
        enc_ctx->pix_fmt = AV_PIX_FMT_YUV420P;
        const auto fps_r = av_d2q(fps, 1 << 16);
        enc_ctx->time_base = av_inv_q(fps_r);
        enc_ctx->framerate = fps_r;
        enc_ctx->gop_size = std::max(12, static_cast<int>(fps) * 2);

        if (fmt_ctx->oformat->flags & AVFMT_GLOBALHEADER) {
            enc_ctx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
        }

        // Quality: CRF for libx264, qmax for everything else.
        if (std::string(codec->name) == "libx264") {
            av_opt_set(enc_ctx->priv_data, "crf", "23", 0);
            av_opt_set(enc_ctx->priv_data, "preset", "veryfast", 0);
        } else {
            enc_ctx->qmax = 5;
        }

        err = avcodec_open2(enc_ctx, codec, nullptr);
        check(err, "VideoWriter: avcodec_open2 failed");

        err = avcodec_parameters_from_context(stream->codecpar, enc_ctx);
        check(err, "VideoWriter: avcodec_parameters_from_context failed");
        stream->time_base = enc_ctx->time_base;

        sws = sws_getContext(width, height, AV_PIX_FMT_BGR24, width, height, AV_PIX_FMT_YUV420P, SWS_BILINEAR, nullptr,
                             nullptr, nullptr);
        if (sws == nullptr) {
            throw std::runtime_error("VideoWriter: sws_getContext failed");
        }

        yuv_frame->format = AV_PIX_FMT_YUV420P;
        yuv_frame->width = width;
        yuv_frame->height = height;
        err = av_frame_get_buffer(yuv_frame, 0);
        check(err, "VideoWriter: av_frame_get_buffer failed");

        err = avio_open(&fmt_ctx->pb, path.string().c_str(), AVIO_FLAG_WRITE);
        check(err, "VideoWriter: avio_open failed for " + path.string());

        err = avformat_write_header(fmt_ctx, nullptr);
        check(err, "VideoWriter: avformat_write_header failed");
        header_written = true;
    }

    void encode_and_write(const AVFrame *frame) {
        int err = avcodec_send_frame(enc_ctx, frame);
        if (err < 0 && err != AVERROR(EAGAIN)) {
            check(err, "VideoWriter: avcodec_send_frame failed");
        }
        while (true) {
            err = avcodec_receive_packet(enc_ctx, packet);
            if (err == AVERROR(EAGAIN) || err == AVERROR_EOF) {
                break;
            }
            if (err < 0) {
                check(err, "VideoWriter: avcodec_receive_packet failed");
            }
            av_packet_rescale_ts(packet, enc_ctx->time_base, stream->time_base);
            packet->stream_index = stream->index;
            err = av_interleaved_write_frame(fmt_ctx, packet);
            if (err < 0) {
                char buf[AV_ERROR_MAX_STRING_SIZE] = {};
                av_strerror(err, buf, sizeof(buf));
                av_packet_unref(packet);
                throw std::runtime_error(std::string("VideoWriter: av_interleaved_write_frame failed: ") + buf);
            }
            av_packet_unref(packet);
        }
    }

    bool write(const Image &frame) {
        if (frame.width != width || frame.height != height) {
            throw std::runtime_error("VideoWriter: frame size " + std::to_string(frame.width) + "x" +
                                     std::to_string(frame.height) + " does not match writer " + std::to_string(width) +
                                     "x" + std::to_string(height));
        }
        if (frame.bgr.size() != static_cast<size_t>(width) * static_cast<size_t>(height) * kChannels) {
            throw std::runtime_error("VideoWriter: frame buffer size mismatch");
        }

        int err = av_frame_make_writable(yuv_frame);
        check(err, "VideoWriter: av_frame_make_writable failed");

        const uint8_t *src_data[1] = {frame.data()};
        const int src_linesize[1] = {width * kChannels};
        sws_scale(sws, src_data, src_linesize, 0, height, yuv_frame->data, yuv_frame->linesize);
        yuv_frame->pts = pts++;
        encode_and_write(yuv_frame);
        return true;
    }

    void flush() {
        if (!header_written || enc_ctx == nullptr) {
            return;
        }
        encode_and_write(nullptr);
    }
};

#endif // USE_OPENCV

VideoWriter::VideoWriter(const std::filesystem::path &path, int width, int height, double fps)
    : impl_(std::make_unique<Impl>(path, width, height, fps)) {}

VideoWriter::~VideoWriter() = default;

bool VideoWriter::write(const Image &frame) { return impl_->write(frame); }

int VideoWriter::width() const noexcept { return impl_->width; }
int VideoWriter::height() const noexcept { return impl_->height; }

} // namespace rfdetr::media
