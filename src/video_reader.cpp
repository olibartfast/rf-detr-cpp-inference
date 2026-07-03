#include "video_reader.hpp"

#include <stdexcept>
#include <string>

#ifdef USE_OPENCV
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#else
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
#include <libavutil/pixdesc.h>
#include <libswscale/swscale.h>
}
#endif

namespace rfdetr::media {

#ifdef USE_OPENCV

struct VideoReader::Impl {
    cv::VideoCapture cap;
    int width{0};
    int height{0};
    double fps{25.0};

    explicit Impl(const std::filesystem::path &path) {
        if (!std::filesystem::exists(path)) {
            throw std::runtime_error("Video file does not exist: " + path.string());
        }
        if (!cap.open(path.string())) {
            throw std::runtime_error("VideoReader: could not open " + path.string());
        }
        width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        if (width <= 0 || height <= 0) {
            throw std::runtime_error("VideoReader: invalid dimensions " + std::to_string(width) + "x" +
                                     std::to_string(height));
        }
        const double f = cap.get(cv::CAP_PROP_FPS);
        if (f > 0.0) {
            fps = f;
        }
    }

    bool read(Image &out) {
        cv::Mat mat;
        if (!cap.read(mat) || mat.empty()) {
            return false;
        }
        // Normalise to 3-channel BGR (handles grayscale or BGRA sources).
        if (mat.channels() == 4) {
            cv::cvtColor(mat, mat, cv::COLOR_BGRA2BGR);
        } else if (mat.channels() == 1) {
            cv::cvtColor(mat, mat, cv::COLOR_GRAY2BGR);
        }
        out.resize(mat.cols, mat.rows);
        if (mat.isContinuous()) {
            std::copy_n(mat.data, static_cast<size_t>(mat.cols) * static_cast<size_t>(mat.rows) * 3, out.data());
        } else {
            for (int r = 0; r < mat.rows; ++r) {
                std::copy_n(mat.ptr<uint8_t>(r), static_cast<size_t>(mat.cols) * 3,
                            out.data() + static_cast<size_t>(r) * static_cast<size_t>(mat.cols) * 3);
            }
        }
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

} // namespace

struct VideoReader::Impl {
    AVFormatContext *fmt_ctx{nullptr};
    const AVStream *video_stream{nullptr};
    AVCodecContext *dec_ctx{nullptr};
    SwsContext *sws{nullptr};
    AVFrame *frame{nullptr};
    AVPacket *packet{nullptr};
    int width{0};
    int height{0};
    double fps{25.0};
    int stream_index{-1};
    bool eof{false};

    explicit Impl(const std::filesystem::path &path) {
        frame = av_frame_alloc();
        packet = av_packet_alloc();
        if (frame == nullptr || packet == nullptr) {
            throw std::runtime_error("VideoReader: av_frame/av_packet alloc failed");
        }
        open_input(path);
    }

    ~Impl() {
        if (sws != nullptr) {
            sws_freeContext(sws);
        }
        if (dec_ctx != nullptr) {
            avcodec_free_context(&dec_ctx);
        }
        if (fmt_ctx != nullptr) {
            avformat_close_input(&fmt_ctx);
        }
        if (frame != nullptr) {
            av_frame_free(&frame);
        }
        if (packet != nullptr) {
            av_packet_free(&packet);
        }
    }

    void open_input(const std::filesystem::path &path) {
        if (!std::filesystem::exists(path)) {
            throw std::runtime_error("Video file does not exist: " + path.string());
        }

        int err = avformat_open_input(&fmt_ctx, path.string().c_str(), nullptr, nullptr);
        check(err, "VideoReader: avformat_open_input failed for " + path.string());

        err = avformat_find_stream_info(fmt_ctx, nullptr);
        if (err < 0) {
            check(err, "VideoReader: avformat_find_stream_info failed");
        }

        stream_index = av_find_best_stream(fmt_ctx, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
        if (stream_index < 0) {
            throw std::runtime_error("VideoReader: no video stream found in " + path.string());
        }
        video_stream = fmt_ctx->streams[stream_index];

        const AVCodec *codec = avcodec_find_decoder(video_stream->codecpar->codec_id);
        if (codec == nullptr) {
            throw std::runtime_error("VideoReader: unsupported codec id " +
                                     std::to_string(static_cast<int>(video_stream->codecpar->codec_id)));
        }

        dec_ctx = avcodec_alloc_context3(codec);
        if (dec_ctx == nullptr) {
            throw std::runtime_error("VideoReader: avcodec_alloc_context3 failed");
        }
        err = avcodec_parameters_to_context(dec_ctx, video_stream->codecpar);
        check(err, "VideoReader: avcodec_parameters_to_context failed");

        err = avcodec_open2(dec_ctx, codec, nullptr);
        check(err, "VideoReader: avcodec_open2 failed");

        width = dec_ctx->width;
        height = dec_ctx->height;
        if (width <= 0 || height <= 0) {
            throw std::runtime_error("VideoReader: invalid dimensions " + std::to_string(width) + "x" +
                                     std::to_string(height));
        }

        if (video_stream->avg_frame_rate.den != 0 && video_stream->avg_frame_rate.num != 0) {
            fps = static_cast<double>(video_stream->avg_frame_rate.num) /
                  static_cast<double>(video_stream->avg_frame_rate.den);
        } else if (video_stream->r_frame_rate.den != 0 && video_stream->r_frame_rate.num != 0) {
            fps = static_cast<double>(video_stream->r_frame_rate.num) /
                  static_cast<double>(video_stream->r_frame_rate.den);
        }

        sws = sws_getContext(width, height, dec_ctx->pix_fmt, width, height, AV_PIX_FMT_BGR24, SWS_BILINEAR, nullptr,
                             nullptr, nullptr);
        if (sws == nullptr) {
            throw std::runtime_error("VideoReader: sws_getContext failed");
        }
    }

    bool read(Image &out) {
        if (eof) {
            return false;
        }

        while (true) {
            bool draining = false;
            int err = av_read_frame(fmt_ctx, packet);
            if (err < 0) {
                // End of file or read error: flush the decoder by sending a null packet.
                draining = true;
                err = avcodec_send_packet(dec_ctx, nullptr);
                if (err < 0 && err != AVERROR(EAGAIN) && err != AVERROR_EOF) {
                    check(err, "VideoReader: flush avcodec_send_packet failed");
                }
            } else if (packet->stream_index != stream_index) {
                av_packet_unref(packet);
                continue;
            } else {
                err = avcodec_send_packet(dec_ctx, packet);
                if (err < 0 && err != AVERROR(EAGAIN) && err != AVERROR_EOF) {
                    av_packet_unref(packet);
                    check(err, "VideoReader: avcodec_send_packet failed");
                }
            }

            // Drain all available decoded frames for this packet (or the flush).
            while (true) {
                err = avcodec_receive_frame(dec_ctx, frame);
                if (err == AVERROR(EAGAIN) || err == AVERROR_EOF) {
                    break;
                }
                if (err < 0) {
                    check(err, "VideoReader: avcodec_receive_frame failed");
                }

                out.resize(width, height);
                uint8_t *dst_data[1] = {out.data()};
                int dst_linesize[1] = {width * 3};
                sws_scale(sws, frame->data, frame->linesize, 0, height, dst_data, dst_linesize);
                av_packet_unref(packet);
                return true;
            }

            av_packet_unref(packet);

            if (draining) {
                // Flush fully consumed: no more frames will come.
                eof = true;
                return false;
            }
        }
    }
};

#endif // USE_OPENCV

VideoReader::VideoReader(const std::filesystem::path &path) : impl_(std::make_unique<Impl>(path)) {}

VideoReader::~VideoReader() = default;

bool VideoReader::read(Image &out) { return impl_->read(out); }

int VideoReader::width() const noexcept { return impl_->width; }
int VideoReader::height() const noexcept { return impl_->height; }
double VideoReader::fps() const noexcept { return impl_->fps; }

} // namespace rfdetr::media
