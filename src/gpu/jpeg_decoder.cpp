#include "jpeg_decoder.hpp"

#ifdef USE_CUDA_PREPROCESS

#include <array>
#include <cuda_runtime_api.h>
#include <nvjpeg.h>
#include <stdexcept>
#include <string>

namespace rfdetr::gpu {

namespace {

const char *status_name(nvjpegStatus_t status) {
    switch (status) {
    case NVJPEG_STATUS_SUCCESS:
        return "NVJPEG_STATUS_SUCCESS";
    case NVJPEG_STATUS_NOT_INITIALIZED:
        return "NVJPEG_STATUS_NOT_INITIALIZED";
    case NVJPEG_STATUS_INVALID_PARAMETER:
        return "NVJPEG_STATUS_INVALID_PARAMETER";
    case NVJPEG_STATUS_BAD_JPEG:
        return "NVJPEG_STATUS_BAD_JPEG";
    case NVJPEG_STATUS_JPEG_NOT_SUPPORTED:
        return "NVJPEG_STATUS_JPEG_NOT_SUPPORTED";
    case NVJPEG_STATUS_ALLOCATOR_FAILURE:
        return "NVJPEG_STATUS_ALLOCATOR_FAILURE";
    case NVJPEG_STATUS_EXECUTION_FAILED:
        return "NVJPEG_STATUS_EXECUTION_FAILED";
    case NVJPEG_STATUS_ARCH_MISMATCH:
        return "NVJPEG_STATUS_ARCH_MISMATCH";
    case NVJPEG_STATUS_INTERNAL_ERROR:
        return "NVJPEG_STATUS_INTERNAL_ERROR";
    case NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED:
        return "NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED";
    case NVJPEG_STATUS_INCOMPLETE_BITSTREAM:
        return "NVJPEG_STATUS_INCOMPLETE_BITSTREAM";
    }
    return "unknown nvjpegStatus_t";
}

void nvjpeg_check(nvjpegStatus_t status, const char *what) {
    if (status != NVJPEG_STATUS_SUCCESS) {
        throw std::runtime_error(std::string("nvJPEG error: ") + status_name(status) + " in " + what);
    }
}

struct HeaderInfo {
    ImageSize size;
    bool decodable{false};
};

HeaderInfo read_header(nvjpegHandle_t handle, std::span<const std::uint8_t> bytes) {
    int components = 0;
    nvjpegChromaSubsampling_t subsampling = NVJPEG_CSS_UNKNOWN;
    std::array<int, NVJPEG_MAX_COMPONENT> widths{};
    std::array<int, NVJPEG_MAX_COMPONENT> heights{};
    if (bytes.empty() || nvjpegGetImageInfo(handle, bytes.data(), bytes.size(), &components, &subsampling,
                                            widths.data(), heights.data()) != NVJPEG_STATUS_SUCCESS) {
        return {};
    }
    // Plane 0 is luma at full resolution. Anything but greyscale or three-component
    // YCbCr (CMYK, YCCK) has no defined BGR conversion here; stb handles those.
    const bool decodable =
        (components == 1 || components == 3) && subsampling != NVJPEG_CSS_UNKNOWN && widths[0] > 0 && heights[0] > 0;
    return {{widths[0], heights[0]}, decodable};
}

} // namespace

struct JpegDecoder::Impl {
    nvjpegHandle_t handle{nullptr};
    nvjpegJpegState_t state{nullptr};

    Impl() {
        nvjpeg_check(nvjpegCreateSimple(&handle), "nvjpegCreateSimple");
        const nvjpegStatus_t status = nvjpegJpegStateCreate(handle, &state);
        if (status != NVJPEG_STATUS_SUCCESS) {
            nvjpegDestroy(handle);
            nvjpeg_check(status, "nvjpegJpegStateCreate");
        }
    }

    ~Impl() {
        nvjpegJpegStateDestroy(state);
        nvjpegDestroy(handle);
    }

    Impl(const Impl &) = delete;
    Impl &operator=(const Impl &) = delete;
    Impl(Impl &&) = delete;
    Impl &operator=(Impl &&) = delete;
};

JpegDecoder::JpegDecoder() : impl_(std::make_unique<Impl>()) {}

JpegDecoder::~JpegDecoder() = default;

std::optional<ImageSize> JpegDecoder::probe(std::span<const std::uint8_t> bytes) const {
    const HeaderInfo info = read_header(impl_->handle, bytes);
    if (!info.decodable) {
        return std::nullopt;
    }
    return info.size;
}

ImageSize JpegDecoder::decode(std::span<const std::uint8_t> bytes, DeviceBuffer &dst, StreamHandle stream) {
    const HeaderInfo info = read_header(impl_->handle, bytes);
    if (!info.decodable) {
        throw std::runtime_error("nvJPEG cannot decode these bytes to BGR (not a greyscale or YCbCr JPEG)");
    }

    const auto row_bytes = static_cast<size_t>(info.size.width) * 3;
    dst.reserve(row_bytes * static_cast<size_t>(info.size.height));

    nvjpegImage_t image{};
    image.channel[0] = static_cast<unsigned char *>(dst.get());
    image.pitch[0] = row_bytes;
    nvjpeg_check(nvjpegDecode(impl_->handle, impl_->state, bytes.data(), bytes.size(), NVJPEG_OUTPUT_BGRI, &image,
                              static_cast<cudaStream_t>(stream)),
                 "nvjpegDecode");
    return info.size;
}

} // namespace rfdetr::gpu

#endif // USE_CUDA_PREPROCESS
