#include "rfdetr_preprocess.hpp"

#ifdef USE_CUDA_PREPROCESS

#include "cuda_check.hpp"

#include <stdexcept>

namespace rfdetr::gpu {

namespace {

constexpr int kBlockDim = 16;

/// Everything the kernel needs that is not a buffer, passed by value.
struct PreprocessParams {
    int src_h;
    int src_w;
    int resolution;
    float scale_x; ///< src_w / resolution, computed on the host exactly as the CPU path does
    float scale_y; ///< src_h / resolution
    float mean[3]; ///< RGB order
    float stdev[3];
};

/// One thread per output pixel, all three channels.
///
/// The arithmetic is `preprocess_bgr_image` (src/media.cpp) line for line: the
/// source coordinate is clamped before the sample index is taken (see
/// clamp_source_coord), the bilinear blend keeps the CPU's nesting, and
/// normalisation is `/255` followed by `(v - mean) / std` with the unfolded
/// ImageNet constants. Folding them into `mean*255`/`std*255`, as DALI did,
/// changes the rounding.
__global__ void preprocess_bgr(const uint8_t *__restrict__ bgr, float *__restrict__ out, PreprocessParams p) {
    const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= p.resolution || y >= p.resolution) {
        return;
    }

    const float src_y =
        fminf(fmaxf((static_cast<float>(y) + 0.5F) * p.scale_y - 0.5F, 0.0F), static_cast<float>(p.src_h - 1));
    const int y0 = min(static_cast<int>(src_y), p.src_h - 1);
    const int y1 = min(y0 + 1, p.src_h - 1);
    const float wy = src_y - static_cast<float>(y0);

    const float src_x =
        fminf(fmaxf((static_cast<float>(x) + 0.5F) * p.scale_x - 0.5F, 0.0F), static_cast<float>(p.src_w - 1));
    const int x0 = min(static_cast<int>(src_x), p.src_w - 1);
    const int x1 = min(x0 + 1, p.src_w - 1);
    const float wx = src_x - static_cast<float>(x0);

    const int64_t row0 = static_cast<int64_t>(y0) * p.src_w;
    const int64_t row1 = static_cast<int64_t>(y1) * p.src_w;
    const int64_t channel_size = static_cast<int64_t>(p.resolution) * p.resolution;
    const int64_t dst = static_cast<int64_t>(y) * p.resolution + x;

    for (int c = 0; c < 3; ++c) {
        const float p00 = static_cast<float>(bgr[(row0 + x0) * 3 + c]);
        const float p01 = static_cast<float>(bgr[(row0 + x1) * 3 + c]);
        const float p10 = static_cast<float>(bgr[(row1 + x0) * 3 + c]);
        const float p11 = static_cast<float>(bgr[(row1 + x1) * 3 + c]);
        const float value = (p00 * (1.0F - wx) + p01 * wx) * (1.0F - wy) + (p10 * (1.0F - wx) + p11 * wx) * wy;

        // BGR in, RGB planes out: source channel c lands in plane 2 - c.
        const int plane = 2 - c;
        out[plane * channel_size + dst] = (value / 255.0F - p.mean[plane]) / p.stdev[plane];
    }
}

} // namespace

void preprocess_bgr_device(const std::uint8_t *bgr_device, int height, int width, void *dst_device, int resolution,
                           std::span<const float, 3> means, std::span<const float, 3> stds, StreamHandle stream) {
    if (bgr_device == nullptr || dst_device == nullptr) {
        throw std::invalid_argument("preprocess_bgr_device: null device pointer");
    }
    if (height <= 0 || width <= 0 || resolution <= 0) {
        throw std::invalid_argument("preprocess_bgr_device: image and resolution must be non-empty");
    }

    PreprocessParams params{height,
                            width,
                            resolution,
                            static_cast<float>(width) / static_cast<float>(resolution),
                            static_cast<float>(height) / static_cast<float>(resolution),
                            {means[0], means[1], means[2]},
                            {stds[0], stds[1], stds[2]}};

    const dim3 block(kBlockDim, kBlockDim);
    const dim3 grid(static_cast<unsigned>((resolution + kBlockDim - 1) / kBlockDim),
                    static_cast<unsigned>((resolution + kBlockDim - 1) / kBlockDim));
    preprocess_bgr<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(bgr_device, static_cast<float *>(dst_device),
                                                                          params);
    CUDA_CHECK_LAST();
}

} // namespace rfdetr::gpu

#endif // USE_CUDA_PREPROCESS
