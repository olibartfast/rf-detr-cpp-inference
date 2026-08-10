#include "rfdetr_postprocess.hpp"

#ifdef USE_CUDA_POSTPROCESS

#include "cuda_check.hpp"

#include <cub/cub.cuh>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace rfdetr::gpu {

namespace {

constexpr int kBlockSize = 256;

/// Device mirror of SegPostprocessParams. Passed by value into every launch so
/// no kernel dereferences host memory.
struct DeviceParams {
    int num_queries;
    int num_classes;
    int dets_stride;
    int mask_h;
    int mask_w;
    int orig_w;
    int orig_h;
    float resolution;
    float scale_w;
    float scale_h;
    float threshold;
    float mask_threshold;
    int max_detections;
    int num_labels;
};

DeviceParams to_device_params(const SegPostprocessParams &p) {
    return DeviceParams{p.num_queries,
                        p.num_classes,
                        p.dets_stride,
                        p.mask_h,
                        p.mask_w,
                        p.orig_w,
                        p.orig_h,
                        static_cast<float>(p.resolution),
                        p.scale_w,
                        p.scale_h,
                        p.threshold,
                        p.mask_threshold,
                        p.max_detections,
                        p.num_labels};
}

/// Matches rfdetr::processing::sigmoid.
__device__ __forceinline__ float sigmoid(float x) { return 1.0F / (1.0F + __expf(-x)); }

/// One thread per (query, class) pair. The segmentation path ranks every pair
/// globally rather than taking a per-query argmax, so a single query can yield
/// several detections — see postprocess_segmentation_outputs.
__global__ void decode_scores(const float *__restrict__ labels, float *__restrict__ scores,
                              int32_t *__restrict__ indices, int total) {
    const int i = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= total) {
        return;
    }
    scores[i] = sigmoid(labels[i]);
    indices[i] = i;
}

/// Walks the ranked candidate list in order and compacts the survivors.
///
/// Deliberately single-threaded: `num_select` is at most max_detections (300),
/// and a sequential scan reproduces the CPU path's output ordering exactly,
/// which an atomic-append kernel would not.
__global__ void select_and_decode(const float *__restrict__ sorted_scores, const int32_t *__restrict__ sorted_indices,
                                  const float *__restrict__ dets, int num_select, DeviceParams p,
                                  int32_t *__restrict__ out_count, int32_t *__restrict__ out_query,
                                  int32_t *__restrict__ out_class, float *__restrict__ out_scores,
                                  float *__restrict__ out_boxes) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }

    int count = 0;
    for (int k = 0; k < num_select; ++k) {
        const float score = sorted_scores[k];
        // Threshold after ranking, exactly as the CPU path does: the cap applies
        // to the candidates examined, the threshold to the ones kept.
        if (score <= p.threshold) {
            continue;
        }

        const int flat = sorted_indices[k];
        const int query = flat / p.num_classes;
        // Logit 0 is background; shift so that logit 1 becomes class 0.
        const int class_id = (flat % p.num_classes) - 1;
        if (class_id < 0 || class_id >= p.num_labels) {
            continue;
        }

        const int det_offset = query * p.dets_stride;
        const float cx = dets[det_offset + 0] * p.resolution;
        const float cy = dets[det_offset + 1] * p.resolution;
        const float w = dets[det_offset + 2] * p.resolution;
        const float h = dets[det_offset + 3] * p.resolution;

        // cxcywh -> xyxy -> scale -> clamp, mirroring processing_utils.cpp. The
        // segmentation path clamps to the original frame, not to scale * res.
        float x_min = (cx - w * 0.5F) * p.scale_w;
        float y_min = (cy - h * 0.5F) * p.scale_h;
        float x_max = (cx + w * 0.5F) * p.scale_w;
        float y_max = (cy + h * 0.5F) * p.scale_h;
        const float max_w = static_cast<float>(p.orig_w);
        const float max_h = static_cast<float>(p.orig_h);
        x_min = fminf(fmaxf(x_min, 0.0F), max_w);
        y_min = fminf(fmaxf(y_min, 0.0F), max_h);
        x_max = fminf(fmaxf(x_max, 0.0F), max_w);
        y_max = fminf(fmaxf(y_max, 0.0F), max_h);

        out_query[count] = query;
        out_class[count] = class_id;
        out_scores[count] = score;
        out_boxes[count * 4 + 0] = x_min;
        out_boxes[count * 4 + 1] = y_min;
        out_boxes[count * 4 + 2] = x_max;
        out_boxes[count * 4 + 3] = y_max;
        ++count;
    }
    *out_count = count;
}

/// One thread per output mask pixel. Bilinear-resamples one [mask_h, mask_w]
/// slice up to the full frame and thresholds it, mirroring
/// rfdetr::media::resize_threshold_mask including its float arithmetic and its
/// raw-logit comparison (no sigmoid).
__global__ void resize_threshold_masks(const float *__restrict__ masks, const int32_t *__restrict__ selected_query,
                                       DeviceParams p, uint8_t *__restrict__ out, int64_t total) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= total) {
        return;
    }

    // Every mask is the same full-frame size, so the owning detection is a plain
    // division — no offset search is needed.
    const int64_t pixels_per_mask = static_cast<int64_t>(p.orig_w) * p.orig_h;
    const int detection = static_cast<int>(index / pixels_per_mask);
    const int64_t local = index % pixels_per_mask;
    const int y = static_cast<int>(local / p.orig_w);
    const int x = static_cast<int>(local % p.orig_w);

    const float *slice = masks + static_cast<int64_t>(selected_query[detection]) * p.mask_h * p.mask_w;

    const float scale_x = static_cast<float>(p.mask_w) / static_cast<float>(p.orig_w);
    const float scale_y = static_cast<float>(p.mask_h) / static_cast<float>(p.orig_h);

    // Clamp the source coordinate, not the sample index: on upscale the leading output pixels map
    // to a negative coordinate, and clamping only the index leaves a negative weight that
    // extrapolates past the edge. Matches media.cpp::clamp_source_coord (and torch's
    // align_corners=False bilinear); the CPU/GPU parity test compares the two.
    const float src_y =
        fminf(fmaxf((static_cast<float>(y) + 0.5F) * scale_y - 0.5F, 0.0F), static_cast<float>(p.mask_h - 1));
    const int y0 = min(static_cast<int>(src_y), p.mask_h - 1);
    const int y1 = min(y0 + 1, p.mask_h - 1);
    const float wy = src_y - static_cast<float>(y0);

    const float src_x =
        fminf(fmaxf((static_cast<float>(x) + 0.5F) * scale_x - 0.5F, 0.0F), static_cast<float>(p.mask_w - 1));
    const int x0 = min(static_cast<int>(src_x), p.mask_w - 1);
    const int x1 = min(x0 + 1, p.mask_w - 1);
    const float wx = src_x - static_cast<float>(x0);

    const float p00 = slice[y0 * p.mask_w + x0];
    const float p01 = slice[y0 * p.mask_w + x1];
    const float p10 = slice[y1 * p.mask_w + x0];
    const float p11 = slice[y1 * p.mask_w + x1];
    const float value = (p00 * (1.0F - wx) + p01 * wx) * (1.0F - wy) + (p10 * (1.0F - wx) + p11 * wx) * wy;

    out[index] = value > p.mask_threshold ? 255 : 0;
}

int grid_for(int64_t items) {
    const int64_t blocks = (items + kBlockSize - 1) / kBlockSize;
    if (blocks > static_cast<int64_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("Segmentation postprocess launch exceeds the CUDA grid limit");
    }
    return static_cast<int>(blocks);
}

} // namespace

struct SegPostprocessor::Impl {
    SegPostprocessParams params;
    cudaStream_t stream{nullptr};

    // Ranking scratch, sized by num_queries * num_classes.
    DeviceBuffer scores_in;
    DeviceBuffer scores_out;
    DeviceBuffer indices_in;
    DeviceBuffer indices_out;
    DeviceBuffer sort_temp;

    // Selection scratch, sized by max_detections.
    DeviceBuffer sel_count;
    DeviceBuffer sel_query;
    DeviceBuffer sel_class;
    DeviceBuffer sel_scores;
    DeviceBuffer sel_boxes;

    // Packed mask output. Grow-only: allocating for the worst case
    // (max_detections full frames) would reserve hundreds of megabytes at 1080p,
    // so this tracks the high-water mark of what frames actually produce.
    DeviceBuffer mask_data;

    // Pinned staging for the small results, so the final copies do not stage
    // through a pageable-memory bounce buffer.
    int32_t *host_count{nullptr};
    void *host_small{nullptr};
    size_t host_small_bytes{0};

    explicit Impl(const SegPostprocessParams &p, StreamHandle s) : params(p), stream(static_cast<cudaStream_t>(s)) {
        const int total = p.num_queries * p.num_classes;
        if (total <= 0) {
            throw std::runtime_error("Segmentation postprocess needs positive num_queries and num_classes");
        }
        const auto cap = static_cast<size_t>(std::max(p.max_detections, 1));

        scores_in.reserve(sizeof(float) * static_cast<size_t>(total));
        scores_out.reserve(sizeof(float) * static_cast<size_t>(total));
        indices_in.reserve(sizeof(int32_t) * static_cast<size_t>(total));
        indices_out.reserve(sizeof(int32_t) * static_cast<size_t>(total));

        size_t temp_bytes = 0;
        CUDA_CHECK(cub::DeviceRadixSort::SortPairsDescending(
            nullptr, temp_bytes, static_cast<const float *>(nullptr), static_cast<float *>(nullptr),
            static_cast<const int32_t *>(nullptr), static_cast<int32_t *>(nullptr), total, 0, sizeof(float) * 8,
            stream));
        sort_temp.reserve(std::max<size_t>(temp_bytes, 1));

        sel_count.reserve(sizeof(int32_t));
        sel_query.reserve(sizeof(int32_t) * cap);
        sel_class.reserve(sizeof(int32_t) * cap);
        sel_scores.reserve(sizeof(float) * cap);
        sel_boxes.reserve(sizeof(float) * 4 * cap);

        CUDA_CHECK(cudaHostAlloc(&host_count, sizeof(int32_t), cudaHostAllocDefault));
        host_small_bytes = cap * (sizeof(int32_t) + sizeof(float) + 4 * sizeof(float));
        CUDA_CHECK(cudaHostAlloc(&host_small, host_small_bytes, cudaHostAllocDefault));
    }

    ~Impl() {
        if (host_count != nullptr) {
            cudaFreeHost(host_count);
        }
        if (host_small != nullptr) {
            cudaFreeHost(host_small);
        }
    }

    Impl(const Impl &) = delete;
    Impl &operator=(const Impl &) = delete;
};

SegPostprocessor::SegPostprocessor(const SegPostprocessParams &params, StreamHandle stream)
    : impl_(std::make_unique<Impl>(params, stream)) {}

SegPostprocessor::~SegPostprocessor() = default;

void SegPostprocessor::set_frame_geometry(int orig_w, int orig_h, float scale_w, float scale_h) {
    impl_->params.orig_w = orig_w;
    impl_->params.orig_h = orig_h;
    impl_->params.scale_w = scale_w;
    impl_->params.scale_h = scale_h;
}

const SegPostprocessParams &SegPostprocessor::params() const noexcept { return impl_->params; }

void SegPostprocessor::run(const void *dets_device, const void *labels_device, const void *masks_device,
                           SegPostprocessResult &out) {
    if (dets_device == nullptr || labels_device == nullptr || masks_device == nullptr) {
        throw std::runtime_error("Segmentation postprocess received a null device tensor");
    }

    Impl &s = *impl_;
    const SegPostprocessParams &p = s.params;
    if (p.orig_w <= 0 || p.orig_h <= 0) {
        throw std::runtime_error("Segmentation postprocess frame geometry has not been set");
    }

    const int total = p.num_queries * p.num_classes;
    const DeviceParams dp = to_device_params(p);

    decode_scores<<<grid_for(total), kBlockSize, 0, s.stream>>>(static_cast<const float *>(labels_device),
                                                                static_cast<float *>(s.scores_in.get()),
                                                                static_cast<int32_t *>(s.indices_in.get()), total);
    CUDA_CHECK_LAST();

    size_t temp_bytes = s.sort_temp.capacity();
    CUDA_CHECK(cub::DeviceRadixSort::SortPairsDescending(
        s.sort_temp.get(), temp_bytes, static_cast<const float *>(s.scores_in.get()),
        static_cast<float *>(s.scores_out.get()), static_cast<const int32_t *>(s.indices_in.get()),
        static_cast<int32_t *>(s.indices_out.get()), total, 0, sizeof(float) * 8, s.stream));

    const int num_select = std::min(p.max_detections, total);
    select_and_decode<<<1, 1, 0, s.stream>>>(
        static_cast<const float *>(s.scores_out.get()), static_cast<const int32_t *>(s.indices_out.get()),
        static_cast<const float *>(dets_device), num_select, dp, static_cast<int32_t *>(s.sel_count.get()),
        static_cast<int32_t *>(s.sel_query.get()), static_cast<int32_t *>(s.sel_class.get()),
        static_cast<float *>(s.sel_scores.get()), static_cast<float *>(s.sel_boxes.get()));
    CUDA_CHECK_LAST();

    // The one mid-pipeline synchronisation: the mask launch geometry and the
    // size of the packed mask buffer both depend on how many detections survived.
    CUDA_CHECK(cudaMemcpyAsync(s.host_count, s.sel_count.get(), sizeof(int32_t), cudaMemcpyDeviceToHost, s.stream));
    CUDA_CHECK(cudaStreamSynchronize(s.stream));
    const int count = *s.host_count;
    if (count < 0 || count > p.max_detections) {
        throw std::runtime_error("Segmentation postprocess produced an out-of-range detection count");
    }

    out.count = count;
    out.scores.resize(static_cast<size_t>(count));
    out.class_ids.resize(static_cast<size_t>(count));
    out.boxes.resize(static_cast<size_t>(count));
    out.mask_offsets.assign(static_cast<size_t>(count) + 1, 0);
    const auto pixels_per_mask = static_cast<int64_t>(p.orig_w) * p.orig_h;
    for (int i = 0; i <= count; ++i) {
        out.mask_offsets[static_cast<size_t>(i)] = static_cast<int64_t>(i) * pixels_per_mask;
    }
    const int64_t total_pixels = static_cast<int64_t>(count) * pixels_per_mask;
    out.mask_data.resize(static_cast<size_t>(total_pixels));

    if (count == 0) {
        return;
    }

    // Stage the small results through pinned memory in one shot each.
    auto *staging = static_cast<std::byte *>(s.host_small);
    auto *h_class = reinterpret_cast<int32_t *>(staging);
    auto *h_scores = reinterpret_cast<float *>(staging + static_cast<size_t>(count) * sizeof(int32_t));
    auto *h_boxes = reinterpret_cast<float *>(staging + static_cast<size_t>(count) * (sizeof(int32_t) + sizeof(float)));

    CUDA_CHECK(cudaMemcpyAsync(h_class, s.sel_class.get(), static_cast<size_t>(count) * sizeof(int32_t),
                               cudaMemcpyDeviceToHost, s.stream));
    CUDA_CHECK(cudaMemcpyAsync(h_scores, s.sel_scores.get(), static_cast<size_t>(count) * sizeof(float),
                               cudaMemcpyDeviceToHost, s.stream));
    CUDA_CHECK(cudaMemcpyAsync(h_boxes, s.sel_boxes.get(), static_cast<size_t>(count) * 4 * sizeof(float),
                               cudaMemcpyDeviceToHost, s.stream));

    s.mask_data.reserve(static_cast<size_t>(total_pixels));
    resize_threshold_masks<<<grid_for(total_pixels), kBlockSize, 0, s.stream>>>(
        static_cast<const float *>(masks_device), static_cast<const int32_t *>(s.sel_query.get()), dp,
        static_cast<uint8_t *>(s.mask_data.get()), total_pixels);
    CUDA_CHECK_LAST();

    CUDA_CHECK(cudaMemcpyAsync(out.mask_data.data(), s.mask_data.get(), static_cast<size_t>(total_pixels),
                               cudaMemcpyDeviceToHost, s.stream));
    CUDA_CHECK(cudaStreamSynchronize(s.stream));

    for (int i = 0; i < count; ++i) {
        const auto idx = static_cast<size_t>(i);
        out.scores[idx] = h_scores[i];
        out.class_ids[idx] = h_class[i];
        out.boxes[idx] = BoundingBox{h_boxes[i * 4 + 0], h_boxes[i * 4 + 1], h_boxes[i * 4 + 2], h_boxes[i * 4 + 3]};
    }
}

} // namespace rfdetr::gpu

#endif // USE_CUDA_POSTPROCESS
