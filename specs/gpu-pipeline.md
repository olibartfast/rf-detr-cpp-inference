# GPU Pipeline — Design Reference

Standing constraints for the already-built code in `src/gpu/`. This is a contract, not a plan —
read it before modifying anything under `src/gpu/` or the DALI pipelines in `data/dali/`. The
remaining GPU *work* lives in [roadmap.md](roadmap.md) Phases 2–4.

## Architecture

```text
image file (JPEG/PNG bytes)          video frame (BGR, host, from FFmpeg)
        |                                        |
        v                                        v
  DALI pipeline "encoded"                 DALI pipeline "frame"
  external_source IMAGE (cpu, uint8)      external_source FRAME (gpu, uint8, HWC)
  -> decoders.image(device="mixed")       -> color_space_conversion(BGR->RGB)
  -> resize(res, res)                     -> resize(res, res)
  -> crop_mirror_normalize(CHW, mean, std) -> crop_mirror_normalize(CHW, mean, std)
        |                                        |
        +----------------+-----------------------+
                         v
        device float[1,3,res,res]  (no H2D of the tensor)
                         v
        TensorRT enqueueV3(stream)  — same stream, no sync
                         v
        +----------------+----------------------------+
        |                                             |
   detection / keypoint                          segmentation
   D2H dets+labels (small)                  CUDA postprocess on stream:
   existing CPU postprocess                   decode_scores -> topk
                                              -> decode_boxes
                                              -> resize_threshold_masks (packed)
                                                         v
                                              one D2H: count, boxes, scores,
                                              classes, mask_offsets, mask_data
                         v
                 CPU drawing (unchanged)
```

One CUDA stream per inference context. DALI writes the input tensor, TensorRT consumes it, the postprocess kernels consume TensorRT's output bindings, and only the final packed results cross to the host. No intermediate synchronisation.

## Model contract

RF-DETR does not behave like the CNN detectors most GPU pipeline code is written for. Each rule below is load-bearing: breaking one produces a **silently wrong result**, not a crash. This is the review checklist for any change to `src/gpu/`.

1. **No letterbox.** Preprocessing is a plain stretch to `res×res` — independent `scale_x`, `scale_y`, no padding (`src/media.cpp:212-213`). No `fn.paste` in the DALI pipeline, and box decode stays `scale_w = orig_w / res`, `scale_h = orig_h / res`.
2. **ImageNet normalisation, not `/255` alone.** `/255` then mean `{0.485, 0.456, 0.406}` / std `{0.229, 0.224, 0.225}`. Folded into DALI as `mean = m*255`, `std = s*255`.
3. **DETR head — no NMS.** The model emits normalised `cxcywh` boxes; `num_queries` comes from tensor shapes, never a hardcoded 300. Candidates are ranked by sigmoid class scores and filtered without NMS.
4. **Full per-query masks.** Segmentation output is `masks[1, num_queries, mask_h, mask_w]` — one complete mask per query, no prototype tensor and no coefficient dot product. All dimensions come from tensor shapes. The kernel is a straight bilinear resize of one `mask_h × mask_w` slice plus threshold; multiple selected classes for a query reuse that query's mask.
5. **No sigmoid on masks.** The raw value is compared against `mask_threshold`, which defaults to `0.0` — a logit threshold. Applying sigmoid and comparing against 0.5 is equivalent only at those defaults and diverges at any other value. Keep the raw-logit comparison.
6. **Shared global top-k selection.** Detection, segmentation, and keypoint CPU paths rank sigmoid scores over the flattened query/foreground-class grid, so one query can yield several detections. The configured background column is excluded before top-k. Examine at most `min(max_detections, num_queries * num_foreground)` ranked candidates, then retain only scores strictly greater than `threshold`; rejected candidates are not replaced from below the cap. GPU segmentation follows the same selection contract.
7. **Compact foreground class indices.** Class IDs index the foreground columns after removing the configured background slot, then candidates outside the label list are dropped. `background_class_id` defaults to `0` (skip the first exported column), supports negative indices counted from the end, and accepts `std::nullopt` to keep every column. Subtracting one unconditionally is incorrect; CPU selection and the GPU kernel use the compact foreground index directly.
8. **DALI hosts preprocessing only.** Postprocess kernels live directly in `src/gpu/` and are called by our own code. There is no external scheduler to satisfy, so wrapping them in DALI operators would cost the operator schema, pipeline serialisation, and plugin loading for no benefit.

## Packed output contract

`src/gpu/rfdetr_postprocess.hpp` — shaped for a single D2H:

| Buffer | Type | Shape |
|--------|------|-------|
| `count` | `int32` | `[1]` |
| `boxes` | `float32` | `[max_detections, 4]` xyxy in original-image pixels |
| `scores` | `float32` | `[max_detections]` |
| `classes` | `int32` | `[max_detections]` |
| `mask_offsets` | `int64` | `[max_detections + 1]` prefix sums into `mask_data` |
| `mask_data` | `uint8` | `[sum of orig_w*orig_h per detection]` — 0 or 255 |

`mask_offsets` is always full-length even at `count == 0`, so the host reads a fixed stride and can reject a short buffer.

## Correctness rules

- **`daliOutputRelease` ordering.** Release **after** the TensorRT enqueue, never before. Getting this wrong hands DALI's buffer back to its pool while TensorRT is still reading it, producing intermittent garbage rather than a crash.
- **`antialias=False` in the DALI resize.** DALI antialiases when downscaling by default; `preprocess_bgr_image` is a plain 4-tap bilinear sample. Without it, tensors diverge on every image larger than `res`, worst on the largest.
- **`float` accumulation in the mask kernel, not `double`.** The CPU reference is `float`; matching it matters more than extra precision.
- **Rank all foreground pairs, cap examined candidates, then apply the strict threshold.** Filtering can leave fewer than `max_detections` outputs; do not backfill. See model contract rule 6.
- **Read shapes from tensors.** `mask_h`, `mask_w`, `num_queries`, `num_classes` come from the tensor shapes — never hardcode them, or the kernels lock to one engine.
- **Finite equal-score ties use ascending flattened foreground index.** CPU `select_topk_multiclass` explicitly breaks ties by index. CUDA `decode_scores` initializes indices in ascending order and CUB `DeviceRadixSort::SortPairsDescending` is stable, preserving that order for equal scores. This does not assert CPU/GPU NaN ranking equality.
- **Resize is a tolerance gate, never an equality gate.** DALI resize will not bit-match the CPU bilinear. Do not promise equality for resampling.

## Risks

| Risk | Mitigation |
|------|------------|
| DALI resize never bit-matches the CPU bilinear | Tolerance gate from the start; never promise equality for resampling |
| DALI version coupled to CUDA/TensorRT versions | Container tag pinned; the triple is recorded in [tech-stack.md](tech-stack.md) |
| `daliOutputRelease` ordering bug | Explicit ordering rule above, plus a `compute-sanitizer` run in the [roadmap.md](roadmap.md) Phase 4 gate |
| Kernels hardcode one engine's shapes | Shapes read from tensors; covered by the parity fixtures |
| Single stream erases video-pipeline overlap | Ring-buffer size must be ≥ DALI's prefetch depth. Measure stream-per-slot against DALI's internal queueing rather than assuming |
| CI cannot execute any of it | Compile in CI, `GTEST_SKIP()` at runtime, manual GPU gate — matches the existing TensorRT posture |
