# Roadmap

Work this project has committed to, as a phased queue. Phases are ordered so each one leaves the tree building and the CPU path untouched. Release history lives in [CHANGELOG.md](../CHANGELOG.md) and is not repeated here.

## Status

- Last tag **v0.4.0** (2026-08-04). `develop` is **49 commits ahead of `master`**.
- The rfdetr **1.9.1** and **1.9.2** alignments and the whole GPU pipeline are unreleased on `develop`, so **v0.5.0 is staged**.
- Flow is git-flow: `develop` → `release/vX.Y.Z` → `master`, merged back.
- GPU pipeline: preprocessing and segmentation postprocessing **work end to end**; what remains is the test, build, and CI scaffolding around them (Phases 2–4).

---

## Phase 1 — Close the known issues

From the Known Issues table in [CHANGELOG.md](../CHANGELOG.md). Independent of each other and of everything below.

- Add segmentation export to `deploy/export_executorch.py`
  - `--model_type` offers only detection classes; the script instantiates `RFDETRNano`…`RFDETR2XLarge`, never `RFDETRSeg*`
  - Not an upstream or runtime limitation: `rfdetr` exports `RFDETRSegMedium` to `.pte` without error, `ExecuTorchBackend::validate_output_order()` inspects only outputs 0 and 1 so a third `masks` output passes, and `postprocess_segmentation_outputs()` addresses outputs positionally
  - Segmentation `.pte` files must be hand-exported today — see [docs/backend-parity-segmentation-video.md](../docs/backend-parity-segmentation-video.md)
- Add an output-path flag to `src/main.cpp`
  - Video output is hardcoded to `output_video.mp4` in the working directory, so comparing backends requires running each from its own directory
  - Image output is likewise hardcoded to `output_image.jpg`
- Verify and close the `.gitignore` item
  - **Partly stale**: `*.pte` is now ignored (`.gitignore:7`), and `output_video.mp4` (`:13`) has no leading slash so it already matches at any depth
  - What remains: once the output flag above lands, arbitrary output filenames are no longer covered. Decide the ignore pattern with that flag, then strike the item from the CHANGELOG

---

## Phase 2 — GPU parity fixtures

The parity gate the GPU work was supposed to be measured against was never built. Everything in Phases 3 and 4 depends on it.

- Add golden CPU fixtures under `tests/data/gpu_parity/` (directory does not exist)
  - A small, a wide, and a tall image; save the CPU-produced preprocessed tensor and the final detections/masks with explicit tolerances
  - Add a **dense synthetic fixture** engineered to produce more than 100 above-threshold detections. Stock photos yield 10–50 detections, below any cap, so they cannot distinguish a truncating postprocessor from a correct one
  - **Verify:** the CPU path reproduces the fixtures twice consecutively, bit-identically
- Add `tests/unit/test_gpu_parity.cpp`
  - Follow `tests/unit/test_gpu_postprocess.cpp` for the `GTEST_SKIP()`-without-a-device pattern
- Add `tests/benchmark/bench_gpu_pipeline.cpp` and register it in `CMakeLists.txt`
  - `tests/benchmark/bench_preprocessing.cpp` currently covers only `sigmoid`, `cxcywh_to_xyxy`, and `normalize_image`
  - Time four stages separately — preprocess, H2D+infer, D2H, postprocess — for a still image and for a video run
  - **Verify:** baseline numbers recorded for the CPU path at 560×560 detection and at segmentation with a 1080p source. Every later phase is measured against these

---

## Phase 3 — GPU build and CI integration

The one incomplete part of the GPU pipeline's build work; dependency declarations and CMake options are already done.

- Add a `gpu-pipeline` configure preset to `CMakePresets.json` (none of the five existing presets covers TensorRT, ExecuTorch, OpenCV, or GPU)
- Add DALI staging and the GPU options to `Dockerfile` — it contains **zero** DALI references today
  - Base the stage on `nvcr.io/nvidia/tensorrt:<tag>` with the DALI libraries staged in
- Add a compile-only GPU job to CI
  - Compile the GPU targets and skip execution, matching the posture already taken for TensorRT. `nvcc` is available on runners; a GPU is not
  - **Verify:** CI green with GPU targets compiled and GPU tests skipped, and the skip **visible** in the test output rather than silent

---

## Phase 4 — GPU parity gate and benchmarks

- Add `tests/integration/integration_test_gpu_parity.cpp`
  - Run every fixture through all four combinations: CPU/CPU, GPU-pre/CPU-post, CPU-pre/GPU-post, GPU/GPU
  - Assert: preprocessed tensor `max |Δ| ≤ 2e-2`; detection sets match on class and count with scores within `1e-3`; box centres within 1 px; mask IoU ≥ 0.999
  - **Verify:** all four pass on the dense fixture as well as the natural images
- Extend the benchmark to the same four combinations, per-stage, still image and video
  - Expect a large improvement in segmentation postprocess — the mask resize is the whole point
  - Expect **little or no end-to-end gain from preprocessing on single still images**: at 560×560 the CPU preprocess is ~1–2 ms and DALI adds its own launch overhead. The wins are the eliminated 3.7 MB H2D, the freed CPU in the video pipeline's preprocess stage, and headroom at higher resolutions. Record what the numbers actually say, including where they are flat
- Run the exit gate on real hardware:
  1. All three tasks run with `--gpu-preprocess` inside the tolerances above
  2. Segmentation runs with `--gpu-postprocess` at mask IoU ≥ 0.999, including on the dense fixture
  3. A 1000-frame video run completes with no leak and no `compute-sanitizer` findings
  4. The default (ONNX Runtime, CPU) build and its results are bit-identical to today
  5. Benchmarks recorded, including the flat ones
  6. README and CHANGELOG updated per [AGENTS.md](../AGENTS.md)

Items 4 and 6 are already satisfied; 1, 2, 3, and 5 are not.

---

## Phase 5 — Release v0.5.0

- Read `AGENTS.md`, `README.md`, and `CHANGELOG.md`, then verify the rfdetr release against upstream — the mandatory "Release Documentation Sync" rule
- Move `[Unreleased]` to `[v0.5.0]`, sync `README.md` version statements against `CMakeLists.txt`, `CMakePresets.json`, `deploy/requirements.txt`, `Dockerfile`, and `docs/export.md`
- Resolve the version disagreement noted in [tech-stack.md](tech-stack.md#known-pin-duplications): `project()` declares none, `vcpkg.json` says `0.1.0`, the README badge says `0.4.0`
- Cut `release/v0.5.0`, merge to `master`, tag, merge back to `develop`

---

## Deferred

Not started, each for a recorded reason. Reopening one is a decision, not a task.

| Item | Why deferred |
|------|--------------|
| Batch size > 1 | Every tensor contract fixes batch 1, as the current code does |
| GPU postprocessing for detection | 300×91 sigmoids and a threshold — not a bottleneck. Moving it costs a kernel launch plus a D2H round trip for no gain |
| GPU postprocessing for keypoint | Cholesky-to-covariance maths and per-class keypoint mapping are branch-heavy; better on the CPU until profiling says otherwise |
| GPU rendering | Drawing stays on the CPU (`src/media.cpp`) |
| ONNX Runtime CUDA execution provider | The backend registers none; the GPU pipeline requires `USE_TENSORRT=ON` |
| Box-cropped masks | Masks are full-frame to match the CPU path. Cropping to the box and carrying the origin changes `rfdetr::media::Mask` and the drawing code |

**Standing obligation:** every upstream `rfdetr` release triggers an alignment pass. Read `AGENTS.md` first and verify against the upstream release notes before touching anything.

---

## Reference — GPU pipeline design

Constraints for the already-built code in `src/gpu/`. Read before modifying it.

### Architecture

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

### Model contract

RF-DETR does not behave like the CNN detectors most GPU pipeline code is written for. Each rule below is load-bearing: breaking one produces a **silently wrong result**, not a crash. This is the review checklist for any change to `src/gpu/`.

1. **No letterbox.** Preprocessing is a plain stretch to `res×res` — independent `scale_x`, `scale_y`, no padding (`src/media.cpp:212-213`). No `fn.paste` in the DALI pipeline, and box decode stays `scale_w = orig_w / res`, `scale_h = orig_h / res`.
2. **ImageNet normalisation, not `/255` alone.** `/255` then mean `{0.485, 0.456, 0.406}` / std `{0.229, 0.224, 0.225}`. Folded into DALI as `mean = m*255`, `std = s*255`.
3. **DETR head — no NMS.** The model emits 300 already-decoded queries in normalised `cxcywh`. There is nothing to suppress; no candidate scan and no score ranking belong in the pipeline.
4. **Full per-query masks.** Segmentation output is `masks[1, 300, mask_h, mask_w]` — one complete mask per query, no prototype tensor and no coefficient dot product. The kernel is a straight bilinear resize of one `mask_h × mask_w` slice plus threshold.
5. **No sigmoid on masks.** The raw value is compared against `mask_threshold`, which defaults to `0.0` — a logit threshold. Applying sigmoid and comparing against 0.5 is equivalent only at those defaults and diverges at any other value. Keep the raw-logit comparison.
6. **Detection and segmentation select differently.** Detection takes a per-query argmax over classes; segmentation takes a global top-k over all 300×`num_classes` score pairs, so one query can yield several detections. The GPU segmentation path must reproduce the global top-k.
7. **Class-index offset.** Both CPU paths subtract 1 from the class index to skip the background logit, then drop anything outside the label list. The kernel must do the same before compaction, or counts differ by however many queries pick background.
8. **DALI hosts preprocessing only.** Postprocess kernels live directly in `src/gpu/` and are called by our own code. There is no external scheduler to satisfy, so wrapping them in DALI operators would cost the operator schema, pipeline serialisation, and plugin loading for no benefit.

### Packed output contract

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

### Correctness rules

- **`daliOutputRelease` ordering.** Release **after** the TensorRT enqueue, never before. Getting this wrong hands DALI's buffer back to its pool while TensorRT is still reading it, producing intermittent garbage rather than a crash.
- **`antialias=False` in the DALI resize.** DALI antialiases when downscaling by default; `preprocess_bgr_image` is a plain 4-tap bilinear sample. Without it, tensors diverge on every image larger than `res`, worst on the largest.
- **`float` accumulation in the mask kernel, not `double`.** The CPU reference is `float`; matching it matters more than extra precision.
- **Threshold after ranking, and cap the output not the input.** See model contract rule 6.
- **Read shapes from tensors.** `mask_h`, `mask_w`, `num_queries`, `num_classes` come from the tensor shapes — never hardcode them, or the kernels lock to one engine.
- **Score-sort ties are the one legitimate ordering difference.** Assert on the *set* of detections, not the order.
- **Resize is a tolerance gate, never an equality gate.** DALI resize will not bit-match the CPU bilinear. Do not promise equality for resampling.

### Risks

| Risk | Mitigation |
|------|------------|
| DALI resize never bit-matches the CPU bilinear | Tolerance gate from the start; never promise equality for resampling |
| DALI version coupled to CUDA/TensorRT versions | Container tag pinned; the triple is recorded in [tech-stack.md](tech-stack.md) |
| `daliOutputRelease` ordering bug | Explicit ordering rule above, plus a `compute-sanitizer` run in the Phase 4 gate |
| Kernels hardcode one engine's shapes | Shapes read from tensors; covered by the parity fixtures |
| Single stream erases video-pipeline overlap | Ring-buffer size must be ≥ DALI's prefetch depth. Measure stream-per-slot against DALI's internal queueing rather than assuming |
| CI cannot execute any of it | Compile in CI, `GTEST_SKIP()` at runtime, manual GPU gate — matches the existing TensorRT posture |
