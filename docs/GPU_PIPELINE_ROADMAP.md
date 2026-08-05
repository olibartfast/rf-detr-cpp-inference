# GPU Pipeline Roadmap — DALI preprocessing + CUDA segmentation postprocessing

Adds GPU preprocessing for all three tasks (detection, segmentation, keypoint) and
GPU postprocessing for segmentation, without a Triton server. The application
hosts DALI in-process through the DALI C API and launches its own CUDA kernels on
the same stream as the TensorRT execution context.

Reference architecture: the
[tritonic YOLO11-seg ensemble](../../tritonic/deploy/instance_segmentation/yolo11/ensemble/README.md)
(DALI preprocess → TensorRT → custom CUDA postprocess operators). This roadmap
keeps the tensor-contract discipline and the CPU-versus-GPU parity gate from that
work, but drops the Triton and DALI-operator-plugin layers — see
[Divergences from tritonic](#divergences-from-tritonic).

Each numbered step is one reviewable change with its own verification. Phases are
ordered so every phase leaves the tree building and the CPU path untouched.

## Scope

In scope:

- DALI GPU preprocessing for detection, segmentation and keypoint models, for
  both still-image and video-frame sources.
- CUDA segmentation postprocessing: score decode, top-k, box decode, per-instance
  mask resize and threshold, packed-mask D2H transfer.
- Device-side I/O on the TensorRT backend, opt-in and non-breaking for ONNX Runtime.
- Parity gate and benchmarks against the existing CPU path.

Out of scope (explicitly deferred):

- GPU postprocessing for detection and keypoint. Detection postprocess is 300×91
  sigmoids and a threshold — it is not a bottleneck, and moving it to the GPU
  costs a kernel launch plus a D2H round trip for no gain. Keypoint postprocess
  carries the Cholesky-to-covariance maths and per-class keypoint mapping, which
  is branch-heavy and better left on the CPU until profiling says otherwise.
- Batch size > 1. Every contract below fixes batch 1, as the current code does.
- GPU rendering. Drawing stays on the CPU (`src/media.cpp`).
- ONNX Runtime CUDA execution provider. GPU pipeline requires `USE_TENSORRT=ON`.

## Current baseline

| Stage | Location | Cost shape |
| --- | --- | --- |
| Preprocess | `src/media.cpp:190` `preprocess_bgr_image` | Scalar bilinear stretch to `res×res`, BGR→RGB, `/255`, then ImageNet mean/std in a second pass over 3·res² floats |
| H2D | `src/backends/tensorrt_backend.cpp:285` | Blocking `cudaMemcpy` of 3·res²·4 bytes (3.7 MB at 560) on the **default stream** |
| Inference | `src/backends/tensorrt_backend.cpp:297` | `enqueueV3(0)` + `cudaStreamSynchronize(0)` |
| D2H | `src/backends/tensorrt_backend.cpp:312` | Blocking `cudaMemcpy` of every output, including the full `[1, 300, 108, 108]` mask tensor (14 MB) |
| Detection post | `src/rfdetr_inference.cpp:134` | Per-query argmax over classes, threshold |
| Segmentation post | `src/rfdetr_inference.cpp:201` | Global top-k over 300×`num_classes` scores, then **one full-resolution bilinear mask resize per surviving detection** (`src/media.cpp:239`) |

The segmentation mask resize is the real target: `resize_threshold_mask` runs
single-threaded and produces `orig_w × orig_h` bytes per detection. At 1080p with
20 detections that is 41 M bilinear taps on one core, after a 14 MB D2H of mask
data that is mostly discarded.

## Target architecture

```text
image file (JPEG/PNG bytes)          video frame (BGR, host, from FFmpeg)
        |                                        |
        v                                        v
  DALI pipeline "encoded"                 DALI pipeline "frame"
  external_source IMAGE (cpu, uint8, ndim=1)   external_source FRAME (gpu, uint8, HWC)
  -> decoders.image(device="mixed")            -> color_space_conversion(BGR->RGB)
  -> resize(res, res)                          -> resize(res, res)
  -> crop_mirror_normalize(CHW, mean, std)     -> crop_mirror_normalize(CHW, mean, std)
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

One CUDA stream per inference context. DALI writes the input tensor, TensorRT
consumes it, the postprocess kernels consume TensorRT's output bindings, and only
the final packed results cross to the host. No intermediate synchronisation.

## Divergences from tritonic

These are the places where copying tritonic code verbatim produces a silently
wrong result. Treat this list as the review checklist for Phases 2 and 3.

1. **No letterbox.** RF-DETR preprocessing is a plain stretch to `res×res`
   (`src/media.cpp:200` — independent `scale_x`, `scale_y`, no padding). The
   tritonic YOLO pipelines apply `fn.paste` letterboxing with a `gain`/`pad`
   coordinate restoration. The RF-DETR DALI pipeline must have **no `fn.paste`**,
   and box decode must stay `scale_w = orig_w / res`, `scale_h = orig_h / res`.
2. **ImageNet normalisation, not `/255` only.** YOLO uses `mean=0, std=255`.
   RF-DETR uses `/255` followed by mean `{0.485, 0.456, 0.406}` / std
   `{0.229, 0.224, 0.225}`. Fold into DALI as `mean = m*255`, `std = s*255`.
3. **DETR head, no NMS.** RF-DETR emits 300 decoded queries in `cxcywh`
   normalised form. The tritonic YOLO plugin's candidate scan, score ranking and
   `ApplyNms` have **no analogue here** and must not be ported.
4. **Full per-query masks, not prototypes.** RF-DETR seg output is
   `masks[1, 300, mask_h, mask_w]` (`docs/export.md:91`). There is no
   `[32, 160, 160]` prototype tensor and no 32-coefficient dot product, so
   tritonic's `MaskSigmoidDot` / `BuildPackedMasks` kernel does not transfer.
   The RF-DETR kernel is a straight bilinear resize of one `mask_h × mask_w`
   slice to `orig_h × orig_w`, plus threshold.
5. **No sigmoid on masks.** `resize_threshold_mask` compares the raw value
   against `config_.mask_threshold`, which defaults to `0.0` (`src/main.cpp:72`) —
   i.e. a logit threshold. tritonic applies `sigmoid` then compares against `0.5`.
   These are equivalent at those specific defaults, and diverge at any other
   value. Keep the raw-logit comparison to match the CPU path.
6. **Detection and segmentation select differently.** Detection takes a per-query
   argmax over classes (`src/rfdetr_inference.cpp:140`); segmentation takes a
   global top-k over all 300×`num_classes` score pairs
   (`src/rfdetr_inference.cpp:212`), so one query can yield several detections.
   The GPU segmentation path must reproduce the global top-k, not the argmax.
7. **Class-index offset.** Both CPU paths subtract 1 from the class index to skip
   the background logit, then drop anything that lands outside the label list.
   The kernel must do the same before compaction, or detection counts will differ
   by however many queries pick the background class.
8. **DALI hosts preprocessing only.** tritonic wraps its CUDA postprocess in DALI
   operators because Triton can only schedule DALI or backend models. Without
   Triton that wrapping buys nothing and costs the operator schema, the
   `DALI_ENFORCE` boilerplate, pipeline serialisation, plugin `.so` loading and a
   forced `Workspace` output-resize round trip. Postprocess kernels go directly
   in `src/gpu/` and are called by our own code.

## Phase 0 — Verify assumptions and freeze contracts

### 0.1 — Confirm the DALI C API is usable from a plain C++ binary

**Files:** `docs/GPU_PIPELINE_ROADMAP.md` (record findings here)

The DALI C API (`dali/c_api.h`) provides everything the design needs:
`daliDeserializeDefault`, `daliSetExternalInputAsync`, `daliRun`,
`daliShareOutput`, `daliOutputCopy`, `daliOutputRelease`, plus
`daliLoadLibrary` for plugins. A newer C API 2.0 (`dali/dali.h`) exists and is
what the DALI TensorFlow plugin migrated to; this roadmap targets the legacy
`c_api.h` because it is stable, ships in the same artifacts, and is the surface
the DALI Triton backend itself uses.

Verify by extraction, not by assumption:

```bash
docker run --rm -v dali-export:/export nvcr.io/nvidia/tritonserver:25.12-py3 \
  sh -lc 'cp -a /opt/tritonserver/backends/dali/wheel/dali/nvidia/dali/. /export/'
docker run --rm -v dali-export:/dali alpine ls /dali /dali/include/dali
```

**Verify:** `include/dali/c_api.h` is present, and `libdali.so` exports the
`dali*` symbols above (`nm -D --defined-only libdali.so | grep daliDeserialize`).
If `c_api.h` is absent from the container's wheel, fall back to a pip wheel
(`pip download nvidia-dali-cuda120`, unzip, same layout) and record which
artifact carries the header. **Do not start Phase 3 until this is confirmed** —
it is the single assumption the DALI half of the plan rests on.

### 0.2 — Record the model contract

**Files:** `docs/GPU_PIPELINE_ROADMAP.md`

Dump TensorRT binding names, shapes, dtypes and IO modes for the detection,
segmentation and keypoint engines actually in use. The existing backend already
prints this (`src/backends/tensorrt_backend.cpp:110`); capture the output.

**Verify:** `mask_h`, `mask_w`, `num_queries` and `num_classes` recorded per
model, and the kernels in Phase 2 read them from the shapes rather than
hardcoding — tritonic's kernels hardcode `kPrototypeWidth = 160` and
`kInputSize = 640` and are consequently locked to one engine.

### 0.3 — Golden CPU fixtures

**Files:** `tests/data/gpu_parity/`, `tests/unit/test_gpu_parity.cpp`

For a small, a wide, and a tall image, save the CPU-produced preprocessed tensor
and the final detections/masks with explicit tolerances.

Add a **dense synthetic fixture**: an image engineered to produce more than 100
above-threshold detections. tritonic learned this the hard way — stock photos
yield 10–50 detections, below any cap, so they cannot distinguish a truncating
postprocessor from a correct one.

**Verify:** CPU path reproduces the fixtures twice consecutively, bit-identically.

### 0.4 — Baseline benchmark

**Files:** `tests/benchmark/bench_gpu_pipeline.cpp`, `CMakeLists.txt`

Extend the benchmark target past the current microbenchmarks
(`tests/benchmark/bench_preprocessing.cpp` only covers `sigmoid`,
`cxcywh_to_xyxy` and `normalize_image`) to time the four stages end to end:
preprocess, H2D+infer, D2H, postprocess — separately, for a still image and for
a video run.

**Verify:** Numbers recorded in `docs/` for the CPU path at 560×560 detection and
at segmentation with a 1080p source. These are the numbers every later phase is
measured against.

## Phase 1 — GPU plumbing

No behaviour change. Everything here is dead code until Phase 2 calls it.

### 1.1 — `GpuContext`

**Files:** `src/gpu/gpu_context.hpp`, `src/gpu/gpu_context.cpp`

RAII owner of the device ordinal and one non-default `cudaStream_t`, plus a
`CUDA_CHECK` macro that throws `std::runtime_error` with the kernel/API name.
Header must be includable from non-CUDA translation units — expose the stream as
an opaque `void *` (or a `rfdetr::gpu::StreamHandle` alias) and keep
`cuda_runtime.h` out of it. Only `.cu` files and the TensorRT backend see the
real type.

**Verify:** Builds under `-DUSE_GPU_PIPELINE=ON`; a unit test constructs and
destroys a context, and skips cleanly when `cudaGetDeviceCount` reports 0 (CI has
no GPU).

### 1.2 — Device I/O on the backend interface

**Files:** `src/backends/inference_backend.hpp`, `src/backends/tensorrt_backend.{hpp,cpp}`

Add default-`false` / default-throwing virtuals so ONNX Runtime is untouched:

```cpp
[[nodiscard]] virtual bool supports_device_io() const { return false; }
virtual void run_inference_device(const void *input_device,
                                  const std::vector<int64_t> &input_shape);
[[nodiscard]] virtual const void *get_output_device_ptr(size_t index) const;
[[nodiscard]] virtual void *stream() const { return nullptr; }
virtual void synchronize() {}
```

`run_inference_device` sets the input tensor address to a caller-owned device
buffer, enqueues on the context stream, and returns without synchronising or
copying anything back. `get_output_device_ptr` hands out the existing
`device_buffers_` entries.

**Verify:** ONNX Runtime build unchanged (`ctest -R UnitTests` green, no new
symbols in the ORT binary). TensorRT build: a test runs the device path with a
manually `cudaMalloc`'d input and compares `get_output_device_ptr` contents
against `get_output_data` from the host path — must be bit-identical.

### 1.3 — Move TensorRT off the default stream

**Files:** `src/backends/tensorrt_backend.cpp`

The current code enqueues on stream `0` and calls `cudaStreamSynchronize(0)` after
every inference (`src/backends/tensorrt_backend.cpp:297`), which serialises
against everything else on the device. Take the `GpuContext` stream, use
`enqueueV3(stream)`, and make the host path's copies `cudaMemcpyAsync` on that
stream followed by one sync — so the existing host path keeps its current
semantics while the device path can skip the sync entirely.

**Verify:** Detection and segmentation results on the golden fixtures are
unchanged, bit for bit, on the host path. Benchmark shows no regression.

## Phase 2 — CUDA segmentation postprocessing

Independently useful: it works with the existing CPU preprocessing, so it can
land and be measured before DALI exists.

### 2.1 — Output contract

**Files:** `src/gpu/rfdetr_postprocess.hpp`

Borrow tritonic's packed-mask layout, which is the right shape for a single D2H:

| Buffer | Type | Shape |
| --- | --- | --- |
| `count` | `int32` | `[1]` |
| `boxes` | `float32` | `[max_detections, 4]` xyxy in original-image pixels |
| `scores` | `float32` | `[max_detections]` |
| `classes` | `int32` | `[max_detections]` |
| `mask_offsets` | `int64` | `[max_detections + 1]` prefix sums into `mask_data` |
| `mask_data` | `uint8` | `[sum of orig_w*orig_h per detection]` — 0 or 255 |

`mask_offsets` is always full-length even at `count == 0`, so the host reads a
fixed stride and can reject a short buffer. Masks are full-frame here (the CPU
path produces `orig_w × orig_h` per detection, `src/rfdetr_inference.cpp:251`),
not box-cropped as in tritonic — a later optimisation is to crop to the box and
carry the origin, but that changes `rfdetr::media::Mask` and the drawing code, so
it is deliberately not in this phase.

**Verify:** Header compiles standalone; a host-side reference implementation of
the unpacking round-trips a hand-built buffer.

### 2.2 — Score decode and top-k

**Files:** `src/gpu/rfdetr_postprocess.cu`

- `decode_scores`: one thread per `(query, class)` pair — `sigmoid(logit)` into a
  `[num_queries * num_classes]` score array. 300×91 = 27 300 elements.
- Top-k: `cub::DeviceRadixSort::SortPairsDescending` over scores with the flat
  index as value, then a compaction kernel that walks the first
  `max_detections` entries, applies the confidence threshold, recovers
  `query = idx / num_classes` and `class = idx % num_classes - 1`, drops
  out-of-range classes, and appends survivors with `atomicAdd` on `count`.

CUB is header-only and ships with the CUDA Toolkit, so this adds no dependency.
Order matters: threshold **after** ranking, and cap the output not the input —
divergence 6 and tritonic's own comment on candidate capping both apply.

**Verify:** Against the golden fixtures, the GPU `(count, scores, classes)`
triple matches the CPU path exactly for the class/count, and within `1e-5` for
scores. Ties in the score sort are the one legitimate source of ordering
difference — assert on the *set* of detections, not the order, and document that.

### 2.3 — Box decode

**Files:** `src/gpu/rfdetr_postprocess.cu`

One thread per surviving detection: `cxcywh × res` → `xyxy` → scale by
`(scale_w, scale_h)` → clamp to `(orig_w, orig_h)`. Mirrors
`processing_utils.cpp` exactly, including the clamp bound difference between the
detection path (`scale_w * res`) and the segmentation path (`orig_w`) — the
segmentation kernel uses `orig_w`/`orig_h`.

**Verify:** Max absolute box difference against CPU ≤ `1e-3` px on all fixtures.

### 2.4 — Mask resize and threshold

**Files:** `src/gpu/rfdetr_postprocess.cu`

Two kernels:

1. `compute_mask_offsets`: prefix sum of `orig_w * orig_h` per detection
   (`cub::DeviceScan::ExclusiveSum`), then a D2H of the last element to size
   `mask_data`. This is the one unavoidable mid-pipeline sync; allocate
   `mask_data` from a pool sized to `max_detections * orig_w * orig_h` once at
   startup to avoid a per-frame `cudaMalloc`.
2. `resize_threshold_masks`: one thread per output mask pixel. Locate the owning
   detection from `mask_offsets` (binary search, not tritonic's linear
   `while` walk — at 100 detections and millions of pixels the linear scan is
   measurable), bilinear-sample the `[mask_h, mask_w]` slice at
   `src = (dst + 0.5) * scale - 0.5` with the same clamping as
   `src/media.cpp:253`, compare against `mask_threshold`, write 0 or 255.

Use `float` accumulation, not tritonic's `double` — the CPU reference is `float`
(`src/media.cpp:266`), and matching it matters more than extra precision.

**Verify:** Per-detection mask IoU against the CPU masks ≥ 0.999 on every
fixture, and byte-exact on a synthetic mask whose values sit far from the
threshold. Report any detection whose IoU drops below the gate with its index and
box, so a single bad instance is debuggable.

### 2.5 — Wire into `RFDETRInference`

**Files:** `src/rfdetr_inference.{hpp,cpp}`, `src/main.cpp`

Add `postprocess_segmentation_outputs_gpu(...)` with the same signature as the
CPU version, filling the same `std::vector` outputs from one D2H of the packed
buffers. Select at runtime with `--gpu-postprocess`, defaulting off, so both
paths live in one binary and parity is a flag flip.

**Verify:** `--segmentation` with and without `--gpu-postprocess` on the same
image produces visually identical output and passes the 2.4 gate. Benchmark the
segmentation stage against the 0.4 baseline.

## Phase 3 — DALI GPU preprocessing

### 3.1 — Pipeline generators

**Files:** `deploy/dali/generate_preprocess_pipeline.py`, `deploy/requirements-dali.txt`

Two serialised pipelines, both batch 1, parameterised by `--resolution`,
`--mean`, `--std`:

- `preprocess_encoded.dali` — `external_source(name="IMAGE", device="cpu", ndim=1, dtype=UINT8)`
  → `decoders.image(device="mixed", output_type=RGB)`
  → `resize(resize_x=res, resize_y=res, interp_type=INTERP_LINEAR, antialias=False)`
  → `crop_mirror_normalize(dtype=FLOAT, output_layout="CHW", mean=m*255, std=s*255)`
- `preprocess_frame.dali` — `external_source(name="FRAME", device="gpu", ndim=3, dtype=UINT8, layout="HWC")`
  → `color_space_conversion(image_type=BGR, output_type=RGB)` → same resize + CMN.

`antialias=False` is required: DALI antialiases when downscaling by default,
while `preprocess_bgr_image` is a plain 4-tap bilinear sample. Without it the
tensors diverge on every image larger than `res`, most on the largest ones.

No `fn.paste`, no `fn.peek_image_shape`-driven aspect maths — see divergence 1.
Original size comes from the caller, which already has it
(`src/rfdetr_inference.cpp:83`), so the pipeline does not need a second output.

**Verify:** `python deploy/dali/generate_preprocess_pipeline.py --resolution 560`
serialises both files; a Python-side check feeds a fixture and compares the
tensor against the 0.3 golden with `max |Δ| ≤ 2e-2` (resampling and
`float` ordering differ; this is a tolerance gate, not an equality gate — do not
promise bit parity for resize).

### 3.2 — `DaliPreprocessor` wrapper

**Files:** `src/gpu/dali_preprocessor.{hpp,cpp}`

RAII wrapper over the C API: `daliDeserializeDefault` at construction from the
serialised file; `daliLoadLibrary` only if a plugin is ever needed (it is not,
for preprocessing); `daliDeletePipeline` in the destructor.

Per call: `daliSetExternalInputAsync` with the encoded bytes or the device frame
→ `daliRun` → `daliShareOutput` → read the output device pointer → hand it to
`run_inference_device` → `daliOutputRelease` **after** the TensorRT enqueue, not
before. Getting that order wrong hands DALI's buffer back to its pool while
TensorRT is still reading it, which produces intermittent garbage rather than a
crash — call it out in the code comment.

Set `prefetch_queue_depth = 1` for the still-image path (prefetching a single
image is pure latency) and match the video ring-buffer depth for the video path.

**Verify:** Feed the 0.3 fixtures through `DaliPreprocessor`, `cudaMemcpy` the
result to the host, and apply the 3.1 tolerance gate from C++. Run 1000
iterations under `compute-sanitizer` — clean, and no growth in
`cudaMemGetInfo` free bytes.

### 3.3 — Still-image path

**Files:** `src/rfdetr_inference.{hpp,cpp}`, `src/main.cpp`

`preprocess_image_gpu(path)` reads the file bytes and feeds the `encoded`
pipeline — the whole image decode moves to nvJPEG and the 3.7 MB float H2D
becomes a ~100 KB byte H2D. Behind `--gpu-preprocess`, default off.

**Verify:** Detection, segmentation and keypoint each produce results within the
0.3 tolerances with `--gpu-preprocess`. Non-JPEG inputs (PNG) still work — DALI's
mixed decoder falls back to host decode for formats nvJPEG does not handle, which
is correct but slower; note it rather than treat it as a bug.

### 3.4 — Video path

**Files:** `src/video_pipeline.{hpp,cpp}`

`FrameSlot` currently holds a host `std::vector<float> tensor`
(`src/video_pipeline.hpp:25`). Add a per-slot device buffer for the decoded BGR
frame; `preprocess_stage` becomes "H2D the frame, feed the `frame` pipeline,
publish the device tensor pointer" instead of filling a host tensor.

Two things to get right:

- Ring-buffer size must be ≥ DALI's prefetch depth, or the pipeline serialises on
  `daliShareOutput`.
- A single shared stream removes the overlap the four-stage pipeline exists to
  create. Either give each in-flight slot its own stream, or let DALI's internal
  queueing own the depth and keep one stream — measure both; do not assume.

**Verify:** Full video run with `--gpu-preprocess` produces the same frame count
and per-frame detection counts as the CPU path within tolerance. `nvidia-smi dmon`
shows GPU utilisation up and the preprocess thread's CPU time down. No leak over
a 1000-frame run.

## Phase 4 — Build, deps, CI

### 4.1 — Dependency declarations

**Files:** `cmake/deps/packages/CUDAToolkit.cmake`, `cmake/deps/packages/DALI.cmake`

Follow the existing `deps_declare` idiom (`cmake/deps/packages/TensorRT.cmake`).

CUDA Toolkit: `APT ON` (`nvidia-cuda-toolkit`) with `find_package(CUDAToolkit)`
as the provided path; CUB comes with it.

DALI is the awkward one. NVIDIA ships no standalone C++ tarball — the libraries
and headers live inside a pip wheel whose filenames carry an unguessable build
number (`nvidia_dali_cuda120-1.50.0-<build>-py3-none-manylinux2014_x86_64.whl`),
so a pinned `PROVIDED_URL` is not reliable. Two acquisition strategies, in order:

1. **Container extraction** (primary — already proven in
   `tritonic/.../dali_plugin/build_plugin.sh`): copy
   `/opt/tritonserver/backends/dali/wheel/dali/nvidia/dali/.` out of a pinned
   `nvcr.io/nvidia/tritonserver:<tag>` image. Gives `include/` and `libdali.so`
   in one directory, which is exactly what `DALI_ROOT` wants.
2. **Wheel download** — `pip download nvidia-dali-cuda120==<version>` then
   `file(ARCHIVE_EXTRACT)` (a wheel is a zip). Needs pip at configure time.

Respect `DEPS_OFFLINE` in both, per the existing handler contract.

**Verify:** `-DDEPS_DEBUG=ON` reports which handler resolved DALI and CUDAToolkit;
configure succeeds in `apt`, `conan` and `vcpkg` modes with the GPU pipeline off,
and in `apt` mode with it on.

### 4.2 — CMake options and CUDA language

**Files:** `CMakeLists.txt`

- `option(USE_DALI ...)` and `option(USE_CUDA_POSTPROCESS ...)`, both `OFF`, plus
  `USE_GPU_PIPELINE` as a convenience that turns on both.
- Either implies `USE_TENSORRT` — `FATAL_ERROR` if `USE_ONNX_RUNTIME` is the
  selected backend, with a message that names the reason (no device pointers).
- `enable_language(CUDA)` and `CMAKE_CUDA_ARCHITECTURES` (default `86`, matching
  tritonic) only inside the guard, so the default build needs no `nvcc`.
- `.cu` sources get their own warning flags — the project's `-Wold-style-cast`,
  `-Wconversion`, `-Wdouble-promotion` set does not survive contact with CUDA
  headers. Do not weaken the C++ flags to accommodate them; scope them with
  `$<COMPILE_LANGUAGE:CXX>`.
- RPATH for `libdali*.so` alongside the existing TensorRT RPATH handling.

**Verify:** All four combinations configure and build: ORT default, TensorRT only,
TensorRT + CUDA postprocess, TensorRT + CUDA postprocess + DALI. `-DWERROR=ON`
stays clean.

### 4.3 — Presets, Docker, CI

**Files:** `CMakePresets.json`, `Dockerfile`, `.github/workflows/`

A `gpu-pipeline` preset. A Docker stage based on
`nvcr.io/nvidia/tensorrt:<tag>` with the DALI libraries staged in. CI compiles the
GPU targets (nvcc is available on the runners; a GPU is not) and skips execution —
same posture the repo already takes for TensorRT: "CI does not test TensorRT
backend; test manually" (`AGENTS.md`). Gate the GPU tests on a runtime
`cudaGetDeviceCount` check and `GTEST_SKIP()`.

**Verify:** CI green with GPU targets compiled and GPU tests skipped, and the
skip is visible in the test output rather than silent.

### 4.4 — Documentation

**Files:** `README.md`, `docs/export.md`, `AGENTS.md`, `CHANGELOG.md`

Required by the repo's own release-documentation rule (`AGENTS.md`): new build
options, the DALI/CUDA version constraints, the pipeline generation step, and the
new runtime flags all go in `README.md` in the same change. Add the DALI
generator's pip requirement to the export tooling docs.

**Verify:** README statements cross-checked against `CMakeLists.txt`,
`CMakePresets.json`, `deploy/requirements*.txt` and `Dockerfile`, per the existing
checklist.

## Phase 5 — Parity gate and benchmarks

### 5.1 — Parity harness

**Files:** `tests/integration/integration_test_gpu_parity.cpp`

Run every fixture through all four combinations (CPU/CPU, GPU-pre/CPU-post,
CPU-pre/GPU-post, GPU/GPU) and assert: preprocessed tensor `max |Δ| ≤ 2e-2`;
detection sets match on class and count with scores within `1e-3`; box centres
within 1 px; mask IoU ≥ 0.999.

**Verify:** All four pass on the dense fixture as well as the natural images.

### 5.2 — Four-path benchmark

**Files:** `tests/benchmark/bench_gpu_pipeline.cpp`, `docs/`

Same four combinations, per-stage timings, still image and video, against the 0.4
baseline.

**Verify:** Segmentation postprocess shows a large improvement (the mask resize
is the whole point). Be prepared for **preprocessing to show little or no
end-to-end gain on single still images** — at 560×560 the CPU preprocess is
~1–2 ms and DALI adds its own launch overhead; the wins are the eliminated 3.7 MB
H2D, the freed CPU in the video pipeline's preprocess stage, and headroom at
higher resolutions. Record what the numbers actually say, including where they
are flat.

## Exit gate

The GPU pipeline is done when, on real hardware:

1. All three tasks run with `--gpu-preprocess` inside the 5.1 tolerances.
2. Segmentation runs with `--gpu-postprocess` at mask IoU ≥ 0.999, including on
   the dense fixture.
3. A 1000-frame video run completes with no leak and no `compute-sanitizer`
   findings.
4. The default (ORT, CPU) build and its results are bit-identical to today.
5. Benchmarks recorded, including the flat ones.
6. README/CHANGELOG updated per `AGENTS.md`.

## Risks

| Risk | Mitigation |
| --- | --- |
| `dali/c_api.h` absent from the container wheel | Step 0.1 verifies before any DALI work; pip-wheel fallback recorded |
| DALI resize never bit-matches the CPU bilinear | Tolerance gate from the start; never promise equality for resampling |
| DALI version coupled to CUDA/TensorRT versions | Pin the container tag; record the triple in README next to the existing TensorRT pin |
| `daliOutputRelease` ordering bug | Explicit ordering rule in 3.2 plus a `compute-sanitizer` run in the gate |
| Kernels hardcode one engine's shapes | 0.2 requires reading `mask_h`/`num_queries`/`num_classes` from tensor shapes |
| Single stream erases video-pipeline overlap | 3.4 measures stream-per-slot against DALI queueing rather than assuming |
| CI cannot execute any of it | Compile in CI, `GTEST_SKIP()` at runtime, manual GPU gate — matches the existing TensorRT posture |
