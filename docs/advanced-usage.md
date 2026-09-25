# Advanced Usage and Customization

Everything past the [README](../README.md)'s quick start: what each backend actually does,
the complete CMake option reference, runtime tuning, the opt-in GPU pipeline, and embedding
`RFDETRInference` in your own C++ code.

> Part of the [RF-DETR C++ Inference](../README.md) documentation.

---

## Table of Contents
- [Backends in Depth](#backends-in-depth)
- [Media & Display Backends](#media--display-backends)
- [CMake Option Reference](#cmake-option-reference)
- [Overriding a Pinned Version](#overriding-a-pinned-version)
- [Dependency Resolution Modes](#dependency-resolution-modes)
- [Tuning Inference at Runtime](#tuning-inference-at-runtime)
- [Customizing Labels and Class Layouts](#customizing-labels-and-class-layouts)
- [The GPU Pipeline at Runtime](#the-gpu-pipeline-at-runtime)
- [Performance Notes](#performance-notes)
- [Embedding the Library](#embedding-the-library)
- [CI Coverage and Its Limits](#ci-coverage-and-its-limits)

---

## Backends in Depth

Backend selection is a **compile-time** decision: exactly one inference backend is compiled
into the binary, and enabling two is a configure-time error. That keeps the binary small and
the hot path free of dispatch, at the cost of one build per backend. The usage text the binary
prints is specialized to whichever backend it was built with.

| Backend | Model format | Pros | Cons |
|---------|--------------|------|------|
| **ONNX Runtime** | `.onnx` | Easy setup, no GPU or extra SDK needed | CPU only as shipped (see below) |
| **TensorRT** | `.engine` / `.trt` (also accepts `.onnx`, building and caching an engine beside it) | Maximum performance | GPU-only, requires CUDA/TensorRT |
| **ExecuTorch** | `.pte` | Small runtime, delegate-based (XNNPACK) | Requires an ExecuTorch install; rfdetr 1.9.0+ to export |

### ONNX Runtime (default)

The version is `ONNX_RUNTIME_VERSION` in [`versions.env`](../versions.env). The official CPU
archive is downloaded automatically, selected from the **target** OS and architecture
(`CMAKE_SYSTEM_NAME` / `CMAKE_SYSTEM_PROCESSOR`) — so cross-compiling picks the target's
archive rather than the host's:

| Target | Archive |
|--------|---------|
| Linux x86_64 / amd64 | `onnxruntime-linux-x64-<version>.tgz` |
| Linux aarch64 / arm64 | `onnxruntime-linux-aarch64-<version>.tgz` |
| Windows x86_64 / amd64 | `onnxruntime-win-x64-<version>.zip` |
| Windows arm64 | `onnxruntime-win-arm64-<version>.zip` |

That table lists only the *automatic* downloads. Other targets supply a compatible build
through `-DONNXRUNTIME_ROOTDIR=<prefix>` or a package manager (`onnxruntime/<version>` for Conan,
`onnxruntime` for vcpkg). Loading the dependency catalog does not reject those targets when
ONNX Runtime is disabled; an enabled backend fails resolution only when no provider can
supply it.

**Acceleration is CPU only.** `OnnxRuntimeBackend` creates its session without appending an
execution provider, so even a CUDA or DirectML build of ONNX Runtime runs on the CPU here
until the backend is extended to register one. For GPU inference, use TensorRT.

### TensorRT

The version is `TENSORRT_VERSION` in [`versions.env`](../versions.env), downloaded automatically
during the build if not found. It needs the CUDA Toolkit series `CUDA_VERSION` pins, which must
be installed manually — the bundled TensorRT archive is built against it. (From 11.x NVIDIA names
the archive `TensorRT-Enterprise-….tar.zst`; that is the standard TensorRT, not a paid edition.)

- Linux with an NVIDIA GPU only.
- TensorRT libraries are configured with RPATH, so no `LD_LIBRARY_PATH` is needed.
- A pre-built `.engine` or `.trt` is loaded directly, skipping ONNX-to-TensorRT conversion.
  Passing an `.onnx` instead builds an engine and caches it beside the model — convenient for
  a first run, but the conversion cost is paid once per model and machine.
- **TensorRT 10.x** is still supported alongside the pinned 11.x; point `-DTENSORRT_ROOTDIR` at a
  10.x prefix and pass the matching `-DTENSORRT_VERSION` (and `-DCUDA_VERSION` for its CUDA
  series). CI compile-checks the backend against `TENSORRT_LEGACY_VERSION` (10.x, with
  `DALI_LEGACY_VERSION`) and against the newer `TENSORRT_COMPAT_VERSION`. TensorRT 11 builds
  strongly typed engines only: on 11.x the engine takes the ONNX model's own precision — convert
  it first for FP16 ([export guide](export.md#tensorrt-11-and-fp16)); on 10.x an `.onnx` gets an
  FP16 engine through the FP16 builder flag.
- Every engine input and output must be float32; the backend rejects an engine that is not.
- Engines are tied to the TensorRT version that built them — rebuild cached `.engine` files
  after switching TensorRT versions.
- It is also the only backend that can drive the [GPU pipeline](#the-gpu-pipeline-at-runtime):
  DALI writes into, and the CUDA kernels read from, the inference engine's device buffers, and
  only this backend exposes device pointers and a CUDA stream.

Build steps: [building.md](building.md#build-with-tensorrt-backend).

### ExecuTorch

The version is `EXECUTORCH_VERSION` in [`versions.env`](../versions.env), resolved from an install
prefix via `-DEXECUTORCH_ROOTDIR`, otherwise built from source (slow — and it needs a Python interpreter carrying ExecuTorch's build
dependencies, since a bare `python3` cannot `import torchgen`).

- Model format is `.pte`, exported by `rfdetr[executorch]` 1.9.0 or newer.
- The delegate linked with `-DEXECUTORCH_DELEGATE` (`xnnpack` by default, or `portable`) **must
  match** the delegate baked into the `.pte` at export time. A mismatch fails at run time, not
  at link time.
- The prefix must be built with `-DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON`, which defaults to
  `OFF`: `.pte` files from rfdetr 1.9.1+ call `aten::linear.out`, which only
  `optimized_native_cpu_ops_lib` registers.
- Linux; CPU inference through the linked delegate.

The full prefix recipe, including why exactly one op library may be linked, is in
[Building the ExecuTorch install prefix](building.md#building-the-executorch-install-prefix).

---

## Media & Display Backends

Image load/save, video decode/encode, and the `--display` preview window are backed by either
**FFmpeg + SDL2 + stb** (the default) or **OpenCV**. Exactly one is compiled in, via
`-DUSE_OPENCV=ON/OFF`. This choice is *orthogonal* to the inference backend — combine them
freely.

| | Default (`-DUSE_OPENCV=OFF`) | OpenCV (`-DUSE_OPENCV=ON`) |
|---|---|---|
| Image I/O | stb ([third_party/stb](../third_party/stb), vendored — no install) | `imgcodecs` |
| Video decode/encode | FFmpeg 5.x+ (`libavcodec`, `libavformat`, `libavutil`, `libswscale`) | `videoio` |
| `--display` window | SDL2 2.x | `highgui` |
| apt packages | `libavcodec-dev libavformat-dev libavutil-dev libswscale-dev libsdl2-dev` | `libopencv-dev` |

Enabling OpenCV replaces FFmpeg, SDL2 **and** stb entirely — none of them are required or
linked. Annotation text is drawn with an 8x8 bitmap font
([third_party/font8x8](../third_party/font8x8)) under **both** backends.

Worked build: [building.md](building.md#build-with-opencv-mediadisplay-backend).

---

## CMake Option Reference

The complete list. The README carries only the handful most builds need.

### Inference backend

| Option | Default | Effect |
|--------|---------|--------|
| `-DUSE_ONNX_RUNTIME=ON/OFF` | `ON` | ONNX Runtime backend |
| `-DUSE_TENSORRT=ON/OFF` | `OFF` | TensorRT backend |
| `-DUSE_EXECUTORCH=ON/OFF` | `OFF` | ExecuTorch backend for `.pte` models |
| `-DEXECUTORCH_ROOTDIR=<path>` | — | ExecuTorch install prefix; without it ExecuTorch is built from source |
| `-DEXECUTORCH_DELEGATE=xnnpack/portable` | `xnnpack` | ExecuTorch delegate library to link |

`USE_ONNX_RUNTIME` defaults to `ON`, so it must be turned **off** explicitly when selecting
another inference backend — enabling two is a configure-time error.

### Media / display backend

| Option | Default | Effect |
|--------|---------|--------|
| `-DUSE_OPENCV=ON/OFF` | `OFF` | Use OpenCV for image/video/display I/O instead of FFmpeg+SDL2+stb |

### GPU pipeline (TensorRT only)

| Option | Default | Effect |
|--------|---------|--------|
| `-DUSE_DALI=ON/OFF` | `OFF` | DALI GPU preprocessing; requires the TensorRT backend and `-DDALI_ROOT` |
| `-DUSE_CUDA_POSTPROCESS=ON/OFF` | `OFF` | CUDA segmentation postprocessing; requires the TensorRT backend and `nvcc` |
| `-DUSE_GPU_PIPELINE=ON/OFF` | `OFF` | Convenience switch that enables both of the above |
| `-DDALI_ROOT=<path>` | — | Directory holding the staged DALI libraries/headers (from `./scripts/fetch_dali.sh`) |
| `-DCMAKE_CUDA_ARCHITECTURES=<list>` | `86` (RTX 30-series) | CUDA architectures for the postprocessing kernels |

Either GPU option combined with the ONNX Runtime backend is a configure-time `FATAL_ERROR`.
The two halves are independent — `-DUSE_DALI=ON` needs no `nvcc`, `-DUSE_CUDA_POSTPROCESS=ON`
does.

What each half resolves: the CUDA Toolkit comes from CMake's `FindCUDAToolkit`, and CUB —
header-only, bundled with the toolkit — is used by the postprocessing kernels. DALI is
ROOT-only: NVIDIA ships no standalone C++ DALI distribution, so `./scripts/fetch_dali.sh`
extracts the C API libraries and headers from a pinned Triton container
(`nvcr.io/nvidia/tritonserver:<NGC_CONTAINER_TAG>-py3`) into `~/dependencies/dali`, which is what you point
`-DDALI_ROOT` at.

### Build type and diagnostics

| Option | Default | Effect |
|--------|---------|--------|
| `-DCMAKE_BUILD_TYPE=Release/Debug` | — | Build configuration |
| `-DWERROR=ON/OFF` | `OFF` | Treat compiler warnings as errors (CI uses `ON`) |
| `-DBENCHMARKS=ON/OFF` | `OFF` | Build the Google Benchmark targets |
| `-DSANITIZERS=ON/OFF` | `OFF` | AddressSanitizer + UndefinedBehaviorSanitizer |
| `-DSTRICT_UBSAN=ON/OFF` | `OFF` | Stricter UBSan — Clang: `undefined,local-bounds,vptr,implicit-conversion`; GCC: `undefined,bounds-strict,vptr` |
| `-DTHREAD_SANITIZER=ON/OFF` | `OFF` | ThreadSanitizer / data-race detection |

The three sanitizer modes are **mutually exclusive** — pick one. Valgrind needs a plain `Debug`
build with no sanitizer at all. Commands for each:
[development.md](development.md#sanitizers-optional).

### Dependency resolution

| Option | Default | Effect |
|--------|---------|--------|
| `-DDEPS_MODE=apt/conan/vcpkg/auto` | `apt` | Package-manager ecosystem used to resolve dependencies |
| `-DDEPS_OFFLINE=ON/OFF` | `OFF` | Disable network lookups; ROOT-provided lookups only |
| `-DDEPS_DEBUG=ON/OFF` | `OFF` | Log which handler resolved each dependency |
| `-DDEPS_PROVIDED_DIR=<path>` | `<build>/_deps` | Where provided-download archives extract |
| `-DDEPS_CONAN_DIR=<path>` | — | Conan CMakeDeps output dir — consumes prebuilt binaries without the Conan toolchain overriding the system compiler |

`-DDEPS_OFFLINE=ON` is the switch for air-gapped builds: combine it with
`-DONNXRUNTIME_ROOTDIR`, `-DEXECUTORCH_ROOTDIR`, or `-DDALI_ROOT` so every dependency comes
from a local prefix.

### Prefixes and paths

Point the build at a dependency you already have, instead of letting it download or build one.
Each is a cache variable, so `-D` on the command line wins.

| Option | Default | Effect |
|--------|---------|--------|
| `-DONNXRUNTIME_ROOTDIR=<path>` | — | Prebuilt ONNX Runtime prefix, instead of the automatic download |
| `-DTENSORRT_ROOTDIR=<path>` | — | TensorRT prefix, instead of the automatic download (`TensorRT_ROOT` also works) |
| `-DEXECUTORCH_ROOTDIR=<path>` | — | ExecuTorch install prefix; without it ExecuTorch is built from source |
| `-DDALI_ROOT=<path>` | — | Staged DALI libraries/headers from `./scripts/fetch_dali.sh` (`DALI_ROOTDIR` is the same variable) |
| `-DRFDETR_VERSIONS_ENV=<file>` | `versions.env` at the repo root | Which pin file `cmake/versions.cmake` reads |

Individual version pins are cache variables too — `-DONNX_RUNTIME_VERSION=…`,
`-DTENSORRT_VERSION=…`, and so on. See [Overriding a Pinned Version](#overriding-a-pinned-version).

### Valgrind targets

Created only when Valgrind is found, on a plain `Debug` build with no sanitizer.

| Option | Default | Effect |
|--------|---------|--------|
| `-DVALGRIND_MEMCHECK_OPTS=<flags>` | `--error-exitcode=1 --leak-check=full --show-leak-kinds=definite,indirect --errors-for-leak-kinds=definite,indirect` | Flags the `memcheck` target passes to valgrind; `valgrind.supp` at the repo root is appended automatically when present |
| `-DVALGRIND_PROFILE_ARGS=<args>` | empty | Extra arguments the `callgrind`/`massif` targets pass to the profiled binary |

The targets themselves are in [development.md](development.md#valgrind--profiling-optional).

### Presets

`CMakePresets.json` carries the default build, four diagnostic CPU presets, and `gpu-pipeline`
for TensorRT + DALI + CUDA. ExecuTorch and OpenCV have no preset — pass their options
explicitly.

---

## Overriding a Pinned Version

[`versions.env`](../versions.env) is the single source of truth for every third-party pin.
Nothing else in the tree that *can* read a file repeats a version. You rarely need to edit it:
every pin is overridable in place, and neither loader clobbers a value already set.

```bash
# CMake pins take -D
cmake -S . -B build -DONNX_RUNTIME_VERSION=1.22.0
# A TensorRT release ships for one CUDA series, so a TensorRT override usually needs both
cmake -S . -B build -DTENSORRT_VERSION=10.14.1.48 -DCUDA_VERSION=13.0

# Shell-script pins take the environment
TRITON_IMAGE=nvcr.io/nvidia/tritonserver:26.01-py3 ./scripts/fetch_dali.sh
TENSORRT_VERSION=10.14.1.48 CUDA_VERSION=13.0 ./scripts/ci/stage_gpu_headers.sh
```

Some coordinates are **derived**, not stored — do not add variables for them.
`cmake/versions.cmake` derives `TENSORRT_SHORT_VERSION`; `scripts/versions.sh` derives that
plus `TENSORRT_DEB_VERSION`, `TRITON_IMAGE`, and `TENSORRT_IMAGE`.

To bump a pin for real, edit the one line in `versions.env` and then run:

```bash
./scripts/check_version_sync.sh
```

It reports the five formats that cannot read a file — the backend Dockerfiles' `ARG` defaults,
`conanfile.txt`, `deploy/requirements.txt`, the `deploy/export_*.py` opset defaults, and the
two version tables in `README.md` — and fails until they match. CI runs it as the
`Version Sync` job. No other prose states a pinned version: `docs/` and `specs/` name the
`versions.env` variable instead, so a bump needs no prose edits. Full procedure:
[Bumping a version](../specs/tech-stack.md#bumping-a-version).

---

## Dependency Resolution Modes

Every dependency flows through a unified facade (`find_dependency_unified`) that picks an
acquisition strategy per `-DDEPS_MODE`:

| Mode | Chain | Use when |
|---|---|---|
| `apt` (default) | apt → provided | No extra tooling; system packages plus pinned downloads |
| `conan` | conan → apt → provided | ConanCenter binaries or a local cache |
| `vcpkg` | vcpkg → apt → provided | vcpkg manifest mode |
| `auto` | apt → conan → vcpkg → provided | Mixed: each dependency takes the fastest available |

`apt` is chained as a fallback in conan/vcpkg modes so system packages (Threads) still
resolve. Add `-DDEPS_DEBUG=ON` to see which handler answered for each dependency.

How the facade is built, and what each handler does:
[package-manager-architecture.md](package-manager-architecture.md). Worked conan/vcpkg
invocations: [building.md](building.md#dependency-resolution).

---

## Tuning Inference at Runtime

Nothing here requires a rebuild. [usage.md](usage.md#inference-parameters) lists every flag and
its default; this is when to reach for them.

- **`--threshold`** — raise it to cut false positives, lower it to catch faint objects. The
  default `0.5` is a middle setting, not a tuned one.
- **`--max-detections`** — the top-k cap on ranked *(query, class)* pairs. Lowering it from the
  default `300` cuts postprocessing work on crowded frames; raising it only helps if you are
  genuinely losing detections below the cap.
- **`--mask-threshold`** — the mask logit cutoff, segmentation only. It is a *logit*, so `0.0`
  is the even-odds point and negative values are legal. Raise it for tighter, more conservative
  masks.
- **`--resolution`** — leave it alone unless the model accepts a size other than the one
  recorded in its file. Auto-detection is correct for a normal export.

```bash
./build/inference_app model.onnx image.jpg coco-labels-91.txt --threshold 0.7 --max-detections 100
./build/inference_app model.onnx image.jpg coco-labels-91.txt --segmentation --mask-threshold 0.5
```

These compose with every mode — `--segmentation`, `--keypoint`, video input, and `--display`.

### How detections are selected

RF-DETR scores classes with independent sigmoids rather than a softmax, so one query can
legitimately clear the threshold on several classes at once. Postprocessing therefore ranks the
flattened *(query, class)* grid and keeps the top `--max-detections` pairs **before** applying
`--threshold`, which is what `PostProcess._select_topk` does upstream — a per-query argmax would
silently drop every class but the strongest (the bug rfdetr 1.9.3 fixed in its own
exported-model decoders).

Two consequences worth knowing:

- **Ordering is deterministic.** Results come back in descending-score order, with exact ties
  broken by ascending flattened query/class index, so a given model and image always produce the
  same ordering. Detection, segmentation, keypoint, and the CUDA postprocess kernels all share
  that rule.
- **`--max-detections` is a cap on *candidates*, not on results.** It bounds the ranked set that
  `--threshold` then filters, so the number of detections you actually get is usually far lower.

---

## Customizing Labels and Class Layouts

The third positional argument is a plain text file, one label per line. `data/coco-labels-91.txt`
ships with the repo and is indexed by COCO category id.

A **custom model** usually needs `--background-class-id` adjusted to match its exported logit
layout, because getting it wrong shifts every reported label by one:

| Export layout | Pass | Notes |
|---------------|------|-------|
| Background-first (the shipped RF-DETR exports) | *(nothing — `0` is the default)* | Logit 0 is background, logit *n* is COCO category *n* |
| Every slot a real class (fine-tuned, contiguous 0-based ids) | `--background-class-id none` | Keeps every slot |
| Background in the final slot | `--background-class-id -1` | Negative values count from the end |

This mirrors the `background_class_id` argument rfdetr 1.9.4 added to its own ONNX/TFLite
decoders. Note that upstream's own default is `-1`, which mis-decodes the shipped checkpoints —
this project defaults to `0` instead.

**Keypoint models** additionally take `--keypoint-counts` (`num_keypoints_per_class`, default
`0,17` for COCO's `{background: 0, person: 17}`). An active-first `[17]` export needs
`--keypoint-counts 17 --background-class-id none`. The keypoint *names*, the skeleton edges,
and the uncertainty overlay are `Config` fields with no CLI flag — see
[Embedding the Library](#embedding-the-library).

---

## The GPU Pipeline at Runtime

Opt-in, TensorRT-only, and **off by default even in a GPU-pipeline build** — the CPU paths stay
the default until you ask for otherwise.

```bash
./build/inference_app model.engine image.jpg coco-labels-91.txt \
  --segmentation --gpu-preprocess --gpu-postprocess
```

| Flag | Requires | Effect |
|------|----------|--------|
| `--gpu-preprocess` | built with `-DUSE_DALI=ON` | Decode/resize/normalize on the GPU with DALI |
| `--gpu-postprocess` | built with `-DUSE_CUDA_POSTPROCESS=ON` | Segmentation mask decode/resize/threshold in CUDA kernels; **segmentation only**, so pair it with `--segmentation` |
| `--dali-pipeline-dir <dir>` | `-DUSE_DALI=ON` | Where the serialized `.dali` pipeline files live (default `data/dali`) |

With `--gpu-preprocess`, the video pipeline's preprocess stage becomes a passthrough: DALI's
`frame` pipeline runs on the backend's CUDA stream inside the inference stage, and the CPU cost
of the bilinear resample leaves the pipeline entirely.

### A new input resolution

The serialized `.dali` pipelines are resolution-specific, and **432** and **576** are checked
in. For anything else, regenerate — this needs Docker with `--gpus all`:

```bash
./scripts/generate_dali_pipelines.sh <res>
```

### CUDA architectures

`-DCMAKE_CUDA_ARCHITECTURES` defaults to `86` (RTX 30-series). Set it to your card's
capability — e.g. `89` for Ada — when building the postprocessing kernels. Note that
`scripts/run_gate.sh` deliberately defaults to `CUDA_ARCH=89`, matching the hardware the
verification gate is usually rented on; it is not a pin.

Design constraints and how each half works: [architecture.md](architecture.md#gpu-pipeline) and
[specs/gpu-pipeline.md](../specs/gpu-pipeline.md). The build:
[building.md](building.md#build-with-the-gpu-pipeline-tensorrt--dali--cuda). Verifying it on
real hardware is a maintainer procedure, kept with the specs:
[specs/rented-gpu-runbook.md](../specs/rented-gpu-runbook.md).

---

## Performance Notes

- **Build `Release`.** `-DCMAKE_BUILD_TYPE=Release` is the only configuration worth timing;
  sanitizer and Valgrind builds are one to two orders of magnitude slower by design.
- **Feed TensorRT a pre-built engine.** Passing an `.onnx` makes it build and cache an engine
  first; a `.engine`/`.trt` is loaded directly.
- **Video already parallelizes.** Four `std::jthread`s run one stage each — decode, preprocess,
  infer+postprocess, draw+encode — over a ring of pre-allocated `FrameSlot`s (8 by default).
  Stages pass slot indices through bounded queues, so no frame is ever copied between them and
  backpressure is automatic. There is nothing to configure: the inference stage owns its own
  `RFDETRInference` instance and takes no locks on the hot path.
- **Lower `--max-detections`** to shrink the ranked candidate set on crowded frames.
- **Move preprocessing to the GPU** with `--gpu-preprocess` on a TensorRT + DALI build.
- **Measure before changing anything.** Build with `-DBENCHMARKS=ON` and run `./build/benchmarks`;
  for a profile, `cmake --build build-valg --target callgrind` (read with `callgrind_annotate`)
  or the lower-overhead `perf record ./build/benchmarks`. Details:
  [development.md](development.md#benchmarks).

---

## Embedding the Library

The CLI in `src/main.cpp` is a thin driver over `RFDETRInference`
(`src/rfdetr_inference.hpp`). To use the library directly, fill in a `Config` and construct it:

```cpp
#include "rfdetr_inference.hpp"

Config config;
config.resolution = 0;              // 0 = auto-detect from the model
config.threshold = 0.6f;            // Higher confidence threshold
config.max_detections = 100;        // Fewer ranked candidates
config.mask_threshold = 0.5f;       // More conservative masks
config.model_type = ModelType::SEGMENTATION;

RFDETRInference inference(model_path, label_path, config);
```

The class exposes the pipeline stage by stage — `preprocess_image`, `run_inference`,
`postprocess_outputs` / `postprocess_segmentation_outputs` / `postprocess_keypoint_outputs`,
the matching `draw_*` helpers, and `save_output_image` — so you can stop after postprocessing
and consume the decoded results yourself rather than an annotated image. `get_resolution()`
reports the resolution actually in use after auto-detection, and `gpu_preprocess_active()` /
`gpu_postprocess_active()` report whether a GPU path is compiled in, enabled, **and** backed by
a real device. Decoded types (`BoundingBox`, `rfdetr::media::Mask`, `KeypointResult`) are
described in [architecture.md](architecture.md#c-result-types).

A second constructor takes an already-built `std::unique_ptr<InferenceBackend>`, which is how
the tests inject a fake backend.

### Config reference

`Config` (`src/rfdetr_inference.hpp`) holds every inference setting. Some are reachable from the
command line; the rest are set by editing `src/main.cpp`, or on your own `Config` when
embedding. `src/main.cpp` leaves every field it does not override at its `Config` default, so
changing a default in the header is enough for the fields with no flag.

| `Config` field | Default | CLI override |
|----------------|---------|--------------|
| `model_type` | `ModelType::DETECTION` | `--segmentation` / `--keypoint` |
| `threshold` | `0.5` | `--threshold <val>` |
| `resolution` | `560` — **`0` means auto-detect**, which is what the CLI passes when `--resolution` is omitted | `--resolution <px>` |
| `max_detections` | `300` (top-k selection) | `--max-detections <n>` |
| `mask_threshold` | `0.0` (binary mask generation) | `--mask-threshold <val>` |
| `background_class_id` | `0` (background-first exports) | `--background-class-id <n\|none>` |
| `keypoint_counts` | `{0, 17}` | `--keypoint-counts <n[,n...]>` |
| `gpu_preprocess` / `gpu_postprocess` | `false` | `--gpu-preprocess` / `--gpu-postprocess` |
| `dali_pipeline_dir` | `data/dali` | `--dali-pipeline-dir <dir>` |

The remaining fields have no flag:

| Field | Default | What it controls |
|-------|---------|------------------|
| `means` / `stds` | ImageNet `[0.485, 0.456, 0.406]` / `[0.229, 0.224, 0.225]` | Normalization statistics; must match what the model was exported with |
| `gpu_device_id` | `0` | Which CUDA device the GPU pipeline uses |
| `keypoint_names` | COCO 17-keypoint names | Per-keypoint labels |
| `skeleton` | COCO 19 edges | Which keypoint pairs are joined when drawing |
| `keypoint_uncertainty_alpha` | `0.2f` | Uncertainty-weighted score fusion; `0` disables it |
| `draw_uncertainty` | `false` | Draw uncertainty ellipses on keypoints |
| `keypoint_color` | green `{0, 255, 0}` | Default keypoint color |

The video driver's own knobs live in `VideoPipelineConfig` (`src/video_pipeline.hpp`) —
`ring_buffer_size` (8) and `output_path` — and `src/main.cpp` sets them.

---

## CI Coverage and Its Limits

| Workflow | File | What it does |
|----------|------|-------------|
| **C++ Lint & Build** | `lint.yml` | Version sync, Dockerfile shared blocks, format check, clang-tidy, cppcheck, build with `-DWERROR=ON` |
| **Build & Test** | `ci.yml` | Build with benchmarks, run unit tests, run benchmarks, run unit tests under ASan+UBSan |
| **GPU Backend Compile** | `gpu-compile.yml` | Compiles the TensorRT backend and both GPU halves with `-DWERROR=ON`, across all four `USE_DALI`/`USE_CUDA_POSTPROCESS` combinations, plus the full pipeline against TensorRT 11 headers |
| **Dependency Modes** | `deps-modes.yml` | `workflow_dispatch` only; matrix over apt / conan / vcpkg |

All three push/PR workflows trigger on `master` and `develop`.

**What CI cannot do**, and what that means for you:

- `gpu-compile.yml` runs on GPU-less runners, so it compiles but never links or executes. It
  stages headers-only TensorRT and DALI prefixes with `scripts/ci/stage_gpu_headers.sh` — the
  full TensorRT tarball is 6.2 GB and the DALI wheel 380 MB, against ~130 KB of TensorRT header
  debs and a few MB of DALI headers — and builds only the `rfdetr_inference_lib` static target.
  A compile break is therefore a red PR rather than a surprise on metered hardware, but
  **runtime** behaviour of the GPU path is gated on manual verification.
- The GPU unit tests in `test_gpu_postprocess.cpp` `GTEST_SKIP()` without a CUDA device.
- The **ExecuTorch** backend is neither compiled nor run by CI.
- Integration tests are not run by CI — they need a real model in the compiled-in backend's
  format.

So the TensorRT, GPU-pipeline, and ExecuTorch paths must be exercised by hand before they can
be trusted. `./scripts/run_gate.sh` drives the executable part of that verification unattended
and reports the rest as `UNRUN`. The checklist it implements and the procedure for running it on
rented hardware are maintainer material, kept with the specs —
[specs/rented-gpu-runbook.md](../specs/rented-gpu-runbook.md).

---

## Where to Go Next

- [building.md](building.md) — every build configuration, step by step
- [usage.md](usage.md) — the authoritative command-line and flag reference (the `Config` reference
  is [above](#config-reference), on this page)
- [export.md](export.md) — producing `.onnx`, `.engine`, and `.pte` models
- [architecture.md](architecture.md) — GPU pipeline, ring buffer, tensor contracts
- [development.md](development.md) — sanitizers, Valgrind, tests, benchmarks
- [docker.md](docker.md) — the backend × media-backend image matrix
- [glossary.md](glossary.md) — terms used across the codebase
