# RF-DETR C++ Inference

[![C++](https://img.shields.io/badge/language-C++20-blue.svg)](https://en.cppreference.com/w/cpp)
[![CMake](https://img.shields.io/badge/build%20system-CMake-blue.svg)](https://cmake.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.3.0-blue.svg)](https://github.com/olibartfast/rf-detr-cpp-inference/releases/tag/v0.3.0)

C++ project for performing object detection, instance segmentation, and keypoint inference using the RF-DETR model with **multiple inference backends** (ONNX Runtime and TensorRT) and a swappable **media/display backend** (FFmpeg + SDL2 + stb by default, or OpenCV). Supports both single-image and **multi-threaded video processing** via a zero-copy ring buffer pipeline, plus an opt-in **GPU pipeline** (DALI preprocessing + CUDA segmentation postprocessing) on the TensorRT backend.

---

## Table of Contents
- [Dependencies](#dependencies)
- [Model Setup](#model-setup)
- [Installation](#installation)
- [Building](#building)
- [Usage](#usage)
- [GPU Pipeline](#gpu-pipeline)
- [Configuration](#configuration)
- [Video Processing](#video-processing)
- [Technical Details](#technical-details)
- [Testing](#testing)
- [Docker](#docker)
- [Code Quality Tools](#code-quality-tools)
- [Acknowledgements](#acknowledgements)

---

## Dependencies

### Required (All Backends)
- **C++20 Compiler**: Clang 15+ or GCC 12+ (e.g., `clang++-15` or `g++-12`)
- **CMake**: Version 3.12 or higher
- **Google Test**: 1.12.1 (auto-fetched; see [Dependency Resolution](#dependency-resolution))
- **Ninja**: Optional but recommended (`sudo apt-get install ninja-build`)

> Annotation text is drawn with an 8x8 bitmap font
> ([third_party/font8x8](third_party/font8x8)), which is used by **both**
> media backends below.

### Media/Display Backend (choose one via CMake)

The image load/save, video decode/encode, and `--display` preview window are
backed by either **FFmpeg + SDL2 + stb** (the default) or **OpenCV**. Exactly
one is compiled in via `-DUSE_OPENCV=ON/OFF`.

#### FFmpeg + SDL2 + stb (default, `-DUSE_OPENCV=OFF`)
- **FFmpeg** (libavcodec/libavformat/libavutil/libswscale): 5.x+ for video decode/encode
- **SDL2**: 2.x for the `--display` live preview window
- **stb** ([third_party/stb](third_party/stb)): vendored single-header image I/O — no install needed
- No OpenCV dependency

#### OpenCV (`-DUSE_OPENCV=ON`)
- **OpenCV**: 4.x (`core`, `imgcodecs`, `imgproc`, `videoio`, `highgui`) for image I/O, video decode/encode, and the `--display` preview
- Replaces FFmpeg, SDL2, **and** stb entirely — none of those are required when OpenCV is enabled

### Python / Pip Packages (Export Tooling)
- **RF-DETR export package**: `rfdetr[onnx]==1.8.3` from `deploy/requirements.txt`
- **Python**: 3.10+ (Python 3.11 virtual environment recommended)
- **pre-commit**: Optional for local hooks; install with `pip install pre-commit`

### Backend-Specific Dependencies

#### ONNX Runtime Backend (Default)
- **ONNX Runtime**: Version 1.21.0 (automatically downloaded during build)
- **Platform**: Linux, Windows, macOS
- **Acceleration**: CPU and GPU (CUDA/DirectML)

#### TensorRT Backend (Optional)
- **TensorRT**: Version 10.13.3.9 (automatically downloaded during build if not found)
- **CUDA Toolkit**: Version 13.x for the bundled TensorRT 10.13.3.9 archive - **must be installed manually**
- **Platform**: Linux with NVIDIA GPU
- **Acceleration**: NVIDIA GPU only
- **Note**: TensorRT libraries are automatically configured with RPATH, no LD_LIBRARY_PATH needed

#### GPU Pipeline (Optional, TensorRT only)
- **CUDA Toolkit** (with `nvcc` for `-DUSE_CUDA_POSTPROCESS=ON`): resolved via CMake's `FindCUDAToolkit`; CUB (header-only, bundled with the toolkit) is used by the postprocessing kernels
- **NVIDIA DALI** (for `-DUSE_DALI=ON`): C API libraries + headers staged from a pinned Triton container (`nvcr.io/nvidia/tritonserver:25.12-py3`) via `./scripts/fetch_dali.sh` — NVIDIA ships no standalone C++ DALI distribution. Point the build at the staged directory with `-DDALI_ROOT=<dir>`.
- See [GPU Pipeline](#gpu-pipeline) for build and usage details

---

## Model Setup

This project supports both RF-DETR detection and segmentation models from Roboflow.

1. **Visit the RF-DETR Repository**:
   - Go to the [RF-DETR GitHub repository](https://github.com/roboflow/rf-detr) for model details.
   - Read the [Roboflow blog](https://blog.roboflow.com/rf-detr/) for an overview.

2. **Download the ONNX Model**:
   - Follow instructions in the [export documentation](docs/export.md) to export models in ONNX format.
   - **Tested with**: `rfdetr[onnx]==1.8.3` (Python 3.10+; 3.11 venv recommended)
   - **Detection models**: Export with standard configuration (outputs: `dets`, `labels`)
   - **Segmentation models**: Export with segmentation configuration (outputs: `dets`, `labels`, `masks`)
   - **Keypoint models**: Export with keypoint configuration (outputs: `dets`, `labels`, `keypoints`)
   - Place the model (e.g., `rfdetr-medium.onnx` or `rfdetr-seg-medium.onnx`) in a chosen directory.

3. **Prepare the COCO Labels**:
   - Create a `coco-labels-91.txt` file with one label per line:
     ```
     person
     bicycle
     car
     motorbike
     aeroplane
     ...
     ```

---

## Installation

### Install Dependencies (Ubuntu)

```bash
sudo apt-get update
sudo apt-get install -y cmake

# Compiler - either clang or gcc (any C++20-capable version):
sudo apt-get install -y clang-15
# or use the system default gcc (no install needed if already present)

# FFmpeg development libraries (video decode/encode) + pkg-config:
sudo apt-get install -y pkg-config libavcodec-dev libavformat-dev libavutil-dev libswscale-dev

# SDL2 (required for the default FFmpeg + SDL2 + stb media/display backend):
sudo apt-get install -y libsdl2-dev

# OpenCV (alternative media/display backend — install ONLY if you will build with -DUSE_OPENCV=ON,
#          in which case you do NOT need the FFmpeg/SDL2 packages above):
sudo apt-get install -y libopencv-dev

# Optional (faster incremental builds):
sudo apt-get install -y ninja-build

# Optional (linting and formatting — use version 18+ for GCC 14 compatibility):
sudo apt-get install -y clang-format-18 clang-tidy-18
```

---

## Building 

### Backend Selection

This project uses **compile-time backend selection**. Choose your backend when building:

| Backend | Best For | Pros | Cons |
|---------|----------|------|------|
| **ONNX Runtime** | Development, CPU inference | Cross-platform, easy setup | Slower than TensorRT on GPU |
| **TensorRT** | Production on NVIDIA GPUs | Maximum performance | GPU-only, requires CUDA/TensorRT |

**Important**: Only ONE backend can be enabled at a time. The backend is compiled into the binary for optimal performance and smaller binary size.

### Format Code (Optional)

If you have `clang-format-18` installed, you can check and auto-format all source files:

```bash
# Check for formatting issues (no changes made):
find src tests -name '*.cpp' -o -name '*.hpp' | xargs clang-format-18 --dry-run --Werror

# Auto-format in place:
find src tests -name '*.cpp' -o -name '*.hpp' | xargs clang-format-18 -i
```

### Static Analysis (Optional)

If you have `clang-tidy-18` installed, you can run static analysis using the compile commands database:

```bash
# Generate compile_commands.json first:
cmake -S . -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

# Run clang-tidy on project sources:
find src -name '*.cpp' | xargs clang-tidy-18 -p build
```

### Cppcheck (Optional)

If you have `cppcheck` installed, you can run additional static analysis:

```bash
# Install cppcheck:
sudo apt-get install -y cppcheck

# Run manually:
cppcheck --enable=all --std=c++20 \
  --suppress=missingIncludeSystem \
  --suppress=unmatchedSuppression \
  --error-exitcode=1 \
  -I src src/
```

This is also run automatically on every commit via pre-commit (see [Pre-commit](#pre-commit-optional)).

### Sanitizers (Optional)

Build with **AddressSanitizer + UndefinedBehaviorSanitizer** via `-DSANITIZERS=ON`, with stricter **UndefinedBehaviorSanitizer** checks via `-DSTRICT_UBSAN=ON`, or with **ThreadSanitizer** (data-race detection) via `-DTHREAD_SANITIZER=ON`. These modes are mutually exclusive — enable only one per build directory. Use separate build directories to keep them independent:

These options use the sanitizer instrumentation and runtime libraries provided
by the active C++ compiler toolchain (Clang or GCC). The project does not fetch,
vendor, or depend on the archived `google/sanitizers` repository.

```bash
# ASan + UBSan
cmake -S . -B build-san -DCMAKE_BUILD_TYPE=Debug -DSANITIZERS=ON
cmake --build build-san --parallel
./build-san/unit_tests

# Strict UBSan (extra bounds and vptr checks; Clang also enables implicit-conversion)
cmake -S . -B build-strict-ubsan -DCMAKE_BUILD_TYPE=Debug -DSTRICT_UBSAN=ON
cmake --build build-strict-ubsan --parallel
./build-strict-ubsan/unit_tests

# ThreadSanitizer (data races)
cmake -S . -B build-tsan -DCMAKE_BUILD_TYPE=Debug -DTHREAD_SANITIZER=ON
cmake --build build-tsan --parallel
TSAN_OPTIONS="halt_on_error=1" ./build-tsan/unit_tests
```

Sanitizers catch memory errors, use-after-free, undefined behaviour, integer overflow (ASan+UBSan), stricter bounds/vptr issues, plus implicit-conversion issues on Clang (strict UBSan), and data races (TSan) at runtime. ASan+UBSan and TSan run in CI; strict UBSan is opt-in for local diagnosis because it can be noisier.

### Valgrind / Profiling (Optional)

A plain **Debug build without sanitizers** is required (ASan/TSan conflict with Valgrind). When Valgrind is detected, CMake generates `memcheck`, `callgrind`, and `massif` targets:

```bash
cmake -S . -B build-valg -DCMAKE_BUILD_TYPE=Debug

cmake --build build-valg --target memcheck      # memory errors + leaks (run by CI)
cmake --build build-valg --target callgrind     # CPU/cache profile -> callgrind.out.<pid>
cmake --build build-valg --target massif        # heap profile       -> massif.out.<pid>

# Read the profiles:
callgrind_annotate build-valg/callgrind.out.<pid>
ms_print build-valg/massif.out.<pid>
```

The profilers target `benchmarks` (if built with `-DBENCHMARKS=ON`, recommended for self-contained workloads) or `inference_app`; pass extra args with `-DVALGRIND_PROFILE_ARGS="model.onnx image.jpg labels.txt"`. A lower-overhead alternative is `perf record ./build/benchmarks && perf report`. An optional `valgrind.supp` at the repo root is picked up automatically if present.

### Pre-commit (Optional)

[pre-commit](https://pre-commit.com/) runs `clang-format` and `cppcheck` automatically on every commit:

```bash
pip install pre-commit
pre-commit install           # install the git hook
pre-commit run --all-files   # run manually on all files
```

### Strict Compilation (Optional)

To treat all compiler warnings as errors (as CI does), pass `-DWERROR=ON`:

```bash
cmake -S . -B build -DWERROR=ON
cmake --build build
```

### Dependency Resolution

All dependencies flow through a unified facade (`find_dependency_unified`) that
picks the acquisition strategy per `-DDEPS_MODE`:

| mode | chain | use when |
|---|---|---|
| `apt` (default) | apt → provided | no extra tooling; system packages + pinned downloads |
| `conan` | conan → apt → provided | ConanCenter binaries or local cache |
| `vcpkg` | vcpkg → apt → provided | vcpkg manifest mode |
| `auto` | apt → conan → vcpkg → provided | mixed: each dep uses fastest available |

`apt` is chained as a fallback in conan/vcpkg modes so system packages (Threads)
resolve correctly. FFmpeg, SDL2, OpenCV, and GTest have conan/vcpkg coordinates
in `conanfile.txt` / `vcpkg.json`; ONNX Runtime and TensorRT stay provided-download.
The GPU-pipeline dependencies also route through the facade: CUDA Toolkit
resolves via CMake's `FindCUDAToolkit` (apt handler), and DALI is ROOT-only —
staged locally by `./scripts/fetch_dali.sh` and pointed to with `-DDALI_ROOT`
(no download fallback exists, since NVIDIA ships no standalone C++ DALI
distribution).

```bash
# Conan (CMakeDeps-only mode — keeps system compiler):
#   sudo apt install libva-dev libegl-dev libgl-dev  # ffmpeg/sdl system deps
#   conan install . -of=build/conan-deps --build=missing
#   cmake -S . -B build -DDEPS_MODE=conan -DDEPS_CONAN_DIR=build/conan-deps
# vcpkg (manifest mode):
#   cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE=<vcpkg>/scripts/buildsystems/vcpkg.cmake \
#     -DDEPS_MODE=vcpkg
```

Architecture details: [docs/package-manager-architecture.md](docs/package-manager-architecture.md)

### Build with ONNX Runtime (Default)

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/usr/bin/clang-15 \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++-15

cmake --build build --parallel
```

Using the system default compiler (typically gcc):

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build --parallel
```

If you don't have Ninja installed, drop `-G Ninja` to use Make instead:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build --parallel
```

### Build with TensorRT Backend

```bash
cmake -S . -B build -G Ninja \
  -DUSE_ONNX_RUNTIME=OFF \
  -DUSE_TENSORRT=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/usr/bin/clang-15 \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++-15

cmake --build build --parallel
```

**What happens**:
- TensorRT 10.13.3.9 is automatically downloaded if not found
- Libraries are configured with RPATH - no need to set `LD_LIBRARY_PATH`
- The executable will use TensorRT for inference
- Requires CUDA 13.x installed manually for the bundled TensorRT 10.13.3.9 build
- Pre-built `.engine` or `.trt` files are loaded directly, skipping ONNX-to-TensorRT conversion

### Build with the GPU Pipeline (TensorRT + DALI + CUDA)

The GPU pipeline requires the TensorRT backend — DALI writes into, and the CUDA
kernels read from, the inference engine's device buffers, and only the TensorRT
backend exposes device pointers and a CUDA stream. Configuring it with the ONNX
Runtime backend fails with an explicit error.

```bash
# 1. Stage the DALI C++ libraries (one-time; extracts from a pinned Triton container):
./scripts/fetch_dali.sh                      # -> ~/dependencies/dali

# 2. Configure with both GPU halves enabled:
cmake -S . -B build -G Ninja \
  -DUSE_ONNX_RUNTIME=OFF \
  -DUSE_TENSORRT=ON \
  -DUSE_GPU_PIPELINE=ON \
  -DDALI_ROOT=$HOME/dependencies/dali \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build --parallel
```

The two halves are independent: `-DUSE_CUDA_POSTPROCESS=ON` alone builds the
CUDA segmentation postprocessing (needs `nvcc`, no DALI), and `-DUSE_DALI=ON`
alone builds the DALI preprocessing (plain C++ against the DALI C API, no
`nvcc`). `-DUSE_GPU_PIPELINE=ON` turns on both. See [GPU Pipeline](#gpu-pipeline)
for the runtime flags and the DALI pipeline files.

### Build with OpenCV Media/Display Backend

By default the project uses FFmpeg + SDL2 + stb for image/video I/O and the
preview window. Pass `-DUSE_OPENCV=ON` to compile OpenCV in instead (image I/O
via `imgcodecs`, video decode/encode via `videoio`, `--display` via `highgui`):

```bash
cmake -S . -B build -G Ninja \
  -DUSE_OPENCV=ON \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build --parallel
```

**What happens**:
- OpenCV (`core`, `imgcodecs`, `imgproc`, `videoio`, `highgui`) is found via CMake's `find_package`
- FFmpeg, SDL2, and stb are **not** required and not linked
- `VideoReader`, `VideoWriter`, `Display`, and image load/save swap to their OpenCV implementations
- Orthogonal to the inference backend — combine freely, e.g. `-DUSE_TENSORRT=ON -DUSE_OPENCV=ON`

### Build Options

- `-DUSE_ONNX_RUNTIME=ON/OFF` - Enable ONNX Runtime backend (default: ON)
- `-DUSE_TENSORRT=ON/OFF` - Enable TensorRT backend (default: OFF)
- `-DUSE_OPENCV=ON/OFF` - Use OpenCV for image/video/display I/O instead of FFmpeg+SDL2+stb (default: OFF)
- `-DUSE_DALI=ON/OFF` - DALI GPU preprocessing; requires TensorRT backend and `-DDALI_ROOT` (default: OFF)
- `-DUSE_CUDA_POSTPROCESS=ON/OFF` - CUDA segmentation postprocessing; requires TensorRT backend and `nvcc` (default: OFF)
- `-DUSE_GPU_PIPELINE=ON/OFF` - Convenience switch that enables both `USE_DALI` and `USE_CUDA_POSTPROCESS` (default: OFF)
- `-DDALI_ROOT=<path>` - Directory with the staged DALI libraries/headers (from `./scripts/fetch_dali.sh`)
- `-DCMAKE_CUDA_ARCHITECTURES=<list>` - CUDA architectures for the postprocessing kernels (default: `86`, RTX 30-series)
- `-DCMAKE_BUILD_TYPE=Release/Debug` - Build configuration
- `-DSANITIZERS=ON/OFF` - Enable AddressSanitizer + UndefinedBehaviorSanitizer (default: OFF)
- `-DSTRICT_UBSAN=ON/OFF` - Enable stricter UndefinedBehaviorSanitizer checks: Clang: `undefined,local-bounds,vptr,implicit-conversion`; GCC: `undefined,bounds-strict,vptr` (default: OFF; mutually exclusive with other sanitizer modes)
- `-DTHREAD_SANITIZER=ON/OFF` - Enable ThreadSanitizer/data-race detection (default: OFF; mutually exclusive with other sanitizer modes)
- `-DWERROR=ON/OFF` - Treat compiler warnings as errors (default: OFF)
- `-DBENCHMARKS=ON/OFF` - Build Google Benchmark targets (default: OFF)
- `-DDEPS_MODE=apt/conan/vcpkg/auto` - Package manager ecosystem for dependency resolution (default: apt)
- `-DDEPS_OFFLINE=ON/OFF` - Disable network lookups; ROOT provided lookups only (default: OFF)
- `-DDEPS_DEBUG=ON/OFF` - Log dependency resolution decisions (default: OFF)
- `-DDEPS_PROVIDED_DIR=<path>` - Where provided-download archives extract (default: `<build>/_deps`)
- `-DDEPS_CONAN_DIR=<path>` - Conan CMakeDeps output dir (CMakeDeps-only mode — consumes prebuilt binaries without the conan toolchain overriding the system compiler)

---

## Usage

### Prepare Input Files

- The RF-DETR model file (`.onnx` for ONNX Runtime, `.onnx`/`.engine`/`.trt` for TensorRT)
- An input image (e.g., `image.jpg`) or video file (e.g., `video.mp4`)
- A COCO labels file (e.g., `coco-labels-91.txt`)

### Run Inference

After building the project, run the inference application:

#### Object Detection

```bash
./build/inference_app /path/to/model.onnx /path/to/image.jpg /path/to/coco-labels-91.txt
```

#### Instance Segmentation

```bash
./build/inference_app /path/to/model.onnx /path/to/image.jpg /path/to/coco-labels-91.txt --segmentation
```

#### Keypoint Detection

```bash
./build/inference_app /path/to/model.onnx /path/to/image.jpg /path/to/coco-labels-91.txt --keypoint
```

#### Video Processing

```bash
./build/inference_app /path/to/model.onnx /path/to/video.mp4 /path/to/coco-labels-91.txt
```

With live preview window:

```bash
./build/inference_app /path/to/model.onnx /path/to/video.mp4 /path/to/coco-labels-91.txt --display
```

Video with segmentation:

```bash
./build/inference_app /path/to/model.onnx /path/to/video.mp4 /path/to/coco-labels-91.txt --segmentation
```

Supported video formats: `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`, `.flv`, `.wmv`. Output is written to `output_video.mp4`.

#### Custom Confidence Threshold

Override the default confidence threshold (0.5) without recompiling using the `--threshold` flag:

```bash
./build/inference_app /path/to/model.onnx /path/to/image.jpg /path/to/coco-labels-91.txt --threshold 0.7
```

`--threshold` works with all modes (`--segmentation`, `--keypoint`, video input, and `--display`).

#### Using Pre-built TensorRT Engine

If you have a pre-built TensorRT engine file (`.engine` or `.trt`), use it directly:

```bash
./build/inference_app /path/to/model.engine /path/to/image.jpg /path/to/coco-labels-91.txt --segmentation
```

#### GPU Pipeline Flags (TensorRT builds with the GPU pipeline compiled in)

```bash
# DALI GPU preprocessing + CUDA GPU segmentation postprocessing:
./build/inference_app /path/to/model.engine /path/to/image.jpg /path/to/coco-labels-91.txt \
  --segmentation --gpu-preprocess --gpu-postprocess
```

- `--gpu-preprocess` — decode/resize/normalize on the GPU with DALI (build with `-DUSE_DALI=ON`)
- `--gpu-postprocess` — segmentation mask decode/resize/threshold with CUDA kernels (build with `-DUSE_CUDA_POSTPROCESS=ON`); segmentation only, requires `--segmentation`
- `--dali-pipeline-dir <dir>` — where the serialized `.dali` pipeline files live (default: `data/dali`)

Both flags default off — the CPU paths remain the default even in a GPU-pipeline build. See [GPU Pipeline](#gpu-pipeline).

**Features:**
- The output image is saved as `output_image.jpg`; video output is saved as `output_video.mp4`
- Detection/segmentation results (bounding boxes, labels, scores, and mask pixels) are printed to the console
- Input resolution is automatically detected from the model (supports 432x432, 560x560, etc.)
- Segmentation mode draws colored masks with transparency overlays
- Uses top-k selection (default: 300 detections) for efficient processing
- Video files are automatically detected by extension and processed with the multi-threaded pipeline

---

## GPU Pipeline

An opt-in pipeline that moves preprocessing and segmentation postprocessing onto
the GPU, on the same CUDA stream as the TensorRT execution context — no Triton
server involved. Design notes and phased plan: [docs/GPU_PIPELINE_ROADMAP.md](docs/GPU_PIPELINE_ROADMAP.md).

```
image bytes ──> DALI "encoded" pipeline ──┐
                (nvJPEG decode, resize,   │
                 normalize — on GPU)      ├──> device float[1,3,res,res]
video frame ──> DALI "frame" pipeline ────┘         │
                (BGR→RGB, resize, normalize)        v
                                          TensorRT enqueue (same stream)
                                                    │
                          ┌─────────────────────────┴───────────────┐
                          v                                         v
                detection / keypoint                          segmentation
                D2H outputs, existing                   CUDA postprocess on stream:
                CPU postprocess                         sigmoid → top-k → box decode
                                                        → mask resize+threshold,
                                                        one packed D2H at the end
```

### What each half does

- **DALI preprocessing** (`-DUSE_DALI=ON`, `--gpu-preprocess`): image decode
  (nvJPEG), resize, and ImageNet normalization run on the GPU and write straight
  into the TensorRT input binding. For still images only the compressed bytes
  cross to the GPU; for video the preprocess pipeline stage becomes a passthrough
  and DALI runs on the backend's stream inside the inference stage. Works with
  detection, segmentation, and keypoint models.
- **CUDA segmentation postprocessing** (`-DUSE_CUDA_POSTPROCESS=ON`,
  `--gpu-postprocess`): score sigmoid, global top-k, box decode, and the
  per-instance mask resize + threshold (the CPU path's dominant cost) run as CUDA
  kernels reading the TensorRT output bindings in place, with a single packed
  device-to-host transfer of the final results. Segmentation only.

Both halves are compile-time optional and runtime opt-in: a GPU-pipeline build
still defaults to the CPU paths, and the ONNX Runtime build is unaffected. The
backend interface gained optional device-side I/O entry points
(`supports_device_io()`, `run_inference_device()`, device pointers, stream) that
only the TensorRT backend implements.

### DALI pipeline files

`--gpu-preprocess` loads serialized DALI pipelines named
`preprocess_encoded_<resolution>.dali` (still images) and
`preprocess_frame_<resolution>.dali` (video frames) from `--dali-pipeline-dir`
(default `data/dali`), where `<resolution>` is the model input resolution
auto-detected from the engine. Pre-generated pipelines for resolutions **432**
and **576** are checked in under [data/dali](data/dali).

For other resolutions, regenerate them (runs inside the pinned Triton container,
needs `--gpus all`; no local DALI pip install required):

```bash
./scripts/generate_dali_pipelines.sh 560           # -> data/dali/
```

The generator itself is [deploy/dali/generate_preprocess_pipeline.py](deploy/dali/generate_preprocess_pipeline.py)
(uses the `nvidia-dali` Python package inside the container). The C++ side
validates the produced tensor size against the TensorRT input binding, so a
stale pipeline file fails loudly rather than silently degrading results.

### Version pinning

DALI libraries and pipeline serialization both come from the same pinned
container, `nvcr.io/nvidia/tritonserver:25.12-py3` (override with
`TRITON_IMAGE=...` on both scripts), keeping the DALI/CUDA/TensorRT triple
consistent with the TensorRT 10.13.3.9 / CUDA 13.x pin above.

---

## Configuration

The inference engine supports various configuration options that can be modified in `src/main.cpp`:

- **Model Type**: `ModelType::DETECTION`, `ModelType::SEGMENTATION`, or `ModelType::KEYPOINT` (selected via the `--segmentation` / `--keypoint` CLI flags)
- **Resolution**: Set to `0` for auto-detection from model, or specify manually (e.g., `432`, `560`)
- **Confidence Threshold**: Default `0.5` (adjustable in `Config::threshold`)
- **Max Detections**: Default `300` for top-k selection (adjustable in `Config::max_detections`)
- **Mask Threshold**: Default `0.0` for binary mask generation (adjustable in `Config::mask_threshold`)
- **Normalization**: ImageNet mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`
- **GPU Pipeline**: `Config::gpu_preprocess` / `Config::gpu_postprocess` (set via the `--gpu-preprocess` / `--gpu-postprocess` CLI flags, default `false`), `Config::dali_pipeline_dir` (default `data/dali`, via `--dali-pipeline-dir`), and `Config::gpu_device_id` (default `0`)

### Example Custom Configuration

```cpp
Config config;
config.resolution = 0;              // Auto-detect
config.threshold = 0.6f;            // Higher confidence threshold
config.max_detections = 100;        // Fewer detections
config.mask_threshold = 0.5f;       // More conservative masks
config.model_type = ModelType::SEGMENTATION;
```

---

## Video Processing

Video files are processed using a **four-stage ring buffer pipeline** that maximizes throughput with zero frame copies between stages:

```
                   free_slots (recycled)
                 +-------------------------+
                 |                         |
                 v                         |
 +--------+ idx  +-----------+ idx  +------++ idx  +------+
 | Decode | ---> | Preprocess| ---> | Infer| ----> | Draw |
 +--------+      +-----------+      +------+       +------+
  media           resize+norm        run model      annotate +
  decode into     into slot.tensor   postprocess    media encode
  slot.raw_frame  (pre-allocated)    into slot.*    + optional
                                                    preview
```

The default media/display backend uses FFmpeg for video decode/encode, SDL2 for
preview, and stb for image I/O. `-DUSE_OPENCV=ON` swaps those pieces for OpenCV
`videoio`, `highgui`, and `imgcodecs`.

- **4 `std::jthread`s** run concurrently, one per stage
- **Pre-allocated `FrameSlot`s** are reused via a ring buffer (default size: 8)
- Stages pass slot indices (not frames) through **bounded queues** with backpressure
- The inference stage owns its own `RFDETRInference` instance — no locks on the hot path
- Graceful shutdown via poison pill (`SIZE_MAX`) propagated through all queues
- Frame ordering is preserved (all stages are single-threaded FIFO)
- With `--gpu-preprocess`, the preprocess stage becomes a passthrough: DALI's `frame` pipeline runs on the backend's CUDA stream inside the inference stage, and the CPU cost of the bilinear resample disappears from the pipeline entirely

Use `--display` to open a live preview window (press ESC to quit early).

---

## Technical Details

### Model Outputs

#### Detection Model
- **dets**: `float32[batch, num_queries, 4]` - Bounding boxes in `cxcywh` format (normalized)
- **labels**: `float32[batch, num_queries, num_classes]` - Class logits

#### Segmentation Model
- **dets**: `float32[batch, num_queries, 4]` - Bounding boxes in `cxcywh` format (normalized)
- **labels**: `float32[batch, num_queries, num_classes]` - Class logits
- **masks**: `float32[batch, num_queries, mask_h, mask_w]` - Segmentation masks (e.g., 108x108)

#### Keypoint Model
- **dets**: `float32[batch, num_queries, 4]` - Bounding boxes in `cxcywh` format (normalized)
- **labels**: `float32[batch, num_queries, num_classes+1]` - Class logits (index 0 = background)
- **keypoints**: `float32[batch, num_queries, C*K_max, 8]` - Keypoints (8 channels per keypoint)

### C++ Result Types

Postprocessing APIs expose decoded boxes as `std::vector<BoundingBox>`, with `x_min`, `y_min`, `x_max`, and `y_max` fields in pixel-space `xyxy` format. Segmentation masks use `std::vector<rfdetr::media::Mask>`, and keypoints use `std::vector<std::vector<KeypointResult>>` for per-detection keypoint metadata.

### Processing Pipeline

1. **Preprocessing**:
   - Resize image to model input resolution (auto-detected)
   - Convert BGR to RGB
   - Normalize with ImageNet statistics
   - Convert to CHW format

2. **Inference**:
   - Run ONNX Runtime session
   - Auto-detect output tensor names from model

3. **Postprocessing**:
   - **Detection**: Select predictions above confidence threshold
   - **Segmentation**: 
     - Apply sigmoid to class logits
     - Top-k selection across all classes and queries
     - Resize masks to original image dimensions using bilinear interpolation
     - Apply threshold to create binary masks
   - Convert bounding boxes from `cxcywh` to `xyxy` format
   - Scale coordinates to original image size

4. **Visualization**:
   - Draw bounding boxes with class labels
   - Overlay segmentation masks with transparency (alpha = 0.5)
   - Use deterministic colors based on class IDs

---

## Testing

### Unit Tests

Unit tests use Google Test and run without any model files:

```bash
cmake -S . -B build
cmake --build build --parallel
ctest --test-dir build --output-on-failure -R UnitTests
```

Integration tests need a real ONNX model. They auto-detect `~/Downloads/rfdetr-medium.onnx` (or legacy `inference_model.onnx`), or set `RFDETR_TEST_MODEL`:

```bash
export RFDETR_TEST_MODEL=/path/to/rfdetr-medium.onnx
ctest --test-dir build --output-on-failure -R IntegrationTests
```

Without a model, integration tests that need inference are skipped.

In a `-DUSE_CUDA_POSTPROCESS=ON` build, the unit tests additionally include a
CPU-versus-GPU parity gate for the segmentation postprocessor
(`tests/unit/test_gpu_postprocess.cpp`). It needs no model and no DALI —
synthetic tensors are served from both host and device memory through a mock
backend — and every case skips (rather than fails) when no CUDA device is
present, so CI can compile the GPU targets on runners without a GPU.

### Benchmarks

Benchmarks use [Google Benchmark](https://github.com/google/benchmark) to measure preprocessing performance. Enable with `-DBENCHMARKS=ON`:

```bash
cmake -S . -B build -DBENCHMARKS=ON
cmake --build build --target benchmarks --parallel
./build/benchmarks
```

### CI

Two GitHub Actions workflows run on every push/PR to `master` and `develop`:

| Workflow | File | What it does |
|----------|------|-------------|
| **C++ Lint & Build** | `lint.yml` | Format check, clang-tidy, cppcheck, build with `-DWERROR=ON` |
| **Build & Test** | `ci.yml` | Build with benchmarks, run unit tests, run benchmarks, run unit tests under ASan+UBSan |

---

## Docker

A single parametric `Dockerfile` builds the full **inference-backend × media-backend** matrix via two build args:

| `INFERENCE_BACKEND` | `MEDIA_BACKEND` | Image |
|---------------------|-----------------|-------|
| `onnx` (default)    | `ffmpeg` (default) | ONNX Runtime + FFmpeg/SDL2/stb |
| `onnx`              | `opencv`        | ONNX Runtime + OpenCV |
| `tensorrt`          | `ffmpeg`        | TensorRT + FFmpeg/SDL2/stb |
| `tensorrt`          | `opencv`        | TensorRT + OpenCV |

Build all four variants:

```bash
# ONNX Runtime (CPU) — FFmpeg/SDL2/stb media backend (default)
docker build -t rfdetr-onnx-ffmpeg .
# ONNX Runtime (CPU) — OpenCV media backend
docker build -t rfdetr-onnx-opencv --build-arg MEDIA_BACKEND=opencv .
# TensorRT (GPU) — FFmpeg/SDL2/stb media backend
docker build -t rfdetr-trt-ffmpeg --build-arg INFERENCE_BACKEND=tensorrt .
# TensorRT (GPU) — OpenCV media backend
docker build -t rfdetr-trt-opencv --build-arg INFERENCE_BACKEND=tensorrt --build-arg MEDIA_BACKEND=opencv .
```

Run (mount your model, image, and labels under `/data`):

```bash
# ONNX Runtime — use an .onnx model
docker run -v $(pwd)/data:/data -v $(pwd)/exports:/exports rfdetr-onnx-ffmpeg \
  /exports/model.onnx /data/dog.jpg /data/coco-labels-91.txt

# TensorRT — requires --gpus all and a .engine/.trt model
docker run --gpus all -v $(pwd)/data:/data -v $(pwd)/exports:/exports rfdetr-trt-opencv \
  /exports/model.engine /data/dog.jpg /data/coco-labels-91.txt
```

> The ONNX Runtime images are multi-stage and slim (Ubuntu 24.04 runtime). The TensorRT images use the `nvcr.io/nvidia/tensorrt:25.12-py3` base for the bundled CUDA/TensorRT runtime.

---

## Code Quality Tools

| Tool | Purpose | How to run |
|------|---------|------------|
| `clang-format-18` | Code formatting | `find src tests -name '*.cpp' -o -name '*.hpp' \| xargs clang-format-18 -i` |
| `clang-tidy-18` | Static analysis (AST-based) | `find src -name '*.cpp' \| xargs clang-tidy-18 -p build` |
| `cppcheck` | Static analysis (flow-based) | `cppcheck --enable=all --std=c++20 -I src src/` |
| AddressSanitizer(ASan) + UndefinedBehaviorSanitizer(UBSan) | Runtime memory/UB detection | `-DSANITIZERS=ON` at configure time |
| Strict UndefinedBehaviorSanitizer (UBSan) | Extra bounds and vptr checks; Clang also enables implicit-conversion | `-DSTRICT_UBSAN=ON` at configure time |
| ThreadSanitizer (TSan) | Runtime data-race detection | `-DTHREAD_SANITIZER=ON` at configure time |
| Valgrind (memcheck) | Memory errors + leak detection | `cmake --build build-valg --target memcheck` |
| Valgrind (callgrind/massif) | CPU/cache + heap profiling | `cmake --build build-valg --target callgrind` / `massif` |
| pre-commit | Automates format + cppcheck on commit | `pre-commit install` |

---

## Acknowledgements

- The RF-DETR model used in this project is sourced from **Roboflow**, special thanks to the Roboflow team — check out their [GitHub repository](https://github.com/roboflow/rf-detr) and [site](https://blog.roboflow.com/rf-detr/).
- **Postprocessing implementation** is based on Roboflow's reference implementations:
  - Detection postprocessing: [benchmark_rfdetr.py](https://github.com/roboflow/single_artifact_benchmarking/blob/main/sab/models/benchmark_rfdetr.py)
  - Instance segmentation postprocessing: [benchmark_rfdetr_seg.py](https://github.com/roboflow/single_artifact_benchmarking/blob/main/sab/models/benchmark_rfdetr_seg.py)
  - Keypoint postprocessing: [postprocess.py](https://github.com/roboflow/rf-detr/blob/develop/src/rfdetr/models/postprocess.py)
