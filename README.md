# RF-DETR C++ Inference

[![C++](https://img.shields.io/badge/language-C++20-blue.svg)](https://en.cppreference.com/w/cpp)
[![CMake](https://img.shields.io/badge/build%20system-CMake-blue.svg)](https://cmake.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.4.0-blue.svg)](https://github.com/olibartfast/rf-detr-cpp-inference/releases/tag/v0.4.0)

C++ project for performing object detection, instance segmentation, and keypoint inference using the RF-DETR model with **multiple inference backends** (ONNX Runtime, TensorRT, and ExecuTorch) and a swappable **media/display backend** (FFmpeg + SDL2 + stb by default, or OpenCV). Supports both single-image and **multi-threaded video processing** via a zero-copy ring buffer pipeline, plus an opt-in **GPU pipeline** (DALI preprocessing + CUDA segmentation postprocessing) on the TensorRT backend.

---

## Documentation

| Document | Covers |
|----------|--------|
| **[docs/building.md](docs/building.md)** | Toolchain install, every build configuration, dependency-resolution modes, the ExecuTorch install prefix, the GPU pipeline build |
| **[docs/usage.md](docs/usage.md)** | Every run mode and command-line flag, tuning, `Config` reference |
| **[docs/architecture.md](docs/architecture.md)** | GPU pipeline, video ring buffer, model output shapes, processing stages |
| **[docs/development.md](docs/development.md)** | Formatting, static analysis, sanitizers, Valgrind, tests, benchmarks |
| **[docs/docker.md](docs/docker.md)** | The parametric `Dockerfile` and its image matrix |
| **[docs/export.md](docs/export.md)** | Exporting `.onnx` / `.engine` / `.pte` models from `rfdetr` |
| **[docs/glossary.md](docs/glossary.md)** | Terms used across the codebase |
| **[docs/package-manager-architecture.md](docs/package-manager-architecture.md)** | How `find_dependency_unified` resolves each dependency |
| **[docs/rented-gpu-runbook.md](docs/rented-gpu-runbook.md)** | Running the GPU verification gate on rented hardware |

This page keeps the reference material: supported versions, backend constraints, and the
full CMake option list.

---

## Table of Contents
- [Dependencies](#dependencies)
- [Model Setup](#model-setup)
- [Quick Start](#quick-start)
- [Backend Selection](#backend-selection)
- [Build Options](#build-options)
- [Usage](#usage)
- [Testing and CI](#testing-and-ci)
- [Acknowledgements](#acknowledgements)

---

## Dependencies

### Required (All Backends)
- **C++20 Compiler**: Clang 15+ or GCC 12+ (e.g., `clang++-15` or `g++-12`)
- **CMake**: Version 3.12 or higher (**3.17+** if you let ExecuTorch fall back to the FetchContent source build, which fetches submodules recursively — supplying `-DEXECUTORCH_ROOTDIR` avoids that requirement)
- **Google Test**: 1.12.1 (auto-fetched; see [Dependency Resolution](docs/building.md#dependency-resolution))
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
- **RF-DETR export package**: `rfdetr[onnx]==1.9.4` from `deploy/requirements.txt`
- **ExecuTorch export (optional)**: `rfdetr[executorch]==1.9.4` — only needed to produce `.pte` models for the ExecuTorch backend. Pin it: the extra constrains ExecuTorch only to `>=1.3,<2.0`. Check what pip actually installed (`pip show executorch`) and ensure it matches the C++ runtime this project pins to v1.4.0
- **TensorRT export (optional)**: `rfdetr[tensorrt]==1.9.4` — provides `tensorrt` + `polygraphy` for in-process engine builds (1.9.0+); `pycuda` moved to the separate `rfdetr[tensorrt-bench]` extra
- **Python**: 3.10+ (Python 3.11 virtual environment recommended)
- **pre-commit**: Optional for local hooks; install with `pip install pre-commit`

### Backend-Specific Dependencies

#### ONNX Runtime Backend (Default)
- **ONNX Runtime**: Version 1.21.0 — the official CPU archive is downloaded automatically, selected from the **target** OS and architecture (`CMAKE_SYSTEM_NAME` / `CMAKE_SYSTEM_PROCESSOR`), so cross-compiling picks the target's archive rather than the host's:

  | Target | Archive |
  |--------|---------|
  | Linux x86_64 / amd64 | `onnxruntime-linux-x64-1.21.0.tgz` |
  | Linux aarch64 / arm64 | `onnxruntime-linux-aarch64-1.21.0.tgz` |
  | Windows x86_64 / amd64 | `onnxruntime-win-x64-1.21.0.zip` |
  | Windows arm64 | `onnxruntime-win-arm64-1.21.0.zip` |

- **Platform**: Any combination in the table above works out of the box. Anything else (macOS, 32-bit Windows, other processors) is a configure-time `FATAL_ERROR` telling you to supply your own build — point `-DONNXRUNTIME_ROOTDIR=<prefix>` at it, or use the conan/vcpkg coordinates (`onnxruntime/1.21.0` / `onnxruntime`)
- **Acceleration**: CPU only. `OnnxRuntimeBackend` creates its session without appending an execution provider, so even a CUDA or DirectML build of ONNX Runtime runs on CPU here until the backend is extended to register one

#### TensorRT Backend (Optional)
- **TensorRT**: Version 10.13.3.9 (automatically downloaded during build if not found)
- **CUDA Toolkit**: Version 13.x for the bundled TensorRT 10.13.3.9 archive - **must be installed manually**
- **Platform**: Linux with NVIDIA GPU
- **Acceleration**: NVIDIA GPU only
- **Note**: TensorRT libraries are automatically configured with RPATH, no LD_LIBRARY_PATH needed

#### GPU Pipeline (Optional, TensorRT only)
- **CUDA Toolkit** (with `nvcc` for `-DUSE_CUDA_POSTPROCESS=ON`): resolved via CMake's `FindCUDAToolkit`; CUB (header-only, bundled with the toolkit) is used by the postprocessing kernels
- **NVIDIA DALI** (for `-DUSE_DALI=ON`): C API libraries + headers staged from a pinned Triton container (`nvcr.io/nvidia/tritonserver:25.12-py3`) via `./scripts/fetch_dali.sh` — NVIDIA ships no standalone C++ DALI distribution. Point the build at the staged directory with `-DDALI_ROOT=<dir>`.
- See [GPU Pipeline](docs/architecture.md#gpu-pipeline) for how it works, and [Building](docs/building.md#build-with-the-gpu-pipeline-tensorrt--dali--cuda) for the build
#### ExecuTorch Backend (Optional)
- **ExecuTorch**: Version v1.4.0, built with `EXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON` — resolved from an install prefix via `-DEXECUTORCH_ROOTDIR`, otherwise built from source
- **Model format**: `.pte`, exported by `rfdetr[executorch]` 1.9.0+ (1.9.1+ exports require the optimized kernel set — see [Building the ExecuTorch install prefix](docs/building.md#building-the-executorch-install-prefix))
- **Delegate**: XNNPACK (default) or portable CPU kernels, selected with `-DEXECUTORCH_DELEGATE`
- **Platform**: Linux; CPU inference through the linked delegate
- **Note**: The delegate linked here must match the one baked into the `.pte` at export time

---

## Model Setup

This project supports both RF-DETR detection and segmentation models from Roboflow.

1. **Visit the RF-DETR Repository**:
   - Go to the [RF-DETR GitHub repository](https://github.com/roboflow/rf-detr) for model details.
   - Read the [Roboflow blog](https://blog.roboflow.com/rf-detr/) for an overview.

2. **Download the ONNX Model**:
   - Follow instructions in the [export documentation](docs/export.md) to export models in ONNX format.
   - **Tested with**: `rfdetr[onnx]==1.9.4` (Python 3.10+; 3.11 venv recommended)
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

## Quick Start

Ubuntu toolchain and media libraries, then a default (ONNX Runtime) build:

```bash
sudo apt-get update
sudo apt-get install -y cmake ninja-build pkg-config \
  libavcodec-dev libavformat-dev libavutil-dev libswscale-dev libsdl2-dev

cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel

./build/inference_app model.onnx data/dog.jpg data/coco-labels-91.txt
```

The other three backends, in one line each — see
[docs/building.md](docs/building.md) for what each one needs and what it does:

```bash
# TensorRT (NVIDIA GPU, .engine/.trt/.onnx)
cmake -S . -B build -G Ninja -DUSE_ONNX_RUNTIME=OFF -DUSE_TENSORRT=ON -DCMAKE_BUILD_TYPE=Release

# ExecuTorch (CPU, .pte) — needs an install prefix
cmake -S . -B build -G Ninja -DUSE_ONNX_RUNTIME=OFF -DUSE_EXECUTORCH=ON \
  -DEXECUTORCH_ROOTDIR=$HOME/dependencies/executorch -DCMAKE_BUILD_TYPE=Release

# TensorRT + GPU pipeline (DALI preprocessing + CUDA postprocessing)
./scripts/fetch_dali.sh
cmake -S . -B build -G Ninja -DUSE_ONNX_RUNTIME=OFF -DUSE_TENSORRT=ON \
  -DUSE_GPU_PIPELINE=ON -DDALI_ROOT=$HOME/dependencies/dali -DCMAKE_BUILD_TYPE=Release
```

---

## Backend Selection

This project uses **compile-time backend selection**. Choose your backend when building:

| Backend | Model format | Best For | Pros | Cons |
|---------|--------------|----------|------|------|
| **ONNX Runtime** | `.onnx` | Development, CPU inference | Easy setup, no GPU or extra SDK needed | CPU only as shipped — the download is the CPU archive for the target platform and no execution provider is registered |
| **TensorRT** | `.engine` / `.trt` (also accepts `.onnx`, building/caching an engine beside it) | Production on NVIDIA GPUs | Maximum performance | GPU-only, requires CUDA/TensorRT |
| **ExecuTorch** | `.pte` | On-device / edge deployment | Small runtime, delegate-based (XNNPACK) | Requires an ExecuTorch install; rfdetr 1.9.0+ to export |

**Important**: Only ONE backend can be enabled at a time — enabling two is a configure-time error. The backend is compiled into the binary for optimal performance and smaller binary size.

---

## Build Options

- `-DUSE_ONNX_RUNTIME=ON/OFF` - Enable ONNX Runtime backend (default: ON)
- `-DUSE_TENSORRT=ON/OFF` - Enable TensorRT backend (default: OFF)
- `-DUSE_EXECUTORCH=ON/OFF` - Enable ExecuTorch backend for `.pte` models (default: OFF)
- `-DEXECUTORCH_ROOTDIR=<path>` - ExecuTorch install prefix; without it ExecuTorch is built from source
- `-DEXECUTORCH_DELEGATE=xnnpack/portable` - ExecuTorch delegate library to link (default: xnnpack)
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

```bash
# Object detection
./build/inference_app model.onnx image.jpg coco-labels-91.txt

# Instance segmentation
./build/inference_app model.onnx image.jpg coco-labels-91.txt --segmentation

# Keypoint detection
./build/inference_app model.onnx image.jpg coco-labels-91.txt --keypoint

# Video (multi-threaded pipeline), with a live preview window
./build/inference_app model.onnx video.mp4 coco-labels-91.txt --display

# GPU pipeline (TensorRT builds compiled with it)
./build/inference_app model.engine image.jpg coco-labels-91.txt \
  --segmentation --gpu-preprocess --gpu-postprocess
```

Output goes to `output_image.jpg` or `output_video.mp4`. Input resolution is auto-detected
from the model, and video files are recognised by extension.

Tuning flags (`--threshold`, `--resolution`, `--max-detections`, `--mask-threshold`,
`--background-class-id`), how detections are ranked, and the full `Config` reference are in
**[docs/usage.md](docs/usage.md)**.

> [!WARNING]
> Keypoint models exported with `rfdetr` 1.8.2 or later use the active-first schema (`[17]`) and are
> not decodable by the default build, which still expects background-first `{0, 17}`. See the
> keypoint warning in [docs/export.md](docs/export.md#keypoint-model-export).

---

## Testing and CI

```bash
ctest --test-dir build --output-on-failure -R UnitTests
```

Integration tests need a real model in the format the compiled-in backend accepts; sanitizers,
Valgrind and benchmarks have their own build directories. All of it is in
**[docs/development.md](docs/development.md)**.

Three GitHub Actions workflows run on every push/PR to `master` and `develop`:

| Workflow | File | What it does |
|----------|------|-------------|
| **C++ Lint & Build** | `lint.yml` | Format check, clang-tidy, cppcheck, build with `-DWERROR=ON` |
| **Build & Test** | `ci.yml` | Build with benchmarks, run unit tests, run benchmarks, run unit tests under ASan+UBSan |
| **GPU Backend Compile** | `gpu-compile.yml` | Compiles the TensorRT backend and both GPU halves with `-DWERROR=ON`, across all four `USE_DALI`/`USE_CUDA_POSTPROCESS` combinations |

`gpu-compile.yml` runs on GPU-less runners, so it compiles but never links or executes. It stages
headers-only TensorRT and DALI prefixes with `scripts/ci/stage_gpu_headers.sh` — the full TensorRT
tarball is 6.2 GB and the DALI wheel 380 MB, against ~130 KB of TensorRT header debs and a few MB
of DALI headers — and builds only the `rfdetr_inference_lib` static target. Runtime behaviour of
the GPU path is still gated on manual verification, on real hardware.

---

## Acknowledgements

- The RF-DETR model used in this project is sourced from **Roboflow**, special thanks to the Roboflow team — check out their [GitHub repository](https://github.com/roboflow/rf-detr) and [site](https://blog.roboflow.com/rf-detr/).
- **Postprocessing implementation** is based on Roboflow's reference implementations:
  - Detection postprocessing: [benchmark_rfdetr.py](https://github.com/roboflow/single_artifact_benchmarking/blob/main/sab/models/benchmark_rfdetr.py)
  - Instance segmentation postprocessing: [benchmark_rfdetr_seg.py](https://github.com/roboflow/single_artifact_benchmarking/blob/main/sab/models/benchmark_rfdetr_seg.py)
  - Keypoint postprocessing: [postprocess.py](https://github.com/roboflow/rf-detr/blob/develop/src/rfdetr/models/postprocess.py)
