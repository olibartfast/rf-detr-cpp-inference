# RF-DETR C++ Inference

[![C++](https://img.shields.io/badge/language-C++20-blue.svg)](https://en.cppreference.com/w/cpp)
[![CMake](https://img.shields.io/badge/build%20system-CMake-blue.svg)](https://cmake.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.5.1-blue.svg)](https://github.com/olibartfast/rf-detr-cpp-inference/releases/tag/v0.5.1)

Object detection, instance segmentation, and keypoint inference with the
[RF-DETR](https://github.com/roboflow/rf-detr) model, in C++20.

Point it at a model, an image or a video, and a labels file — it writes back an annotated
file. Images and videos both work out of the box; video runs through a multi-threaded,
zero-copy pipeline.

**Start here, then go deeper:** this page is the short path — install, build, run.
[docs/usage.md](docs/usage.md) is the full command-line reference, and everything past it —
the complete CMake option list, backend internals, tuning strategy, and embedding the library
in your own code — lives in **[docs/advanced-usage.md](docs/advanced-usage.md)**.

---

## Table of Contents
- [Quick Start](#quick-start) — from a clean Ubuntu box to an annotated image
- [Running It](#running-it) — detection, segmentation, keypoints, video
- [Common Flags](#common-flags)
- [Choosing a Backend](#choosing-a-backend)
- [Versions at a Glance](#versions-at-a-glance)
- [Common Build Options](#common-build-options)
- [Documentation](#documentation)
- [Testing](#testing)
- [Acknowledgements](#acknowledgements)

---

## Quick Start

Four steps on Ubuntu, using the default backend (ONNX Runtime, CPU — no GPU, no CUDA,
no extra SDK).

### 1. Install the toolchain

```bash
sudo apt-get update
sudo apt-get install -y cmake ninja-build pkg-config \
  libavcodec-dev libavformat-dev libavutil-dev libswscale-dev libsdl2-dev \
  python3 python3-venv
```

`python3-venv` is what step 3 needs — without it `python3 -m venv` fails with
`ensurepip is not available`. The rest are the media libraries for the default
FFmpeg + SDL2 + stb backend. Prefer OpenCV?
Install `libopencv-dev` instead and build with `-DUSE_OPENCV=ON` — see
[Media backends](docs/advanced-usage.md#media--display-backends).

### 2. Build

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

ONNX Runtime is downloaded automatically for your platform — nothing to install by hand.

### 3. Export a model

The model is not shipped with the repo; export one from the `rfdetr` Python package:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r deploy/requirements.txt   # the pinned rfdetr[onnx]
python deploy/export_detection.py --model_type nano     # -> output/rfdetr-nano.onnx
```

A COCO labels file is already in the repo at `data/coco-labels-91.txt`, and a sample image
at `data/dog.jpg`. Segmentation, keypoint, TensorRT and ExecuTorch exports are in
**[docs/export.md](docs/export.md)**.

### 4. Run

```bash
./build/inference_app output/rfdetr-nano.onnx data/dog.jpg data/coco-labels-91.txt
```

Detections are printed to the console and drawn into `output_image.jpg`.

---

## Running It

The three positional arguments are always the same — **model**, **input**, **labels** — and
the mode is picked by a flag. Video files are recognised by extension, so the same command
line handles both.

```bash
# Object detection (default)
./build/inference_app model.onnx image.jpg coco-labels-91.txt

# Instance segmentation — needs a model exported with masks
./build/inference_app model.onnx image.jpg coco-labels-91.txt --segmentation

# Keypoint detection — needs a keypoint export
./build/inference_app model.onnx image.jpg coco-labels-91.txt --keypoint

# Video, with a live preview window (ESC quits)
./build/inference_app model.onnx video.mp4 coco-labels-91.txt --display
```

Output goes to `output_image.jpg` or `output_video.mp4` unless you pass `--output`. The
model's input resolution is auto-detected, so there is nothing to configure for a normally
exported model.

Supported video containers: `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`, `.flv`, `.wmv`.

---

## Common Flags

| Flag | Default | Effect |
|------|---------|--------|
| `--segmentation` | off | Instance segmentation; draws masks with a transparent overlay |
| `--keypoint` | off | Keypoint detection; draws the COCO 17-keypoint skeleton |
| `--display` | off | Live preview window while processing a video |
| `--threshold <val>` | `0.5` | Confidence threshold for keeping a detection |
| `--output <path>` | `output_image.jpg` / `output_video.mp4` | Where to write the annotated result |

Every remaining flag — `--resolution`, `--max-detections`, `--mask-threshold`,
`--background-class-id`, `--keypoint-counts`, and the GPU pipeline flags — is documented in
**[docs/usage.md](docs/usage.md)**.

> [!NOTE]
> The official keypoint checkpoint is background-first and decodes with the default config. An
> active-first (`[17]`) export needs `--keypoint-counts 17 --background-class-id none`. See the
> keypoint note in [docs/export.md](docs/export.md#keypoint-model-export).

---

## Choosing a Backend

Exactly **one** inference backend is compiled into the binary; enabling two is a
configure-time error.

| Backend | Model format | Best for | Build with |
|---------|--------------|----------|------------|
| **ONNX Runtime** (default) | `.onnx` | Development, CPU inference | *(nothing — it is the default)* |
| **TensorRT** | `.engine` / `.trt` (also `.onnx`) | Production on NVIDIA GPUs | `-DUSE_ONNX_RUNTIME=OFF -DUSE_TENSORRT=ON` |
| **ExecuTorch** | `.pte` | On-device / edge deployment | `-DUSE_ONNX_RUNTIME=OFF -DUSE_EXECUTORCH=ON -DEXECUTORCH_ROOTDIR=<prefix>` |

```bash
# TensorRT (NVIDIA GPU) — the CUDA Toolkit series CUDA_VERSION pins must already be installed
cmake -S . -B build -G Ninja -DUSE_ONNX_RUNTIME=OFF -DUSE_TENSORRT=ON -DCMAKE_BUILD_TYPE=Release

# ExecuTorch (CPU, .pte) — needs an install prefix
cmake -S . -B build -G Ninja -DUSE_ONNX_RUNTIME=OFF -DUSE_EXECUTORCH=ON \
  -DEXECUTORCH_ROOTDIR=$HOME/dependencies/executorch -DCMAKE_BUILD_TYPE=Release

# TensorRT + the optional GPU pipeline (DALI preprocessing + CUDA postprocessing)
./scripts/fetch_dali.sh
cmake -S . -B build -G Ninja -DUSE_ONNX_RUNTIME=OFF -DUSE_TENSORRT=ON \
  -DUSE_GPU_PIPELINE=ON -DDALI_ROOT=$HOME/dependencies/dali -DCMAKE_BUILD_TYPE=Release
```

The trade-offs, platform limits, and per-backend constraints — including why the ONNX Runtime
backend is CPU-only as shipped — are in
[Backends in depth](docs/advanced-usage.md#backends-in-depth); the step-by-step builds are in
[docs/building.md](docs/building.md).

---

## Versions at a Glance

> **Every version below is pinned once, in [`versions.env`](versions.env)** — CMake reads it
> through `cmake/versions.cmake`, the shell scripts through `scripts/versions.sh`. A bump is a
> one-line edit, and any pin is overridable without editing the file
> (`cmake -DTENSORRT_VERSION=…`, `TRITON_IMAGE=… ./scripts/fetch_dali.sh`).
> See [Overriding a pinned version](docs/advanced-usage.md#overriding-a-pinned-version).

| Component | Version | Needed for |
|-----------|---------|-----------|
| C++ compiler | Clang 15+ or GCC 12+ (C++20) | everything |
| CMake | 3.12+ (3.17+ for the ExecuTorch source fallback) | everything |
| **ONNX Runtime** | **1.21.0** | default backend — downloaded automatically |
| **TensorRT** | **11.2.1.2** + CUDA Toolkit **13.3** series | TensorRT backend (CUDA installed manually) |
| **ExecuTorch** | **v1.4.0** | ExecuTorch backend |
| NVIDIA DALI | 2.2.0 (staged from `nvcr.io/nvidia/tritonserver:26.08-py3`) | `-DUSE_DALI=ON` |
| FFmpeg / SDL2 | 5.x+ / 2.x (Conan pins 6.1 / 2.28.5) | default media backend — apt takes whatever the system has |
| OpenCV | 4.x (Conan coordinate 4.8.1) | `-DUSE_OPENCV=ON` |
| GoogleTest | 1.12.1 (auto-fetched) | tests |
| Google Benchmark | 1.9.1 | `-DBENCHMARKS=ON` |

**Python export tooling** (`deploy/requirements.txt`, Python 3.10+; a 3.11 venv is recommended):

| Package | Pin | Provides |
|---------|-----|----------|
| `rfdetr[onnx]` | `==1.10.1` | `.onnx` export, ONNX opset 17 |
| `rfdetr[executorch]` | `==1.10.1` | `.pte` export — check `pip show executorch` matches the pinned v1.4.0 runtime |
| `rfdetr[tensorrt]` | `==1.10.1` | in-process engine builds (`tensorrt` + `polygraphy`) |

These two tables are the only prose that states pinned versions; everything else in `docs/`
and `specs/` points at `versions.env`. The tables, the `ARG` defaults in the backend Dockerfiles,
`conanfile.txt`, `deploy/requirements.txt`, and the `deploy/export_*.py` opset defaults cannot
read a file, so they restate these values; `./scripts/check_version_sync.sh` (CI job
`Version Sync`) fails when any of them drifts. See
[Bumping a version](specs/tech-stack.md#bumping-a-version).

---

## Common Build Options

The handful you are likely to need. **The complete option list is in
[docs/advanced-usage.md](docs/advanced-usage.md#cmake-option-reference).**

| Option | Default | Effect |
|--------|---------|--------|
| `-DCMAKE_BUILD_TYPE=Release/Debug` | — | Build configuration; use `Release` for any timing |
| `-DUSE_ONNX_RUNTIME=ON/OFF` | `ON` | ONNX Runtime backend |
| `-DUSE_TENSORRT=ON/OFF` | `OFF` | TensorRT backend |
| `-DUSE_EXECUTORCH=ON/OFF` | `OFF` | ExecuTorch backend for `.pte` models |
| `-DUSE_OPENCV=ON/OFF` | `OFF` | Use OpenCV for image/video/display I/O instead of FFmpeg+SDL2+stb |
| `-DUSE_GPU_PIPELINE=ON/OFF` | `OFF` | DALI preprocessing **and** CUDA postprocessing (TensorRT only) |
| `-DBENCHMARKS=ON/OFF` | `OFF` | Build the Google Benchmark targets |
| `-DSANITIZERS=ON/OFF` | `OFF` | AddressSanitizer + UndefinedBehaviorSanitizer |
| `-DWERROR=ON/OFF` | `OFF` | Treat compiler warnings as errors (what CI does) |

`CMakePresets.json` carries the default build, four diagnostic CPU presets, and `gpu-pipeline`
for TensorRT + DALI + CUDA.

---

## Documentation

| Document | Covers |
|----------|--------|
| **[docs/usage.md](docs/usage.md)** | Every run mode and every command-line flag — the operational reference |
| **[docs/advanced-usage.md](docs/advanced-usage.md)** | **Advanced usage and customization** — full CMake option reference, backends in depth, tuning strategy, class layouts, the GPU pipeline at runtime, the `Config` reference, embedding `RFDETRInference` in your own code |
| **[docs/building.md](docs/building.md)** | Toolchain install, every build configuration, dependency-resolution modes, the ExecuTorch install prefix, the GPU pipeline build |
| **[docs/export.md](docs/export.md)** | Exporting `.onnx` / `.engine` / `.pte` models from `rfdetr` |
| **[docs/architecture.md](docs/architecture.md)** | GPU pipeline, video ring buffer, model output shapes, processing stages |
| **[docs/development.md](docs/development.md)** | Formatting, static analysis, sanitizers, Valgrind, tests, benchmarks |
| **[docs/docker.md](docs/docker.md)** | The three backend Dockerfiles and their image matrix |
| **[docs/glossary.md](docs/glossary.md)** | Terms used across the codebase |
| **[docs/package-manager-architecture.md](docs/package-manager-architecture.md)** | How `find_dependency_unified` resolves each dependency |

---

## Testing

```bash
ctest --test-dir build --output-on-failure -R UnitTests
```

Integration tests need a real model in the format the compiled-in backend accepts; sanitizers,
Valgrind and benchmarks have their own build directories. All of it is in
**[docs/development.md](docs/development.md)**.

Three GitHub Actions workflows run on every push/PR to `master` and `develop`:

| Workflow | File | What it does |
|----------|------|-------------|
| **C++ Lint & Build** | `lint.yml` | Version sync, format check, clang-tidy, cppcheck, build with `-DWERROR=ON` |
| **Build & Test** | `ci.yml` | Build with benchmarks, run unit tests, run benchmarks, run unit tests under ASan+UBSan |
| **GPU Backend Compile** | `gpu-compile.yml` | Compiles the TensorRT backend and both GPU halves with `-DWERROR=ON`, across all four `USE_DALI`/`USE_CUDA_POSTPROCESS` combinations, plus the full pipeline against TensorRT 11 headers |

CI compiles the GPU paths but cannot execute them — see
[CI coverage and its limits](docs/advanced-usage.md#ci-coverage-and-its-limits).

---

## Acknowledgements

- The RF-DETR model used in this project is sourced from **Roboflow**, special thanks to the Roboflow team — check out their [GitHub repository](https://github.com/roboflow/rf-detr) and [site](https://blog.roboflow.com/rf-detr/).
- **Postprocessing implementation** is based on Roboflow's reference implementations:
  - Detection postprocessing: [benchmark_rfdetr.py](https://github.com/roboflow/single_artifact_benchmarking/blob/main/sab/models/benchmark_rfdetr.py)
  - Instance segmentation postprocessing: [benchmark_rfdetr_seg.py](https://github.com/roboflow/single_artifact_benchmarking/blob/main/sab/models/benchmark_rfdetr_seg.py)
  - Keypoint postprocessing: [postprocess.py](https://github.com/roboflow/rf-detr/blob/develop/src/rfdetr/models/postprocess.py)
