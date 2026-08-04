# RF-DETR C++ Inference

[![C++](https://img.shields.io/badge/language-C++20-blue.svg)](https://en.cppreference.com/w/cpp)
[![CMake](https://img.shields.io/badge/build%20system-CMake-blue.svg)](https://cmake.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.4.0-blue.svg)](https://github.com/olibartfast/rf-detr-cpp-inference/releases/tag/v0.4.0)

C++ project for performing object detection, instance segmentation, and keypoint inference using the RF-DETR model with **multiple inference backends** (ONNX Runtime, TensorRT, and ExecuTorch) and a swappable **media/display backend** (FFmpeg + SDL2 + stb by default, or OpenCV). Supports both single-image and **multi-threaded video processing** via a zero-copy ring buffer pipeline.

---

## Table of Contents
- [Dependencies](#dependencies)
- [Model Setup](#model-setup)
- [Installation](#installation)
- [Building](#building)
- [Usage](#usage)
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
- **CMake**: Version 3.12 or higher (**3.17+** if you let ExecuTorch fall back to the FetchContent source build, which fetches submodules recursively — supplying `-DEXECUTORCH_ROOTDIR` avoids that requirement)
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
- **RF-DETR export package**: `rfdetr[onnx]==1.9.0` from `deploy/requirements.txt`
- **ExecuTorch export (optional)**: `rfdetr[executorch]==1.9.0` — only needed to produce `.pte` models for the ExecuTorch backend. Pin it: the extra does not constrain ExecuTorch itself, and 1.9.0 resolves ExecuTorch 1.3.1, matching the C++ runtime this project pins
- **TensorRT export (optional)**: `rfdetr[tensorrt]==1.9.0` — provides `tensorrt` + `polygraphy` for in-process engine builds (1.9.0+); `pycuda` moved to the separate `rfdetr[tensorrt-bench]` extra
- **Python**: 3.10+ (Python 3.11 virtual environment recommended)
- **pre-commit**: Optional for local hooks; install with `pip install pre-commit`

### Backend-Specific Dependencies

#### ONNX Runtime Backend (Default)
- **ONNX Runtime**: Version 1.21.0 — the default download is the **Linux x64 CPU** archive (`onnxruntime-linux-x64-1.21.0.tgz`)
- **Platform**: Linux x64 out of the box. For other platforms, supply your own build — point `-DONNXRUNTIME_ROOTDIR=<prefix>` at it, or use the conan/vcpkg coordinates (`onnxruntime/1.21.0` / `onnxruntime`)
- **Acceleration**: CPU only. `OnnxRuntimeBackend` creates its session without appending an execution provider, so even a CUDA or DirectML build of ONNX Runtime runs on CPU here until the backend is extended to register one

#### TensorRT Backend (Optional)
- **TensorRT**: Version 10.13.3.9 (automatically downloaded during build if not found)
- **CUDA Toolkit**: Version 13.x for the bundled TensorRT 10.13.3.9 archive - **must be installed manually**
- **Platform**: Linux with NVIDIA GPU
- **Acceleration**: NVIDIA GPU only
- **Note**: TensorRT libraries are automatically configured with RPATH, no LD_LIBRARY_PATH needed

#### ExecuTorch Backend (Optional)
- **ExecuTorch**: Version v1.3.1 — resolved from an install prefix via `-DEXECUTORCH_ROOTDIR`, otherwise built from source
- **Model format**: `.pte`, exported by `rfdetr[executorch]` 1.9.0+
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
   - **Tested with**: `rfdetr[onnx]==1.9.0` (Python 3.10+; 3.11 venv recommended)
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

| Backend | Model format | Best For | Pros | Cons |
|---------|--------------|----------|------|------|
| **ONNX Runtime** | `.onnx` | Development, CPU inference | Easy setup, no GPU or extra SDK needed | CPU only as shipped — the default download is the Linux x64 CPU archive and no execution provider is registered |
| **TensorRT** | `.engine` / `.trt` (also accepts `.onnx`, building/caching an engine beside it) | Production on NVIDIA GPUs | Maximum performance | GPU-only, requires CUDA/TensorRT |
| **ExecuTorch** | `.pte` | On-device / edge deployment | Small runtime, delegate-based (XNNPACK) | Requires an ExecuTorch install; rfdetr 1.9.0+ to export |

**Important**: Only ONE backend can be enabled at a time — enabling two is a configure-time error. The backend is compiled into the binary for optimal performance and smaller binary size.

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
ExecuTorch is in no registry, so it resolves the same way in every mode:
`find_package(executorch CONFIG)` against `-DEXECUTORCH_ROOTDIR`, else a
FetchContent source build.

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

### Build with ExecuTorch Backend

Runs `.pte` programs exported by `rfdetr >= 1.9.0`. Point the build at an ExecuTorch install
prefix — the directory containing `lib/cmake/ExecuTorch/executorch-config.cmake`:

```bash
cmake -S . -B build -G Ninja \
  -DUSE_ONNX_RUNTIME=OFF \
  -DUSE_EXECUTORCH=ON \
  -DEXECUTORCH_ROOTDIR=$HOME/dependencies/executorch \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build --parallel
```

#### Building the ExecuTorch install prefix

ExecuTorch **v1.3.1** is the pinned version — it matches the ExecuTorch that
`rfdetr[executorch]==1.9.0` installs for the Python exporter, and `.pte` schema
compatibility across runtime versions is not guaranteed.

```bash
git clone --depth 1 -b v1.3.1 https://github.com/pytorch/executorch.git
cd executorch && git submodule update --init --recursive --depth 1

# ExecuTorch runs operator codegen through PYTHON_EXECUTABLE during its own
# configure, and that code does `import torchgen` — so a bare system python3 is
# not enough. Only torchgen is used, so the CPU-only torch wheel suffices.
python3 -m venv /tmp/et-venv
/tmp/et-venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch
/tmp/et-venv/bin/pip install pyyaml setuptools

# Required: upstream v1.3.1 installs extension_evalue_util into the *build* tree
# instead of the install prefix, which makes find_package fail later. See note below.
sed -i 's|DESTINATION ${CMAKE_BINARY_DIR}/lib|DESTINATION ${CMAKE_INSTALL_LIBDIR}|' \
    extension/evalue_util/CMakeLists.txt

cmake -S . -B cmake-out -GNinja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=$HOME/dependencies/executorch \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
  -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
  -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
  -DEXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP=ON \
  -DEXECUTORCH_BUILD_XNNPACK=ON \
  -DPYTHON_EXECUTABLE=/tmp/et-venv/bin/python

cmake --build cmake-out -j"$(nproc)" && cmake --install cmake-out
```

Build ExecuTorch with the **same compiler** you build this project with, so both
link against one C++ runtime.

> **Upstream bug (v1.3.1)** — `extension/evalue_util/CMakeLists.txt:27` uses
> `DESTINATION ${CMAKE_BINARY_DIR}/lib` where every other extension uses
> `${CMAKE_INSTALL_LIBDIR}`. Without the `sed` above, `libextension_evalue_util.a` never
> reaches `<prefix>/lib` and its exported target keeps an absolute build-tree path, so
> `find_package(executorch CONFIG)` hard-fails once the build tree is deleted — even
> though this project never links that target. The FetchContent fallback is unaffected,
> because it uses the targets directly and never runs the faulty `install()` rule.

**What happens**:
- `EXECUTORCH_ROOTDIR` is added to `CMAKE_PREFIX_PATH` and resolved with `find_package(executorch CONFIG)`
- If no install prefix is found, the build falls back to compiling ExecuTorch v1.3.1 from source (slow; needs a Python interpreter with ExecuTorch's build-time dependencies, since ExecuTorch runs flatbuffers codegen during its own configure)
- `-DEXECUTORCH_DELEGATE=xnnpack` (default) or `portable` selects the delegate library to link, which must match the delegate the `.pte` was exported with — a mismatch fails at run time, not at link time
- At load the backend verifies the program returns `dets` before `labels`, since ExecuTorch outputs are an unnamed tuple and postprocessing addresses them positionally

Export a model with [`deploy/export_executorch.py`](deploy/export_executorch.py); see the [export documentation](docs/export.md#executorch-model-export).

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
- Orthogonal to the inference backend — combine freely, e.g. `-DUSE_ONNX_RUNTIME=OFF -DUSE_TENSORRT=ON -DUSE_OPENCV=ON`
  (`USE_ONNX_RUNTIME` defaults to `ON`, so it must be turned off explicitly when selecting another
  inference backend — enabling two is a configure-time error)

### Build Options

- `-DUSE_ONNX_RUNTIME=ON/OFF` - Enable ONNX Runtime backend (default: ON)
- `-DUSE_TENSORRT=ON/OFF` - Enable TensorRT backend (default: OFF)
- `-DUSE_EXECUTORCH=ON/OFF` - Enable ExecuTorch backend for `.pte` models (default: OFF)
- `-DEXECUTORCH_ROOTDIR=<path>` - ExecuTorch install prefix; without it ExecuTorch is built from source
- `-DEXECUTORCH_DELEGATE=xnnpack/portable` - ExecuTorch delegate library to link (default: xnnpack)
- `-DUSE_OPENCV=ON/OFF` - Use OpenCV for image/video/display I/O instead of FFmpeg+SDL2+stb (default: OFF)
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

**Features:**
- The output image is saved as `output_image.jpg`; video output is saved as `output_video.mp4`
- Detection/segmentation results (bounding boxes, labels, scores, and mask pixels) are printed to the console
- Input resolution is automatically detected from the model (supports 432x432, 560x560, etc.)
- Segmentation mode draws colored masks with transparency overlays
- Uses top-k selection (default: 300 detections) for efficient processing
- Video files are automatically detected by extension and processed with the multi-threaded pipeline

---

## Configuration

The inference engine supports various configuration options that can be modified in `src/main.cpp`:

- **Model Type**: `ModelType::DETECTION`, `ModelType::SEGMENTATION`, or `ModelType::KEYPOINT` (selected via the `--segmentation` / `--keypoint` CLI flags)
- **Resolution**: Set to `0` for auto-detection from model, or specify manually (e.g., `432`, `560`)
- **Confidence Threshold**: Default `0.5` (adjustable in `Config::threshold`)
- **Max Detections**: Default `300` for top-k selection (adjustable in `Config::max_detections`)
- **Mask Threshold**: Default `0.0` for binary mask generation (adjustable in `Config::mask_threshold`)
- **Normalization**: ImageNet mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`

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

Integration tests need a real model **in the format the compiled-in backend accepts**. They probe
`~/Downloads/`, `exports/`, and `output/` for `rfdetr-medium` (or legacy `inference_model`) with the
extensions below, or you can point `RFDETR_TEST_MODEL` at one directly:

| Build | Extensions probed | Example |
|-------|-------------------|---------|
| ONNX Runtime (default) | `.onnx` | `export RFDETR_TEST_MODEL=/path/to/rfdetr-medium.onnx` |
| TensorRT | `.engine`, `.trt`, then `.onnx` | `export RFDETR_TEST_MODEL=/path/to/rfdetr-medium.engine` |
| ExecuTorch | `.pte` | `export RFDETR_TEST_MODEL=/path/to/rfdetr-medium.pte` |

```bash
export RFDETR_TEST_MODEL=/path/to/rfdetr-medium.onnx
ctest --test-dir build --output-on-failure -R IntegrationTests
```

TensorRT keeps `.onnx` as a lower-priority candidate so the ONNX-to-engine conversion path stays
covered; a prebuilt engine is preferred because it loads directly. Keypoint tests use the same
scheme with `rfdetr-keypoint` / `rfdetr-keypoint-preview` and `RFDETR_KEYPOINT_MODEL`.

Without a matching model, integration tests that need inference are skipped, and the skip message
names the format the build expects.

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
| `executorch`        | `ffmpeg`        | ExecuTorch + FFmpeg/SDL2/stb |
| `executorch`        | `opencv`        | ExecuTorch + OpenCV |

> **ExecuTorch images build the ExecuTorch C++ runtime from source** (there is no distro
> or registry package), so the first build is slow — it clones ExecuTorch with recursive
> submodules and installs a CPU-only `torch` wheel for the operator codegen. Pin a
> different runtime with `--build-arg EXECUTORCH_VERSION=<tag>`; it defaults to `v1.3.1`
> to match the exporter that `rfdetr[executorch]==1.9.0` installs. The build applies the
> upstream `extension_evalue_util` install fix automatically. ExecuTorch links
> statically, so the runtime image ships no extra shared libraries and needs no GPU.

Build all six variants:

```bash
# ONNX Runtime (CPU) — FFmpeg/SDL2/stb media backend (default)
docker build -t rfdetr-onnx-ffmpeg .
# ONNX Runtime (CPU) — OpenCV media backend
docker build -t rfdetr-onnx-opencv --build-arg MEDIA_BACKEND=opencv .
# TensorRT (GPU) — FFmpeg/SDL2/stb media backend
docker build -t rfdetr-trt-ffmpeg --build-arg INFERENCE_BACKEND=tensorrt .
# TensorRT (GPU) — OpenCV media backend
docker build -t rfdetr-trt-opencv --build-arg INFERENCE_BACKEND=tensorrt --build-arg MEDIA_BACKEND=opencv .
# ExecuTorch (CPU) — FFmpeg/SDL2/stb media backend
docker build -t rfdetr-et-ffmpeg --build-arg INFERENCE_BACKEND=executorch .
# ExecuTorch (CPU) — OpenCV media backend
docker build -t rfdetr-et-opencv --build-arg INFERENCE_BACKEND=executorch --build-arg MEDIA_BACKEND=opencv .
```

Run (mount your model, image, and labels under `/data`):

```bash
# ONNX Runtime — use an .onnx model
docker run -v $(pwd)/data:/data -v $(pwd)/exports:/exports rfdetr-onnx-ffmpeg \
  /exports/model.onnx /data/dog.jpg /data/coco-labels-91.txt

# TensorRT — requires --gpus all and a .engine/.trt model
docker run --gpus all -v $(pwd)/data:/data -v $(pwd)/exports:/exports rfdetr-trt-opencv \
  /exports/model.engine /data/dog.jpg /data/coco-labels-91.txt

# ExecuTorch — CPU only, use a .pte model exported with the xnnpack delegate
docker run -v $(pwd)/data:/data -v $(pwd)/exports:/exports rfdetr-et-ffmpeg \
  /exports/model.pte /data/dog.jpg /data/coco-labels-91.txt
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
