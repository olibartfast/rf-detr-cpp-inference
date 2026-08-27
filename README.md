# RF-DETR C++ Inference

[![C++](https://img.shields.io/badge/language-C++20-blue.svg)](https://en.cppreference.com/w/cpp)
[![CMake](https://img.shields.io/badge/build%20system-CMake-blue.svg)](https://cmake.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.4.0-blue.svg)](https://github.com/olibartfast/rf-detr-cpp-inference/releases/tag/v0.4.0)

C++ project for performing object detection, instance segmentation, and keypoint inference using the RF-DETR model with **multiple inference backends** (ONNX Runtime, TensorRT, and ExecuTorch) and a swappable **media/display backend** (FFmpeg + SDL2 + stb by default, or OpenCV). Supports both single-image and **multi-threaded video processing** via a zero-copy ring buffer pipeline, plus an opt-in **GPU pipeline** (DALI preprocessing + CUDA segmentation postprocessing) on the TensorRT backend.

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
- See [GPU Pipeline](#gpu-pipeline) for build and usage details
#### ExecuTorch Backend (Optional)
- **ExecuTorch**: Version v1.4.0, built with `EXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON` — resolved from an install prefix via `-DEXECUTORCH_ROOTDIR`, otherwise built from source
- **Model format**: `.pte`, exported by `rfdetr[executorch]` 1.9.0+ (1.9.1+ exports require the optimized kernel set — see [Building the ExecuTorch install prefix](#building-the-executorch-install-prefix))
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
| **ONNX Runtime** | `.onnx` | Development, CPU inference | Easy setup, no GPU or extra SDK needed | CPU only as shipped — the download is the CPU archive for the target platform and no execution provider is registered |
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
The GPU-pipeline dependencies also route through the facade: CUDA Toolkit
resolves via CMake's `FindCUDAToolkit` (apt handler), and DALI is ROOT-only —
staged locally by `./scripts/fetch_dali.sh` and pointed to with `-DDALI_ROOT`
(no download fallback exists, since NVIDIA ships no standalone C++ DALI
distribution).
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

ExecuTorch **v1.4.0** is the pinned C++ runtime. The `rfdetr[executorch]==1.9.4`
extra allows ExecuTorch `>=1.3,<2.0` and does not guarantee that version; `.pte` schema
compatibility across ExecuTorch versions is not guaranteed.

```bash
git clone --depth 1 -b v1.4.0 https://github.com/pytorch/executorch.git
cd executorch && git submodule update --init --recursive --depth 1

# ExecuTorch runs operator codegen through PYTHON_EXECUTABLE during its own
# configure, and that code does `import torchgen` — so a bare system python3 is
# not enough. Only torchgen is used, so the CPU-only torch wheel suffices.
python3 -m venv /tmp/et-venv
/tmp/et-venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch
/tmp/et-venv/bin/pip install pyyaml setuptools

cmake -S . -B cmake-out -GNinja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=$HOME/dependencies/executorch \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
  -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
  -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
  -DEXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP=ON \
  -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON \
  -DEXECUTORCH_BUILD_XNNPACK=ON \
  -DPYTHON_EXECUTABLE=/tmp/et-venv/bin/python

cmake --build cmake-out -j"$(nproc)" && cmake --install cmake-out
```

Build ExecuTorch with the **same compiler** you build this project with, so both
link against one C++ runtime.

> **`-DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON` is required, not optional.** It defaults to
> `OFF`, and without it the prefix ships only `portable_ops_lib`, which registers
> `aten::addmm.out` and `aten::mm.out` but no `aten::linear.out`. rfdetr 1.9.1 recombines
> the `addmm` ops XNNPACK leaves un-delegated back into `aten.linear` (6 such calls in an
> `RFDETRNano` export), so a `.pte` from 1.9.1 or newer fails at load on a portable-only prefix.
> The build links `optimized_native_cpu_ops_lib` when the prefix provides it — it is a
> superset of the portable set — and falls back to `portable_ops_lib` with a CMake warning
> when it does not. Exactly one op library is linked either way: each registers its kernels
> from a static initializer, and registering an op twice aborts the runtime at startup.

> **Upstream bug (v1.3.1 and earlier)** — `extension/evalue_util/CMakeLists.txt` used
> `DESTINATION ${CMAKE_BINARY_DIR}/lib` where every other extension uses
> `${CMAKE_INSTALL_LIBDIR}`, so `libextension_evalue_util.a` never reached `<prefix>/lib`
> and its exported target kept an absolute build-tree path, making
> `find_package(executorch CONFIG)` hard-fail once the build tree was deleted — even
> though this project never links that target. Fixed in v1.4.0. On an older tag, patch it
> before configuring:
> ```bash
> sed -i 's|DESTINATION ${CMAKE_BINARY_DIR}/lib|DESTINATION ${CMAKE_INSTALL_LIBDIR}|' \
>     extension/evalue_util/CMakeLists.txt
> ```

**What happens**:
- `EXECUTORCH_ROOTDIR` is added to `CMAKE_PREFIX_PATH` and resolved with `find_package(executorch CONFIG)`
- If no install prefix is found, the build falls back to compiling ExecuTorch v1.4.0 from source with the optimized kernels enabled (slow; needs a Python interpreter with ExecuTorch's build-time dependencies, since ExecuTorch runs flatbuffers codegen during its own configure)
- `-DEXECUTORCH_DELEGATE=xnnpack` (default) or `portable` selects the delegate library to link, which must match the delegate the `.pte` was exported with — a mismatch fails at run time, not at link time
- At load the backend verifies the program returns `dets` before `labels`, since ExecuTorch outputs are an unnamed tuple and postprocessing addresses them positionally

Export a model with [`deploy/export_executorch.py`](deploy/export_executorch.py); see the [export documentation](docs/export.md#executorch-model-export).

> [!NOTE]
> `deploy/export_executorch.py` covers detection models only — it offers no `RFDETRSeg*` option. The
> ExecuTorch backend itself runs segmentation `.pte` programs correctly, but the `.pte` must be
> exported by hand. See
> [docs/backend-parity-segmentation-video.md](docs/backend-parity-segmentation-video.md).

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

> [!WARNING]
> Keypoint models exported with `rfdetr` 1.8.2 or later use the active-first schema (`[17]`) and are
> not decodable by the default build, which still expects background-first `{0, 17}`. See the
> keypoint warning in [docs/export.md](docs/export.md#keypoint-model-export).

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

#### Tuning Flags

The inference parameters can be overridden without recompiling:

| Flag | Default | Effect |
|------|---------|--------|
| `--threshold <val>` | `0.5` | Confidence threshold for keeping a detection; must be in `[0, 1]` |
| `--resolution <px>` | auto-detect | Model input resolution; omit to detect it from the model |
| `--max-detections <n>` | `300` | Top-k cap on the number of query/class pairs ranked before thresholding (upstream's `num_select`) |
| `--mask-threshold <val>` | `0.0` | Mask logit cutoff for binary mask generation (segmentation only); may be negative |
| `--background-class-id <n\|none>` | `0` | Exported logit slot holding background, excluded before ranking; negative counts from the end, `none` keeps every slot |

```bash
./build/inference_app /path/to/model.onnx /path/to/image.jpg /path/to/coco-labels-91.txt \
  --threshold 0.7 --max-detections 100

./build/inference_app /path/to/model.onnx /path/to/image.jpg /path/to/coco-labels-91.txt \
  --segmentation --mask-threshold 0.5
```

These flags work with all modes (`--segmentation`, `--keypoint`, video input, and `--display`). `--resolution`
is only useful for models that accept an input size other than the one recorded in the model file — the
auto-detected value is correct for a normally exported model.

`--background-class-id` mirrors the argument rfdetr 1.9.4 added to its own ONNX/TFLite decoders. The default
`0` matches the shipped RF-DETR exports, whose logit 0 is background and whose logit *n* is COCO category *n*
— which is exactly how `data/coco-labels-91.txt` is indexed. Change it only for a checkpoint with a different
class layout: `none` for one where every logit slot is a real class (a fine-tuned model with contiguous
0-based ids), or `-1` for one whose background sits in the final slot. Getting it wrong shifts every reported
label by one.

#### How detections are selected

RF-DETR scores classes with independent sigmoids rather than a softmax, so one query can legitimately clear
the threshold on several classes at once. Postprocessing therefore ranks the flattened *(query, class)* grid
and keeps the top `--max-detections` pairs **before** applying `--threshold`, which is what
`PostProcess._select_topk` does upstream — a per-query argmax would silently drop every class but the
strongest (the bug rfdetr 1.9.3 fixed in its own exported-model decoders). Results come back in
descending-score order, with exact ties broken by ascending flattened query/class index, so a given model and
image always produce the same ordering. Detection, segmentation, keypoint, and the CUDA postprocess kernels
all share that rule.

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
server involved. Design constraints: [specs/gpu-pipeline.md](specs/gpu-pipeline.md) — remaining
phases: [specs/roadmap.md](specs/roadmap.md).

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

`Config` (`src/rfdetr_inference.hpp`) holds the inference settings. Most of them are reachable from the
command line — see [Tuning Flags](#tuning-flags) — and the rest require editing `src/main.cpp`:

| `Config` field | Default | CLI override |
|----------------|---------|--------------|
| `model_type` | `ModelType::DETECTION` | `--segmentation` / `--keypoint` |
| `threshold` | `0.5` | `--threshold <val>` |
| `resolution` | auto-detected from the model | `--resolution <px>` |
| `max_detections` | `300` (top-k selection) | `--max-detections <n>` |
| `mask_threshold` | `0.0` (binary mask generation) | `--mask-threshold <val>` |
| `background_class_id` | `0` (background-first exports) | `--background-class-id <n\|none>` |
| `gpu_preprocess` / `gpu_postprocess` | `false` | `--gpu-preprocess` / `--gpu-postprocess` |
| `dali_pipeline_dir` | `data/dali` | `--dali-pipeline-dir <dir>` |
| `gpu_device_id` | `0` | — (edit `src/main.cpp`) |
| `means` / `stds` | ImageNet `[0.485, 0.456, 0.406]` / `[0.229, 0.224, 0.225]` | — (edit `src/main.cpp`) |
| `keypoint_*`, `skeleton`, `draw_uncertainty` | COCO 17-keypoint layout | — (edit `src/main.cpp`) |

`src/main.cpp` leaves every field it does not override at its `Config` default, so changing a default in
`src/rfdetr_inference.hpp` is enough for the fields with no CLI flag.

### Example Custom Configuration

When embedding `RFDETRInference` rather than using the CLI:

```cpp
Config config;
config.resolution = 0;              // Auto-detect
config.threshold = 0.6f;            // Higher confidence threshold
config.max_detections = 100;        // Fewer detections
config.mask_threshold = 0.5f;       // More conservative masks
config.model_type = ModelType::SEGMENTATION;

RFDETRInference inference(model_path, label_path, config);
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

### Manual Cross-Backend Checks

CI compiles the TensorRT backend but has no GPU to run it on, and does not build ExecuTorch at
all, so backend agreement is verified by hand.
[docs/backend-parity-segmentation-video.md](docs/backend-parity-segmentation-video.md) records a
segmentation video run across all three backends — commands, results, and the parity comparison.

Without a matching model, integration tests that need inference are skipped, and the skip message
names the format the build expects.

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
> different runtime with `--build-arg EXECUTORCH_VERSION=<tag>`; it defaults to `v1.4.0`
> to match the exporter used by `rfdetr[executorch]==1.9.4`, and enables the optimized
> kernel set that 1.9.1+ `.pte` files need. The build applies the
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
