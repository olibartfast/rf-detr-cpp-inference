# Building

Installing the toolchain, choosing an inference backend, and every build configuration this project supports — including the ExecuTorch install prefix and the GPU pipeline.

> Part of the [RF-DETR C++ Inference](../README.md) documentation.

---

## Install Dependencies (Ubuntu)
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

## Backend Selection

Exactly one inference backend is compiled in, and enabling two is a configure-time error.
The comparison table and the model format each one accepts live in the
[README](../README.md#backend-selection); the full CMake option list is in
[Build Options](../README.md#build-options).

## Dependency Versions
Every third-party version is pinned once, in [`versions.env`](../versions.env) at the
repo root. `cmake/versions.cmake` reads it into cache variables before the dependency
declarations are loaded, and `scripts/versions.sh` reads it for the shell scripts.

Override a single pin without editing the file — CMake takes `-D`, the scripts take
the environment:

```bash
cmake -S . -B build -DONNX_RUNTIME_VERSION=1.22.0
TRITON_IMAGE=nvcr.io/nvidia/tritonserver:26.01-py3 ./scripts/fetch_dali.sh
```

To bump a version for real, edit `versions.env` and then run:

```bash
./scripts/check_version_sync.sh
```

That reports the few places which cannot read the file — the Dockerfile's `ARG`
defaults, `conanfile.txt`, `deploy/requirements.txt` and the `deploy/export_*.py`
opset defaults — and fails until they match. CI runs it as the `Version Sync` job.

## Dependency Resolution
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

Architecture details: [package-manager-architecture.md](package-manager-architecture.md)

## Build with ONNX Runtime (Default)
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

## Build with TensorRT Backend
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

## Build with ExecuTorch Backend
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

### Building the ExecuTorch install prefix

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

Export a model with [`deploy/export_executorch.py`](../deploy/export_executorch.py); see the [export documentation](export.md#executorch-model-export).

> [!NOTE]
> `deploy/export_executorch.py` covers detection models only — it offers no `RFDETRSeg*` option. The
> ExecuTorch backend itself runs segmentation `.pte` programs correctly, but the `.pte` must be
> exported by hand. See
> [backend-parity-segmentation-video.md](backend-parity-segmentation-video.md).

## Build with the GPU Pipeline (TensorRT + DALI + CUDA)
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
`nvcc`). `-DUSE_GPU_PIPELINE=ON` turns on both. See [GPU Pipeline](architecture.md#gpu-pipeline) for how it works, and
[Usage](usage.md#gpu-pipeline-flags-tensorrt-builds-with-the-gpu-pipeline-compiled-in)
for the runtime flags.

## Build with OpenCV Media/Display Backend
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
