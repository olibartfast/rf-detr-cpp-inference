# Tech Stack

Every version below cites the file that owns the pin. Change it there, not here — then update this table.

| Layer | Choice | Version | Notes |
|-------|--------|---------|-------|
| Language | C++20 | — | `CMakeLists.txt:5`; CUDA C++20 for `.cu` (`:170`) |
| Build | CMake | ≥ 3.12 | `CMakeLists.txt:1`; **3.17+** if ExecuTorch falls back to the source build |
| Compiler | Clang 15+ / GCC 12+ | — | CI and Docker use clang-18 |
| Generator | Ninja | — | Optional but assumed by every documented command |
| Testing | GoogleTest | 1.12.1 | `cmake/deps/packages/GTest.cmake:12` |
| Benchmarks | Google Benchmark | 1.9.1 | `GoogleBenchmark.cmake:12`; opt-in `-DBENCHMARKS=ON`; covers preprocessing only |
| Dependencies | apt / conan / vcpkg facade | — | `find_dependency_unified()`, `DEPS_MODE` default `apt` (`cmake/deps/Deps.cmake:6`) |
| Format | clang-format | 18 | `.clang-format`: LLVM base, indent 4, column 120 |
| Static analysis | clang-tidy 18, cppcheck | — | `.clang-tidy`; CI excludes `tensorrt_backend.cpp` from clang-tidy |
| Export tooling | `rfdetr[onnx]` | 1.9.2 | `deploy/requirements.txt` — the only pinned pip requirement; ONNX opset 17 |
| Vendored | stb, font8x8 | unversioned | `third_party/` — no install step |

## Inference backends

Exactly one is compiled in.

| Backend | Version | Pin location | Device | Model format |
|---------|---------|--------------|--------|--------------|
| ONNX Runtime (default) | 1.21.0 | `cmake/deps/packages/OnnxRuntime.cmake:1` | **CPU only** | `.onnx` |
| TensorRT | 10.13.3.9 | `cmake/deps/packages/TensorRT.cmake:9` | NVIDIA GPU | `.engine`, `.trt`, `.onnx` |
| ExecuTorch | v1.4.0 | `cmake/deps/packages/ExecuTorch.cmake:29` | CPU (XNNPACK or portable) | `.pte` |

- **ONNX Runtime** downloads the official CPU archive selected from the *target* platform (`CMAKE_SYSTEM_NAME` / `CMAKE_SYSTEM_PROCESSOR`), covering Linux x64/aarch64 and Windows x64/arm64. Anything else is a configure-time `FATAL_ERROR`. It registers no execution provider, so even a CUDA build runs on CPU.
- **TensorRT** implies CUDA Toolkit **13.x**, which must be installed manually.
- **ExecuTorch** requires a prefix built with `EXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON` (it defaults `OFF`): `.pte` files from rfdetr 1.9.1+ call `aten::linear.out`, registered only by `optimized_native_cpu_ops_lib`. The linked delegate must match the one baked into the `.pte`.

## Media and GPU stack

| Layer | Choice | Version | Notes |
|-------|--------|---------|-------|
| Media (default) | FFmpeg + SDL2 + stb | unpinned in CMake | pkg-config; conan pins `ffmpeg/6.1`, `sdl/2.28.5` |
| Media (alternative) | OpenCV 4.x | unpinned | `-DUSE_OPENCV=ON`; replaces FFmpeg, SDL2 **and** stb |
| GPU preprocessing | NVIDIA DALI | unpinned in CMake | Staged from `nvcr.io/nvidia/tritonserver:25.12-py3` via `scripts/fetch_dali.sh` — NVIDIA ships no standalone C++ distribution |
| GPU postprocessing | CUDA Toolkit + CUB | — | `FindCUDAToolkit`; `CMAKE_CUDA_ARCHITECTURES` default `86` |

Only resolutions **432** and **576** have checked-in `.dali` pipelines (`data/dali/`). Others must be regenerated with `./scripts/generate_dali_pipelines.sh <res>`.

## CMake options

| Option | Default | Line |
|--------|---------|------|
| `USE_ONNX_RUNTIME` | **ON** | `CMakeLists.txt:76` |
| `USE_TENSORRT` | OFF | `:77` |
| `USE_EXECUTORCH` | OFF | `:78` |
| `USE_OPENCV` | OFF | `:79` |
| `EXECUTORCH_DELEGATE` | `xnnpack` (or `portable`) | `:86` |
| `USE_DALI` | OFF | `:118` |
| `USE_CUDA_POSTPROCESS` | OFF | `:119` |
| `USE_GPU_PIPELINE` | OFF (enables both above) | `:120` |
| `WERROR` | OFF | `:28` |
| `SANITIZERS` (ASan+UBSan) | OFF | `:38` |
| `STRICT_UBSAN` | OFF | `:39` |
| `THREAD_SANITIZER` | OFF | `:40` |
| `BENCHMARKS` | OFF | `:419` |
| `DEPS_MODE` | `apt` (`apt\|conan\|vcpkg\|auto`) | `cmake/deps/Deps.cmake:6` |
| `DEPS_DEBUG` | OFF | `cmake/deps/Deps.cmake` |

`CMakePresets.json` provides five presets: `default`, `debug-sanitizers`, `debug-tsan`, `debug-strict-ubsan`, `debug-valgrind`. **None** covers TensorRT, ExecuTorch, OpenCV, or the GPU pipeline.

## Constraints

These are enforced at configure time or by the runtime — not style preferences.

- Exactly one of `USE_ONNX_RUNTIME` / `USE_TENSORRT` / `USE_EXECUTORCH`. Two is a `FATAL_ERROR`.
- The three sanitizer modes are mutually exclusive. Valgrind needs a plain Debug build — ASan and TSan conflict with it.
- `USE_DALI` and `USE_CUDA_POSTPROCESS` require the TensorRT backend. Either with ONNX Runtime is a `FATAL_ERROR`.
- `--gpu-postprocess` additionally requires `--segmentation`.
- **CI runners are all `ubuntu-latest` with no GPU.** TensorRT, ExecuTorch, DALI, and CUDA paths are never built or run by CI. You **must** verify those manually — see [AGENTS.md](../AGENTS.md).

## CI coverage

| Workflow | Jobs |
|----------|------|
| `ci.yml` — Build & Test | Build & Unit Tests (+ benchmarks), Sanitizers (ASan+UBSan), ThreadSanitizer, Valgrind Memcheck |
| `lint.yml` — C++ Lint & Build | Format Check, Clang-Tidy, Cppcheck, Build with Strict Warnings (`-DWERROR=ON`) |
| `deps-modes.yml` — Dependency Modes | `workflow_dispatch` only; matrix over apt / conan / vcpkg |

Both push/PR workflows trigger on `master` and `develop`. Integration tests are not run by CI.

## Known pin duplications

Update both sides together:

- TensorRT `10.13.3.9` is pinned in `cmake/deps/packages/TensorRT.cmake:9` **and** hardcoded as a path in `Dockerfile:148-150`.
- The Triton container tag `25.12-py3` appears in `scripts/fetch_dali.sh:16`, `scripts/generate_dali_pipelines.sh:15`, and `export_trt.sh`.
- `project()` declares **no version**. `vcpkg.json` says `0.1.0`; the README badge says `0.4.0`. They disagree.
