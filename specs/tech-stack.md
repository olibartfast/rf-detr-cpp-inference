# Tech Stack

**Every third-party version is pinned once, in [`versions.env`](../versions.env).** Change it
there, then update this table. The "Pin location" columns below name the *consumer* that reads
the value, not a second place to edit.

Two loaders read that file:

| Loader | Consumers |
|--------|-----------|
| `cmake/versions.cmake` | Included by `CMakeLists.txt` before `cmake/deps/Deps.cmake`, so every `cmake/deps/packages/*.cmake` interpolates the values. Each pin is a `CACHE STRING`, so `-DTENSORRT_VERSION=…` overrides it |
| `scripts/versions.sh` | `source`d by `scripts/fetch_dali.sh`, `scripts/generate_dali_pipelines.sh`, `scripts/ci/stage_gpu_headers.sh`, `export_trt.sh`. Never clobbers a value already in the environment, so `TRITON_IMAGE=… ./scripts/fetch_dali.sh` still works |

Four formats cannot read a file — the Dockerfile's `ARG` defaults, `conanfile.txt`,
`deploy/requirements.txt`, and the argparse defaults in `deploy/export_*.py`. They restate the
values, and `scripts/check_version_sync.sh` (the `Version Sync` job in `lint.yml`) fails the
build when a restatement drifts. Run it after editing `versions.env`.

| Layer | Choice | Version | Notes |
|-------|--------|---------|-------|
| Language | C++20 | — | `CMakeLists.txt:5`; CUDA C++20 for `.cu` (`:170`) |
| Build | CMake | ≥ 3.12 | `CMakeLists.txt:1`; **3.17+** if ExecuTorch falls back to the source build |
| Compiler | Clang 15+ / GCC 12+ | — | CI and Docker use clang-18 |
| Generator | Ninja | — | Optional but assumed by every documented command |
| Testing | GoogleTest | 1.12.1 | `versions.env` → `GTEST_VERSION`; used by `GTest.cmake`, `conanfile.txt` |
| Benchmarks | Google Benchmark | 1.9.1 | `versions.env` → `GOOGLE_BENCHMARK_VERSION`; opt-in `-DBENCHMARKS=ON`; covers preprocessing only |
| Dependencies | apt / conan / vcpkg facade | — | `find_dependency_unified()`, `DEPS_MODE` default `apt` (`cmake/deps/Deps.cmake:6`) |
| Format | clang-format | 18 | `.clang-format`: LLVM base, indent 4, column 120 |
| Static analysis | clang-tidy 18, cppcheck | — | `.clang-tidy`; CI excludes `tensorrt_backend.cpp` from clang-tidy |
| Export tooling | `rfdetr[onnx]` | 1.9.4 | `versions.env` → `RFDETR_VERSION`, mirrored into `deploy/requirements.txt`; ONNX opset 17 (`ONNX_OPSET_VERSION`) |
| Vendored | stb, font8x8 | unversioned | `third_party/` — no install step |

## Inference backends

Exactly one is compiled in.

| Backend | Version | Pin location | Device | Model format |
|---------|---------|--------------|--------|--------------|
| ONNX Runtime (default) | 1.21.0 | `versions.env` → `ONNX_RUNTIME_VERSION` | **CPU only** | `.onnx` |
| TensorRT | 10.13.3.9 | `versions.env` → `TENSORRT_VERSION` | NVIDIA GPU | `.engine`, `.trt`, `.onnx` |
| ExecuTorch | v1.4.0 | `versions.env` → `EXECUTORCH_VERSION` | CPU (XNNPACK or portable) | `.pte` |

`TENSORRT_VERSION` is the full four-component number. NVIDIA truncates it differently per
artefact, so the loaders *derive* the rest rather than pinning them separately:

| Derived | Value | Derived by | Used for |
|---------|-------|-----------|----------|
| `TENSORRT_SHORT_VERSION` | `10.13.3` | both loaders | download-URL directory, Conan recipe |
| `TENSORRT_DEB_VERSION` | `10.13.3.9-1+cuda13.0` | `versions.sh` only | apt packages CI stages headers from |
| `TRITON_IMAGE`, `TENSORRT_IMAGE` | `…:25.12-py3` | `versions.sh` only | container-based staging and export |

- **ONNX Runtime** downloads the official CPU archive selected from the *target* platform (`CMAKE_SYSTEM_NAME` / `CMAKE_SYSTEM_PROCESSOR`), covering Linux x64/aarch64 and Windows x64/arm64. Anything else is a configure-time `FATAL_ERROR`. It registers no execution provider, so even a CUDA build runs on CPU.
- **TensorRT** implies CUDA Toolkit **13.x**, which must be installed manually.
- **ExecuTorch** requires a prefix built with `EXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON` (it defaults `OFF`): `.pte` files from rfdetr 1.9.1+ call `aten::linear.out`, registered only by `optimized_native_cpu_ops_lib`. The linked delegate must match the one baked into the `.pte`.

## Media and GPU stack

| Layer | Choice | Version | Notes |
|-------|--------|---------|-------|
| Media (default) | FFmpeg + SDL2 + stb | unpinned in CMake | pkg-config; conan pins `ffmpeg/6.1`, `sdl/2.28.5` (`FFMPEG_VERSION`, `SDL_VERSION`) |
| Media (alternative) | OpenCV 4.x | unpinned | `-DUSE_OPENCV=ON`; replaces FFmpeg, SDL2 **and** stb. `OPENCV_VERSION` (4.8.1) tracks only the commented swap instruction in `conanfile.txt`, not a live pin |
| GPU preprocessing | NVIDIA DALI | 1.51.2 (`DALI_VERSION`, CI header staging only) | Staged from `nvcr.io/nvidia/tritonserver:25.12-py3` (`NGC_CONTAINER_TAG`) via `scripts/fetch_dali.sh` — NVIDIA ships no standalone C++ distribution |
| GPU postprocessing | CUDA Toolkit + CUB | 13.x (`CUDA_VERSION`) | `FindCUDAToolkit`; `CMAKE_CUDA_ARCHITECTURES` defaults to `CUDA_ARCHITECTURES` (`86`) |

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
| `RFDETR_VERSIONS_ENV` | `<repo>/versions.env` | `cmake/versions.cmake` |

`CMakePresets.json` provides five presets: `default`, `debug-sanitizers`, `debug-tsan`, `debug-strict-ubsan`, `debug-valgrind`. **None** covers TensorRT, ExecuTorch, OpenCV, or the GPU pipeline.

## Constraints

These are enforced at configure time or by the runtime — not style preferences.

- Exactly one of `USE_ONNX_RUNTIME` / `USE_TENSORRT` / `USE_EXECUTORCH`. Two is a `FATAL_ERROR`.
- The three sanitizer modes are mutually exclusive. Valgrind needs a plain Debug build — ASan and TSan conflict with it.
- `USE_DALI` and `USE_CUDA_POSTPROCESS` require the TensorRT backend. Either with ONNX Runtime is a `FATAL_ERROR`.
- `--gpu-postprocess` additionally requires `--segmentation`.
- **CI runners have no GPU.** `gpu-compile.yml` *compiles* the TensorRT, DALI and CUDA paths under `-DWERROR=ON` against headers-only prefixes, but nothing links and nothing runs there; ExecuTorch is not built by CI at all. Behaviour — parity, sanitizers, benchmarks — you **must** verify manually: see [AGENTS.md](../AGENTS.md).

## CI coverage

| Workflow | Jobs |
|----------|------|
| `ci.yml` — Build & Test | Build & Unit Tests (+ benchmarks), Sanitizers (ASan+UBSan), ThreadSanitizer, Valgrind Memcheck |
| `lint.yml` — C++ Lint & Build | Version Sync (`scripts/check_version_sync.sh`), Format Check, Clang-Tidy, Cppcheck, Build with Strict Warnings (`-DWERROR=ON`) |
| `gpu-compile.yml` — GPU Backend Compile | Compile-only matrix, `-DWERROR=ON`: TensorRT alone, +DALI, +CUDA postprocess, +both. Builds `rfdetr_inference_lib` only — the staged shared objects are stubs, so no target that links is reachable |
| `deps-modes.yml` — Dependency Modes | `workflow_dispatch` only; matrix over apt / conan / vcpkg |

Both push/PR workflows trigger on `master` and `develop`. Integration tests are not run by CI.

## Bumping a version

1. Edit the one line in `versions.env`.
2. Run `./scripts/check_version_sync.sh`. It reports the restatements that must follow — the
   Dockerfile `ARG` default, `conanfile.txt`, `deploy/requirements.txt`, the `deploy/export_*.py`
   opset — and fails until they match.
3. Reconcile the prose in `README.md` and `docs/` (the `Spec Sync` rule in `AGENTS.md` requires
   the README to *state* the versions; that text is not machine-checked).
4. Update the tables above.

## Known pin duplications

- Prose version statements in `README.md`, `docs/building.md`, `docs/docker.md`,
  `docs/architecture.md` and `docs/package-manager-architecture.md` restate `versions.env` for
  readers. `AGENTS.md` requires them, and nothing verifies them — step 3 above is manual.
- `project()` declares **no version**. `vcpkg.json` says `0.1.0`; the README badge says `0.4.0`. They disagree. This is a *project* version, not a dependency pin, so `versions.env` does not cover it.
- `scripts/run_gate.sh` defaults `CUDA_ARCH=89` rather than the build default
  `CUDA_ARCHITECTURES=86`. Deliberate, and not a pin: the value is a property of whichever
  card the gate runs on, so it stays out of `versions.env`. `docs/rented-gpu-runbook.md`
  carries the card-to-arch table and expects it to be set per run.
