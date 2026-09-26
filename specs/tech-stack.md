# Tech Stack

**Every third-party version is pinned once, in [`versions.env`](../versions.env).** The tables
below name the `versions.env` variable, never its value, so a bump needs no edit here. The
"Pin location" columns name the *consumer* that reads the value, not a second place to edit.

Two loaders read that file:

| Loader | Consumers |
|--------|-----------|
| `cmake/versions.cmake` | Included by `CMakeLists.txt` before `cmake/deps/Deps.cmake`, so every `cmake/deps/packages/*.cmake` interpolates the values. Each pin is a `CACHE STRING`, so `-DTENSORRT_VERSION=…` overrides it. Editing `versions.env` in an **existing** build tree updates the cache on reconfigure — an INTERNAL stamp per pin distinguishes "still the file default" from "overridden with `-D`", so no build directory wipe is needed and no `-D` is clobbered |
| `scripts/versions.sh` | `source`d by `scripts/fetch_dali.sh`, `scripts/generate_dali_pipelines.sh`, `scripts/ci/stage_gpu_headers.sh`, `scripts/run_gate.sh` (for the `environment.txt` provenance record), `export_trt.sh`, and `gpu-compile.yml` (to derive the CUDA apt coordinates). Never clobbers a value already in the environment, so `TRITON_IMAGE=… ./scripts/fetch_dali.sh` still works |

Five formats cannot read a file — the backend Dockerfiles' `ARG` defaults, `conanfile.txt`,
`deploy/requirements.txt`, the argparse defaults in `deploy/export_*.py`, and the two version
tables in `README.md` (the only prose allowed to state a pinned value). They restate the values,
and `scripts/check_version_sync.sh` (the `Version Sync` job in `lint.yml`) fails the build when a
restatement drifts. Run it after editing `versions.env`.

| Layer | Choice | Version | Notes |
|-------|--------|---------|-------|
| Language | C++20 | — | `CMakeLists.txt:5`; CUDA C++20 for `.cu` (`:170`) |
| Build | CMake | ≥ 3.12 | `CMakeLists.txt:1`; **3.17+** if ExecuTorch falls back to the source build |
| Compiler | Clang 15+ / GCC 12+ | — | CI and Docker use clang-18 |
| Generator | Ninja | — | Optional but assumed by every documented command |
| Testing | GoogleTest | `GTEST_VERSION` | `versions.env`; used by `GTest.cmake`, `conanfile.txt` |
| Benchmarks | Google Benchmark | `GOOGLE_BENCHMARK_VERSION` | `versions.env`; opt-in `-DBENCHMARKS=ON`; covers preprocessing only |
| Dependencies | apt / conan / vcpkg facade | — | `find_dependency_unified()`, `DEPS_MODE` default `apt` (`cmake/deps/Deps.cmake:6`) |
| Format | clang-format | 18 | `.clang-format`: LLVM base, indent 4, column 120 |
| Static analysis | clang-tidy 18, cppcheck | — | `.clang-tidy`; CI excludes `tensorrt_backend.cpp` from clang-tidy |
| Export tooling | `rfdetr[onnx]` | `RFDETR_VERSION` | `versions.env`, mirrored into `deploy/requirements.txt`; ONNX opset `ONNX_OPSET_VERSION` |
| Vendored | stb, font8x8 | unversioned | `third_party/` — no install step |

## Inference backends

Exactly one is compiled in.

| Backend | Version | Pin location | Device | Model format |
|---------|---------|--------------|--------|--------------|
| ONNX Runtime (default) | `ONNX_RUNTIME_VERSION` | `versions.env` | **CPU only** | `.onnx` |
| TensorRT | `TENSORRT_VERSION` (11.x only) | `versions.env` | NVIDIA GPU | `.engine`, `.trt`, `.onnx` |
| ExecuTorch | `EXECUTORCH_VERSION` | `versions.env` | CPU (XNNPACK or portable) | `.pte` |

`TENSORRT_VERSION` is the full four-component number. NVIDIA truncates it differently per
artefact, so the loaders *derive* the rest rather than pinning them separately:

| Derived | Form | Derived by | Used for |
|---------|------|-----------|----------|
| `TENSORRT_SHORT_VERSION` | `<major>.<minor>.<patch>` | both loaders | download-URL directory, Conan recipe |
| `TENSORRT_DEB_VERSION` | `<TENSORRT_VERSION>-1+cuda<CUDA_VERSION>` | `versions.sh` only | apt packages CI stages headers from |
| `TRITON_IMAGE`, `TENSORRT_IMAGE` | `nvcr.io/nvidia/{tritonserver,tensorrt}:<NGC_CONTAINER_TAG>-py3` | `versions.sh` only | container-based staging and export |

The TensorRT download name is derived too, by `cmake/deps/packages/TensorRT.cmake`:
`TensorRT-Enterprise-<v>-Linux-x86_64-cuda-<cuda>-Release-external.tar.zst` ("Enterprise" is
NVIDIA's name for standard TensorRT from 11.x, under the same license).

- **ONNX Runtime** downloads the official CPU archive selected from the *target* platform (`CMAKE_SYSTEM_NAME` / `CMAKE_SYSTEM_PROCESSOR`), covering Linux x64/aarch64 and Windows x64/arm64. Other targets require a compatible provided prefix or package-manager build; catalog loading permits them when ONNX Runtime is disabled. It registers no execution provider, so even a CUDA build runs on CPU.
- **TensorRT** implies the CUDA Toolkit series `CUDA_VERSION` pins, which must be installed manually. The backend supports TensorRT 11 only (`NV_TENSORRT_MAJOR < 11` is a compile error, `TENSORRT_VERSION < 11` a configure error). TensorRT 11 is strongly typed only: an `.onnx` build takes the model's own precision (FP16 needs a converted ONNX). Engine I/O must be float32.
- **ExecuTorch** requires a prefix built with `EXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON` (it defaults `OFF`): `.pte` files from rfdetr 1.9.1+ call `aten::linear.out`, registered only by `optimized_native_cpu_ops_lib`. The linked delegate must match the one baked into the `.pte`.

## Media and GPU stack

| Layer | Choice | Version | Notes |
|-------|--------|---------|-------|
| Media (default) | FFmpeg + SDL2 + stb | unpinned in CMake | pkg-config; conan pins `FFMPEG_VERSION`, `SDL_VERSION` |
| Media (alternative) | OpenCV 4.x | unpinned | `-DUSE_OPENCV=ON`; replaces FFmpeg, SDL2 **and** stb. `OPENCV_VERSION` tracks only the commented swap instruction in `conanfile.txt`, not a live pin |
| GPU preprocessing | NVIDIA DALI | `DALI_VERSION` (2.x only; CI header staging) | Staged from `TRITON_IMAGE` (`NGC_CONTAINER_TAG`) via `scripts/fetch_dali.sh` — NVIDIA ships no standalone C++ distribution |
| GPU postprocessing | CUDA Toolkit + CUB | `CUDA_VERSION` | `FindCUDAToolkit`; `CMAKE_CUDA_ARCHITECTURES` defaults to `CUDA_ARCHITECTURES` |

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

`CMakePresets.json` provides six presets: `default`, `debug-sanitizers`, `debug-tsan`, `debug-strict-ubsan`, `debug-valgrind`, and `gpu-pipeline` (TensorRT + DALI + CUDA). ExecuTorch and OpenCV require explicit options.

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
| `gpu-compile.yml` — GPU Backend Compile | Compile-only matrix, `-DWERROR=ON`: TensorRT alone, +DALI, +CUDA postprocess, +both, all against the pinned `TENSORRT_VERSION`/`DALI_VERSION` headers. Builds `rfdetr_inference_lib` only — the staged shared objects are stubs, so no target that links is reachable |
| `deps-modes.yml` — Dependency Modes | `workflow_dispatch` only; matrix over apt / conan / vcpkg |

All three push/PR workflows (`ci.yml`, `lint.yml`, `gpu-compile.yml`) trigger on `master` and `develop`. Integration tests are not run by CI.

## Bumping a version

1. Edit the one line in `versions.env`.
2. Run `./scripts/check_version_sync.sh`. It reports the restatements that must follow — the
   backend Dockerfile `ARG` defaults, `conanfile.txt`, `deploy/requirements.txt`, the `deploy/export_*.py`
   opset, and the README version tables — and fails until they match.
3. Nothing else: `docs/`, `specs/` and the rest of the README name the variable, not the value.
   Only prose that describes *behaviour* tied to a version (a removed API, a changed default)
   needs reading when the major changes.

## Known pin duplications

- The README version tables restate `versions.env` for readers at a glance; `AGENTS.md`
  requires them, and `check_version_sync.sh` verifies them. No other prose states a pinned value.
- `project()` declares `VERSION 0.5.1`; `vcpkg.json` and the README badge agree. This is a *project*
  version, not a dependency pin, so `versions.env` does not cover it.
- `dockerfile.trt` forwards `--build-arg TENSORRT_VERSION` to CMake as `-DTENSORRT_VERSION`, because the TensorRT shim directory it creates must match what CMake looks for. Any future build arg that names a pin needs the same forwarding.
- `scripts/run_gate.sh` defaults `CUDA_ARCH=89` rather than the build default
  `CUDA_ARCHITECTURES=86`. Deliberate, and not a pin: the value is a property of whichever
  card the gate runs on, so it stays out of `versions.env`. `specs/rented-gpu-runbook.md`
  carries the card-to-arch table and expects it to be set per run.
