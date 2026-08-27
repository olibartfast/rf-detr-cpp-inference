# AGENTS.md

## Project Specs
Read these before starting work; this file covers commands, the specs cover intent.
- [specs/mission.md](specs/mission.md) — what this project is and the architectural commitments not to break
- [specs/tech-stack.md](specs/tech-stack.md) — pinned versions and where each pin lives, CMake options, CI coverage
- [specs/roadmap.md](specs/roadmap.md) — phased work queue and deferred items
- [specs/gpu-pipeline.md](specs/gpu-pipeline.md) — GPU design constraints; the 8-rule model contract is the review checklist for any change to `src/gpu/`
- [specs/features/](specs/features/) — one directory per phase of work: `requirements.md`, `plan.md`, `validation.md`

## Workflow
The loop: pick the next unticked phase in `specs/roadmap.md` → write its spec → implement from `plan.md` → pass `validation.md` → update `CHANGELOG.md` → merge to `develop` → tick the phase.

A spec directory under `specs/features/YYYY-MM-DD-<name>/` is **required** for a roadmap phase, a release, an upstream `rfdetr` alignment, or any change touching a path CI cannot execute (`src/gpu/`, `src/backends/tensorrt_backend.cpp`, `src/backends/executorch_backend.cpp`, `deploy/export_executorch.py`). It is **not** required for bug fixes, docs, or dependency bumps with no contract change — those go in `CHANGELOG.md` only. "CHANGELOG only" is about specs, not a mandate to log everything: `CHANGELOG.md` covers this C++ project's own features and fixes and the upstream `rfdetr` releases it tracks, so changes confined to agent tooling (`specs/`, `.claude/`, `.opencode/`, `AGENTS.md` and its pointer files) are recorded in the commit message alone.

Four workflows are written down as skills. Each is a plain markdown checklist, usable by hand with any agent:

| Skill | Use when |
|-------|----------|
| [feature-spec](.claude/skills/feature-spec/SKILL.md) | Starting a roadmap phase — find it, branch, interview, write the spec triple |
| [rfdetr-alignment](.claude/skills/rfdetr-alignment/SKILL.md) | An upstream `rfdetr` release lands — the standing obligation |
| [release](.claude/skills/release/SKILL.md) | Cutting a git-flow release |
| [gpu-verify](.claude/skills/gpu-verify/SKILL.md) | Verifying TensorRT/DALI/CUDA — the gate CI cannot run |

Git-flow: branch from `develop`, never from `master`.

## Backend Selection
Exactly one backend is compiled in; enabling two is a configure-time error.
- ONNX Runtime (default): 
  `cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build build --parallel`
- TensorRT:
  `cmake -S . -B build -G Ninja -DUSE_ONNX_RUNTIME=OFF -DUSE_TENSORRT=ON -DCMAKE_BUILD_TYPE=Release && cmake --build build --parallel`
- ExecuTorch (`.pte` models, rfdetr 1.9.0+):
  `cmake -S . -B build -G Ninja -DUSE_ONNX_RUNTIME=OFF -DUSE_EXECUTORCH=ON -DEXECUTORCH_ROOTDIR=<prefix> -DCMAKE_BUILD_TYPE=Release && cmake --build build --parallel`
  - `-DEXECUTORCH_DELEGATE=xnnpack|portable` (default `xnnpack`) must match the delegate the `.pte` was exported with.
  - Without `EXECUTORCH_ROOTDIR` the build falls back to compiling ExecuTorch v1.4.0 from source, which is slow and needs a Python interpreter with ExecuTorch's build deps (`import torchgen`, i.e. the `torch` wheel — a bare `python3` fails).
  - The prefix must be built with `-DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON` (defaults to `OFF`): `.pte` files from rfdetr 1.9.1+ call `aten::linear.out`, which only `optimized_native_cpu_ops_lib` registers. The build links that lib when present and warns + falls back to `portable_ops_lib` when not — exactly one op library, since duplicate kernel registration aborts at startup. See README "Building the ExecuTorch install prefix".
  - The `extension/evalue_util` install-path patch is only needed on v1.3.1 and older; v1.4.0 fixed it upstream.

## Docker
`Dockerfile` builds an inference-backend × media-backend matrix:
`--build-arg INFERENCE_BACKEND=onnx|tensorrt|executorch` and `--build-arg MEDIA_BACKEND=ffmpeg|opencv`.
The `executorch` variant builds the ExecuTorch runtime from source into `/opt/executorch` (override the tag with `--build-arg EXECUTORCH_VERSION=<tag>`) and applies the upstream install fix automatically.

## GPU Pipeline (TensorRT only)
- Stage DALI first (one-time, extracts from pinned Triton container): `./scripts/fetch_dali.sh` → `~/dependencies/dali`
- Build both halves:
  `cmake -S . -B build -G Ninja -DUSE_ONNX_RUNTIME=OFF -DUSE_TENSORRT=ON -DUSE_GPU_PIPELINE=ON -DDALI_ROOT=$HOME/dependencies/dali -DCMAKE_BUILD_TYPE=Release && cmake --build build --parallel`
- Halves are independent: `-DUSE_DALI=ON` (DALI preprocessing, no nvcc) / `-DUSE_CUDA_POSTPROCESS=ON` (CUDA seg postprocessing, needs nvcc; `CMAKE_CUDA_ARCHITECTURES` default `86`)
- Either option with the ONNX Runtime backend is a configure-time `FATAL_ERROR`
- Runtime flags (default off): `--gpu-preprocess`, `--gpu-postprocess` (segmentation only), `--dali-pipeline-dir <dir>` (default `data/dali`)
- Regenerate `.dali` pipelines for a new resolution: `./scripts/generate_dali_pipelines.sh <res>` (needs `--gpus all` Docker); 432 and 576 are checked in
- GPU unit tests (`test_gpu_postprocess.cpp`) `GTEST_SKIP()` without a CUDA device; like TensorRT, CI compiles but does not execute GPU paths — `gpu-compile.yml` builds all four `USE_DALI`/`USE_CUDA_POSTPROCESS` combinations with `-DWERROR=ON` against headers staged by `scripts/ci/stage_gpu_headers.sh`, so a compile break is a red PR, not a surprise on metered hardware. Behaviour still has to be tested manually with [gpu-verify](.claude/skills/gpu-verify/SKILL.md)
- On a rented GPU box, `./scripts/run_gate.sh` drives the executable part of that checklist unattended and reports the rest as `UNRUN`; it arms a deadline watchdog and stops the instance when done. Env knobs: `CUDA_ARCH` (default `89`), `DEADLINE_HOURS`, `SKIP_DEFAULT_PATH`, `SELF_STOP`, `MODEL`, `VIDEO`. End-to-end procedure — choosing an instance, export prep, setup script, collecting results: [docs/rented-gpu-runbook.md](docs/rented-gpu-runbook.md)
- Design constraints: [specs/gpu-pipeline.md](specs/gpu-pipeline.md) — remaining phases: [specs/roadmap.md](specs/roadmap.md)

## Dependency Resolution
- Default (`-DDEPS_MODE=apt`): system packages + pinned downloads — no extra tooling
- Conan/vcpkg: auto-activate via toolchain; see [docs/package-manager-architecture.md](docs/package-manager-architecture.md)
- `-DDEPS_DEBUG=ON` logs which handler resolved each dependency

## Code Quality
- Format check: `find src tests -name '*.cpp' -o -name '*.hpp' | xargs clang-format-18 --dry-run --Werror`
- Format apply: `find src tests -name '*.cpp' -o -name '*.hpp' | xargs clang-format-18 -i`
- Clang-tidy: 
  `cmake -S . -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON`
  `find src -name '*.cpp' | xargs clang-tidy-18 -p build`
- Cppcheck: `cppcheck --enable=all --std=c++20 --suppress=missingIncludeSystem --suppress=unmatchedSuppression --suppress=unusedFunction --error-exitcode=1 -I src src/`
- Strict warnings (CI): `-DWERROR=ON` at configure time

## Spec Sync
- Mandatory: `git fetch` and check the branch against its upstream **before starting any work** — not just before pushing. A stale local branch hides the rules you are supposed to follow: on 2026-08-24 an alignment pass was written against a `develop` four commits behind `origin/develop`, missing an already-completed 1.9.2 alignment, the `specs/` tree, and the `rfdetr-alignment` skill that governs the task. If the branch is behind, integrate or at least read what landed, then re-read `AGENTS.md`, `.claude/skills/`, and `specs/` before planning.
- Mandatory: before acting on any release, version-alignment, or dependency-sync request, read `AGENTS.md`, `README.md`, and `CHANGELOG.md`, then verify the named release against the official upstream project. Never assume that an upstream version is a local Git tag or infer the required scope from the version string alone; inspect the repository documentation and upstream release notes/diff first.
- A change to `specs/mission.md` or `specs/tech-stack.md` must propagate in the **same commit** to `README.md`, `AGENTS.md`, and any open spec under `specs/features/`. The constitution and what it describes never diverge across commits.
- Mandatory for every release or dependency-facing patch: update `README.md` in the same change when code, build options, backend versions, Docker images, or Python export packages change.
- Verify README dependency/version statements against `CMakeLists.txt`, `CMakePresets.json`, `deploy/requirements.txt`, `Dockerfile*`, and `docs/export.md`.
- README must list current C++ library/runtime versions, CMake options, backend constraints, and pip packages used for export tooling.
- If a release intentionally needs no README change, say why in `CHANGELOG.md` or the PR/release notes.

## Testing
- Unit tests: `ctest --test-dir build --output-on-failure -R UnitTests`
- Integration tests: `ctest --test-dir build --output-on-failure -R IntegrationTests`
- All tests: `cmake --build build --target run_tests`
- Benchmarks (if enabled): `./build/benchmarks`

## Sanitizers
ASan+UBSan, strict UBSan, and TSan are mutually exclusive (pick one).

### AddressSanitizer + UndefinedBehaviorSanitizer
- Configure: `cmake -S . -B build-san -DCMAKE_BUILD_TYPE=Debug -DSANITIZERS=ON`
- Build: `cmake --build build-san --parallel`
- Run unit tests: `./build-san/unit_tests`
- Run integration tests: `./build-san/integration_tests`

### Strict UndefinedBehaviorSanitizer
- Configure: `cmake -S . -B build-strict-ubsan -DCMAKE_BUILD_TYPE=Debug -DSTRICT_UBSAN=ON`
- Build: `cmake --build build-strict-ubsan --parallel`
- Run unit tests: `./build-strict-ubsan/unit_tests`
- Run integration tests: `./build-strict-ubsan/integration_tests`

### ThreadSanitizer (data races)
- Configure: `cmake -S . -B build-tsan -DCMAKE_BUILD_TYPE=Debug -DTHREAD_SANITIZER=ON`
- Build: `cmake --build build-tsan --parallel`
- Run: `TSAN_OPTIONS="halt_on_error=1" ./build-tsan/unit_tests`

## Valgrind / Profiling
Requires a plain Debug build (no sanitizers — ASan/TSan conflict with Valgrind). The `memcheck`, `callgrind`, and `massif` CMake targets are auto-generated when Valgrind is found.
- Configure: `cmake -S . -B build-valg -DCMAKE_BUILD_TYPE=Debug`
- Memcheck (correctness — run by CI): `cmake --build build-valg --target memcheck`
- CPU/cache profile: `cmake --build build-valg --target callgrind` → read with `callgrind_annotate build-valg/callgrind.out.<pid>`
- Heap profile: `cmake --build build-valg --target massif` → read with `ms_print build-valg/massif.out.<pid>`
- Profilers run on `benchmarks` if built (`-DBENCHMARKS=ON`), else `inference_app` (pass args via `-DVALGRIND_PROFILE_ARGS="..."`).
- Lower-overhead alternative: `perf record ./build/benchmarks && perf report`.
- Optional suppressions file: `valgrind.supp` at repo root is picked up automatically if present.

## Pre-commit
- Install: `pip install pre-commit && pre-commit install`
- Run all: `pre-commit run --all-files`

## Usage
- Detection: `./build/inference_app model.onnx image.jpg coco-labels-91.txt`
- Segmentation: add `--segmentation`
- Video: replace image with video file (e.g., video.mp4)
- Display: add `--display`
- TensorRT engine: use .engine or .trt model file

## Notes
- Only one backend (ONNX Runtime, TensorRT, or ExecuTorch) can be enabled at compile time.
- TensorRT requires manually installed CUDA toolkit.
- Data directory is auto-created by CMake.
- CI compiles the TensorRT backend and the GPU pipeline (`gpu-compile.yml`) but cannot run them; ExecuTorch is neither compiled nor run by CI. Test the behaviour of all three manually.