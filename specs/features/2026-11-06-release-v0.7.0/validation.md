# Release validation

Verified on 2026-11-06 on Linux x86_64 (Intel Core i5-6500T, 4 logical CPUs, no NVIDIA
device). The feature work lives in
[`../2026-11-06-litert-backend/validation.md`](../2026-11-06-litert-backend/validation.md);
this record covers the release gates.

## Automated — no GPU

- [x] Project version agrees in CMakeLists.txt, vcpkg.json, README.md and specs/tech-stack.md.
- [x] ./scripts/check_version_sync.sh
- [x] ./scripts/check_dockerfile_parity.sh
- [x] Format: find src tests -name '*.cpp' -o -name '*.hpp' | xargs clang-format-18 --dry-run --Werror
- [x] Clang-tidy: find src -name '*.cpp' ! -name 'tensorrt_backend.cpp' | xargs clang-tidy-18 -p build
- [x] Cppcheck: cppcheck --enable=all --check-level=exhaustive --std=c++20 --suppress=missingIncludeSystem --suppress=unmatchedSuppression --suppress=unusedFunction --error-exitcode=1 -I src src/
- [x] Strict default and LiteRT builds and unit tests; real-model parity records retained.
- [x] Every affected locally runnable Docker combination passes; unavailable cases explicitly UNRUN.
- [x] Fresh tagged checkout builds/tests with default ONNX Runtime + FFmpeg/SDL2 media.

## Hardware checks

TensorRT/ExecuTorch manual runtime checks and GPU display checks must be recorded honestly.
No claim of GPU/NPU LiteRT support is allowed.

- **TensorRT runtime: UNRUN.** No NVIDIA device is present (`nvidia-smi` is unavailable). Both
  TensorRT images build and their `inference_app --help` starts; no engine was executed.
- **ExecuTorch runtime: UNRUN.** No `.pte` fixture exists on this host. Both ExecuTorch images
  build and their `inference_app --help` starts; no `.pte` was executed.
- **GPU display and DALI real-screen playback: UNRUN.** Carried forward from v0.6.1; no display
  hardware on this host.
- **LiteRT: CPU only.** No GPU/NPU delegate is compiled or claimed.

## Manual / publication

- [x] Changelog reads as a concise list of user-visible changes; known limitations included.
- [x] Detection/segmentation-only showcase fully decodes, with the rejected pose section absent.
- [x] Feature merged to develop (merge commit `2689ea8`); Phase 8 marked complete.
- [x] Git-flow release merged/tagged; master merged back into develop. The release commit is
      `d145031`, merged into `master` by `156373f`, which the annotated tag `v0.7.0` (`97d5de0`)
      points at; `master` was merged back into `develop` by `f401f59`.
- [x] Mandatory lint gates rerun before every commit and push.
- [x] Authorized remote branches/tag and GitHub release verified; see [Publication](#publication). The
      tag and release were later re-cut on a green commit; see [Re-cut publication](#re-cut-publication).

## Results

| Gate | Result |
|------|--------|
| Version statements | PASS: 0.7.0 in `CMakeLists.txt`, `vcpkg.json`, README badge and `specs/tech-stack.md`; `CMAKE_PROJECT_VERSION` 0.7.0 in a fresh configure |
| Version sync | PASS: every restated pin agrees with `versions.env` |
| Dockerfile parity | PASS: five shared blocks agree across the three backend Dockerfiles |
| Format | PASS: clang-format 18.1.8, no changes |
| Clang-tidy | PASS: 17/17 files, 0 findings, 6 line-scoped `NOLINT`s |
| Cppcheck | PASS: exhaustive analysis, 18/18 files, 0 findings |
| Default build + tests | PASS: `-DWERROR=ON`, Release, FFmpeg/SDL2 media; 109/109 gtest cases |
| LiteRT build + tests | PASS: `-DWERROR=ON`, Release, OpenCV media, staged `LITERT_ROOTDIR`; 110/110 gtest cases |
| LiteRT real-model parity | PASS: detection and segmentation native/Python comparisons; see the feature validation record |
| ASan + UBSan | PASS: LiteRT unit tests and both real-model lifecycle/parity cases |
| Showcase | PASS: 64 s, 1920x1080 at 30 fps, H.264 + AAC; all 1,920 video frames and the full audio/video stream decode; detection and segmentation only |
| CHANGELOG | PASS: `[Unreleased]` moved to `[v0.7.0] - 2026-11-06`; known issues retained |
| Roadmap | PASS: Phase 8 ticked and marked complete; status baseline moved to v0.7.0 |

### Docker matrix

Every combination was built from the release tree. ONNX Runtime images were also run against a
real model; TensorRT and ExecuTorch images were started with `--help` but their inference paths
are UNRUN for the reasons above.

| Backend | Media backend | Build | Runtime |
|---------|---------------|-------|---------|
| ONNX Runtime | FFmpeg | PASS | PASS: `data/dog.jpg`, 3 detections |
| ONNX Runtime | OpenCV | PASS | PASS: `data/dog.jpg`, 3 detections |
| ExecuTorch | FFmpeg | PASS | UNRUN: no `.pte` fixture |
| ExecuTorch | OpenCV | PASS | UNRUN: no `.pte` fixture |
| TensorRT | FFmpeg | PASS | UNRUN: no NVIDIA device |
| TensorRT | OpenCV | PASS | UNRUN: no NVIDIA device |

The TensorRT build used `GPU_PIPELINE=off`; no GPU pipeline runtime was attempted because the
host has no device. Docker image sizes and IDs are in
`output/release-v0.7.0-validation/docker/` (ignored artifact directory).

### Reproduce the local gates

```bash
./scripts/check_version_sync.sh
./scripts/check_dockerfile_parity.sh
find src tests -name '*.cpp' -o -name '*.hpp' | xargs clang-format-18 --dry-run --Werror
cmake -S . -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
find src -name '*.cpp' ! -name 'tensorrt_backend.cpp' | xargs clang-tidy-18 -p build
cppcheck --enable=all --check-level=exhaustive --std=c++20 \
  --suppress=missingIncludeSystem --suppress=unmatchedSuppression \
  --suppress=unusedFunction --error-exitcode=1 -I src src/
ctest --test-dir build --output-on-failure -R UnitTests
```

The FFmpeg/SDL2 default configure on this host needed the staged SDL2 development package
(`PKG_CONFIG_PATH` pointing at it); that is an environment detail, not a source requirement.

### Fresh tagged checkout

PASS. `git clone` of release commit `15ac269` into `/tmp/rfdetr-fresh`. That commit differs from
the tagged release commit `d145031` only in this record. Then:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DWERROR=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure -R UnitTests
```

Configured with ONNX Runtime 1.28.0 and FFmpeg/SDL2 media (`USE_OPENCV=OFF`); the binary links
the cloned tree's ONNX Runtime, system FFmpeg and SDL2. 109/109 gtest cases pass.

### Publication (first cut, superseded)

Verified after the push from `develop` on 2026-11-06. This tag object and release were later removed
and recreated; see [Re-cut publication](#re-cut-publication).

| Ref | Expected | Verified |
|-----|----------|----------|
| `develop` | `f401f59` | `origin/develop` = `f401f599bd9e…` |
| `master` | `156373f` | `origin/master` = `156373f07851…` |
| Tag `v0.7.0` | annotated object `97d5de0` → `156373f` | `git ls-remote --tags origin` returns the same object and peeled commit |
| GitHub release | <https://github.com/olibartfast/rf-detr-cpp-inference/releases/tag/v0.7.0> | published 2026-11-06T17:05:36Z, target `master`, not a draft or pre-release |

The local `release/v0.7.0` and `feature/litert-backend` branches were deleted after the merge.

#### CI on the published commits

| Commit | Build & Test | GPU Backend Compile | C++ Lint & Build |
|--------|--------------|---------------------|------------------|
| `master` `156373f` (first tag `v0.7.0`) | PASS | PASS | **FAIL**: Format Check and Clang-Tidy |
| `develop` `f401f59` | PASS | PASS | **FAIL**: Format Check and Clang-Tidy |
| `develop` `9c95c35` (fix) | PASS | PASS | PASS |
| `master` `bc06231` (re-cut tag `v0.7.0`) | PASS | PASS | PASS |
| `develop` `994c933` | PASS | PASS | PASS |
| Tag `v0.7.0` `c201b12` → `bc06231` | not triggered on tags | not triggered on tags | PASS, all five jobs |

The two lint jobs failed before reaching the source: `ubuntu-latest` had moved to Ubuntu 26.04,
which no longer ships `clang-format-18` / `clang-tidy-18`, so both commands exited 127. The same
gates passed locally with clang 18.1.8 (see [Results](#results)); this was an image/toolchain
failure, not a finding in the release tree. `9c95c35` pins every `lint.yml` job to `ubuntu-24.04`,
installs clang 18 explicitly and adds a `v*` tag trigger; all three workflows are green on it.

`v0.7.0` itself was **not** linted by CI at first: the tag trigger landed in `9c95c35`, after the
tag was published, so the red Format Check / Clang-Tidy run stays in the history of `156373f`.

The maintainer approved re-cutting the release rather than publishing a tag with red gates. The
published tag and GitHub release were removed once `develop` and `master` were green, and `v0.7.0`
was recreated on the `master` merge commit that contains the fix. Because `lint.yml` now runs on
`v*` tags, the tag itself was linted before the release was republished.

### Re-cut publication

Verified 2026-11-06 after `9c95c35` turned both long-lived branches green. The first tag object
`97d5de0` (→ `156373f`) and its GitHub release were deleted, then the tag was recreated on the
`master` merge commit that contains the fix.

| Ref | Expected | Verified |
|-----|----------|----------|
| `develop` | `994c933` at re-cut (this record is committed on top) | `origin/develop` = `994c933…` |
| `master` | `bc06231` | `origin/master` = `bc062316…` |
| Tag `v0.7.0` | annotated object `c201b12` → `bc06231` | `git ls-remote --tags origin` returns the same object and peeled commit |
| GitHub release | <https://github.com/olibartfast/rf-detr-cpp-inference/releases/tag/v0.7.0> | republished 2026-11-06T19:54:38Z on the new tag, target `master`, not a draft or pre-release |

The `v*` trigger linted the tag itself: [run 42274688377](https://github.com/olibartfast/rf-detr-cpp-inference/actions/runs/42274688377)
passed Format Check, Clang-Tidy, Cppcheck, Version Sync and Build with Strict Warnings. Its Build
with Strict Warnings job took 27m19s because `apt-get install` stalled on the runner's package
mirror; the two preceding runs on `develop` showed the same 20-30 minute delay, so this is runner
infrastructure, not the change. Build & Test and GPU Backend Compile do not trigger on tags; they
were green on the identical `master` commit `bc06231`.
