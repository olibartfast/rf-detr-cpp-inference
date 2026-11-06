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
- [ ] Git-flow release merged/tagged; master merged back into develop.
- [x] Mandatory lint gates rerun before every commit and push.
- [ ] Authorized remote branches/tag and GitHub release verified.

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

PASS. `git clone` of the release commit `15ac269` (the commit amended only by this record) into
`/tmp/rfdetr-fresh`, then:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DWERROR=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure -R UnitTests
```

Configured with ONNX Runtime 1.28.0 and FFmpeg/SDL2 media (`USE_OPENCV=OFF`); the binary links
the cloned tree's ONNX Runtime, system FFmpeg and SDL2. 109/109 gtest cases pass. The post-tag
remote checks are recorded on `develop` after publication.
