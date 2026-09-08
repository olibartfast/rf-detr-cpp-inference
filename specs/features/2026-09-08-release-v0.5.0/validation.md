# Validation

## Recovery

- Latest OpenCode session: ses_f7f65c738ffebDEmj0hT3V6yYF.
- Fetched develop matches origin/develop at 938dd6f53acf5b0f8388d787641f6b2757badc28.
- All three CI workflows passed that exact develop baseline.
- Latest published project release remains v0.4.0.
- Official upstream 1.10.1 release notes checked via GitHub API; local export pin
  already matches. No additional upstream alignment is part of this release.
- Older worktree and its uncommitted files remain untouched.

## Current checks

- Reproduced Darwin/arm64 catalog failure with USE_ONNX_RUNTIME=OFF before repair.
- PASS after repair: all eight catalog cases, covering four archive coordinates,
  disabled ONNX, unsupported architecture and offline provided prefixes.
  These are CMake catalog probes, not native macOS/Windows builds.
- PASS: default Release/WERROR configure/build and 10/10 CTest entries.
- PASS: dependency pin synchronization, shared Docker blocks and git diff --check.
- PASS: explicit real-model integration rerun — 56 unit tests, 5 integration tests with
  no skips, and 7 exporter tests against the real 1.10.1 exports.
- PASS: all twelve Docker builds (onnxrt/executorch/trt × ffmpeg/opencv × GPU_PIPELINE
  off/dali/cuda/on); the rebuilt ONNX container produced detections from the real Nano model.

## Release gates

- PASS: 1000-frame GPU `compute-sanitizer` run — inside the NGC 25.12 (CUDA 13.1) container,
  whose `compute-sanitizer` 2025.4 pairs with the app; the locally installed sanitizer/toolkit
  could not instrument the app. Engine built for the container's TensorRT (10.14), 1000 frames
  produced, 0 error/leak findings, `--error-exitcode 99` clean.
- PASS: four-path per-stage benchmark with a real engine (RTX 3060 Laptop, TensorRT 10.13.3).
  CPU preprocess 6.75 ms (432) / 11.53 ms (560), DALI preprocess 4.51 ms (432) / 4.59 ms (576),
  H2D 0.64 ms, GPU compute 11.61 ms, D2H 1.30 ms, segmentation postprocess 3483 ms CPU vs 570 ms
  GPU (1080p dense fixture).

## Release metadata

- PASS: `CMakeLists.txt` `project()` declares `VERSION 0.5.0`, `vcpkg.json` and the README badge
  agree on 0.5.0.
- Pending: clean release build, integration and publication.
