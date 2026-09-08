# Changelog

Notable user-visible changes to this project and compatibility updates for upstream
[`rfdetr`](https://github.com/roboflow/rf-detr).

## [Unreleased]

### Added

- TensorRT-only GPU pipeline with independently selectable DALI preprocessing and CUDA
  segmentation postprocessing (`USE_DALI`, `USE_CUDA_POSTPROCESS`, or `USE_GPU_PIPELINE`).
  Runtime flags remain opt-in, so the CPU path is unchanged.
- Compile-only CI matrix for TensorRT, DALI, and CUDA postprocessing under `-WERROR`.
- `scripts/run_gate.sh` and the rented-GPU runbook for repeatable hardware verification.
- CLI overrides for resolution, maximum detections, mask threshold, and background-class slot.

### Changed

- Aligned export tooling with [rfdetr 1.10.1](https://github.com/roboflow/rf-detr/releases/tag/1.10.1): CUDA/XLA training fixes; exported tensors, runtime operators and C++ decoding are unchanged. See the [validation record](specs/features/2026-09-08-rfdetr-1.10.1-alignment/validation.md).

- Aligned export tooling with
  [`rfdetr` 1.10.0](https://github.com/roboflow/rf-detr/releases/tag/1.10.0).
  Export scripts now use stable explicit artifact names and report the path returned by upstream.
  The exported tensor contract and ONNX opset 17 are unchanged.
- Aligned decoding with `rfdetr` 1.9.3/1.9.4: detection, segmentation, and keypoint paths rank
  the flattened query/class score grid, allow multiple classes per query, exclude the configured
  background slot before capping, and use deterministic tie ordering.
- Updated to `rfdetr` 1.9.1 resize semantics: CPU and CUDA mask resize clamp image borders, and
  ExecuTorch moved to v1.4.0 with optimized kernels required by current `.pte` exports.
- Centralized third-party pins in `versions.env`; CMake, scripts, CI, and Docker consume or
  validate the shared values.
- Replaced the parametric `Dockerfile` with `dockerfile.onnxrt`, `dockerfile.executorch`, and
  `dockerfile.trt`. Shared blocks and duplicated pin defaults are checked in CI.
- Split procedural material out of the README into focused documents under `docs/`.

### Fixed

- ONNX Runtime downloads now select archives from the target OS and architecture.
- TensorRT builds now carry the CUDA include path, use the valid NVIDIA archive URL, and compile
  cleanly under strict warnings.
- The GPU gate no longer reports false passes, records dependency provenance, and supports local
  execution without arming an unwanted shutdown watchdog.
- GPU score filtering, background handling, and CPU/CUDA mask borders now follow the same
  postprocessing contract.

### Validation status

- A real RTX 3060 gate run completed with 9 passes, 0 failures, and 5 explicitly unrun checks.
- CUDA segmentation postprocessing matched the CPU path on the tested image. DALI preprocessing
  changed one final score by up to 0.0163, so formal parity remains open.
- Still unrun: golden/dense parity fixtures, four-path per-stage benchmarks, and a 1000-frame
  `compute-sanitizer` run. See [`specs/roadmap.md`](specs/roadmap.md).
- TensorRT, DALI, CUDA, ExecuTorch, and Docker runtime behavior is not fully exercised by CI.

### Known issues

- Keypoint models exported with `rfdetr` 1.8.2+ use the active-first `[17]` schema, while the
  default decoder still expects `{0, 17}`. Current documented exports may therefore fail to
  decode; pre-1.8.2 exports remain supported.
- `deploy/export_executorch.py` cannot yet export segmentation variants; they must be exported
  through the upstream Python API.
- Image and video output paths are fixed to `output_image.jpg` and `output_video.mp4`.
- GPU parity fixtures, benchmarks, and the long sanitizer run required for the next release are
  incomplete.

## [v0.4.0] - 2026-08-04

### Added

- ExecuTorch backend for `.pte` programs exported by `rfdetr` 1.9.0+, including XNNPACK and
  portable delegates.
- Unified dependency resolver for apt, Conan, vcpkg, vendored, downloaded, and FetchContent
  dependencies.
- ThreadSanitizer and Valgrind build targets.

### Changed

- Enabling more than one inference backend is now a configure-time error.
- Export tooling moved to `rfdetr` 1.9.0 and ExecuTorch programs were validated against the ONNX
  path: identical detections, box delta at most `1e-4` px, and score delta at most `1e-6`.

## [v0.3.0] - 2026-07-03

- Replaced the default OpenCV media/display stack with FFmpeg, SDL2, and stb; OpenCV remains
  selectable with `USE_OPENCV=ON`.
- Added the inference-backend by media-backend Docker build matrix.
- Stabilized media shutdown and tightened CI/static-analysis checks.

## [v0.2.2] - 2026-07-01

- Completed alignment with `rfdetr` 1.8.3 and updated export documentation and package pins.

## [v0.2.1] - 2026-07-01

- Clamped decoded boxes to image bounds, matching the `rfdetr` 1.8.3 fix.

## [v0.2.0] - 2026-06-17

- Added keypoint model export, decoding, drawing, video support, and tests.
- Updated export tooling to `rfdetr` 1.8.0 and removed the obsolete `--simplify` option.

## [v0.1.3] - 2026-05-29

- Updated export tooling to `rfdetr` 1.7.0, including sized model variants and TensorRT-compatible
  dynamic-shape exports.

## [v0.1.2] - 2026-05-12

- Updated to `rfdetr[onnx]` 1.6.5.post0 and added explicit export-device selection.

## [v0.1.1] - 2026-02-17

- Updated export tooling to `rfdetr` 1.4.3.

## [v0.1.0] - 2026-02-14

- Added sized detection and segmentation exports, ONNX Runtime output-count validation, and the
  initial native inference workflow.

[Unreleased]: https://github.com/olibartfast/rf-detr-cpp-inference/compare/v0.4.0...develop
[v0.4.0]: https://github.com/olibartfast/rf-detr-cpp-inference/compare/v0.3.0...v0.4.0
[v0.3.0]: https://github.com/olibartfast/rf-detr-cpp-inference/compare/v0.2.2...v0.3.0
[v0.2.2]: https://github.com/olibartfast/rf-detr-cpp-inference/compare/v0.2.1...v0.2.2
[v0.2.1]: https://github.com/olibartfast/rf-detr-cpp-inference/compare/v0.2.0...v0.2.1
[v0.2.0]: https://github.com/olibartfast/rf-detr-cpp-inference/compare/v0.1.3...v0.2.0
[v0.1.3]: https://github.com/olibartfast/rf-detr-cpp-inference/compare/v0.1.2...v0.1.3
[v0.1.2]: https://github.com/olibartfast/rf-detr-cpp-inference/compare/v0.1.1...v0.1.2
[v0.1.1]: https://github.com/olibartfast/rf-detr-cpp-inference/compare/v0.1.0...v0.1.1
[v0.1.0]: https://github.com/olibartfast/rf-detr-cpp-inference/releases/tag/v0.1.0
