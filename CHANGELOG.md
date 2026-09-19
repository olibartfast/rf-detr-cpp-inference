# Changelog

Notable user-visible changes to this project and compatibility updates for upstream
[`rfdetr`](https://github.com/roboflow/rf-detr).

## [Unreleased]

### Changed

- Documentation restructured around a two-tier entry point. `README.md` is now a quick start —
  install, build, export a model, run — keeping the version, build-option and backend
  statements the `Spec Sync` rule requires, at a glance. The exhaustive reference moved to the
  new `docs/advanced-usage.md`: every CMake option, the backends in depth, runtime tuning,
  label/class-layout customization, the GPU pipeline at runtime, performance notes, embedding
  `RFDETRInference`, and what CI cannot cover. No behaviour, build option, or pin changed.
- `docs/usage.md` is now purely the operational reference — how to run each mode and what every
  flag does. The material that had accumulated past that (top-k selection theory, class-layout
  guidance, the `Config` table, the embedding example) moved to `docs/advanced-usage.md`, which
  is where it belongs and where it was otherwise duplicated.
- `docs/` now holds only documentation for users of the inference application. The two
  maintainer procedures that had been filed there moved to `specs/`, beside the skills they
  serve: `docs/rented-gpu-runbook.md` → `specs/rented-gpu-runbook.md` and
  `docs/opencode-workflow.md` → `specs/opencode-workflow.md`.

## [v0.5.0] - 2026-09-08

### Added

- TensorRT-only GPU pipeline with independently selectable DALI preprocessing and CUDA
  segmentation postprocessing (`USE_DALI`, `USE_CUDA_POSTPROCESS`, or `USE_GPU_PIPELINE`).
  Runtime flags remain opt-in, so the CPU path is unchanged.
- Compile-only CI matrix for TensorRT, DALI, and CUDA postprocessing under `-WERROR`.
- `scripts/run_gate.sh` and the rented-GPU runbook for repeatable hardware verification.
- CLI overrides for resolution, maximum detections, mask threshold, and background-class slot.
- `--keypoint-counts` CLI flag to decode active-first `[17]` keypoint exports, and `--output` to
  choose the image/video output path.
- Segmentation export in `deploy/export_executorch.py` via `--segmentation`.
- Golden CPU parity fixtures under `tests/data/gpu_parity/`, a `gpu_parity_gen` regenerator, the
  fixture-backed `test_gpu_parity.cpp` (CPU determinism, DALI preprocess parity, no-letterbox,
  end-to-end regression), and the per-stage `bench_gpu_pipeline.cpp`.

### Changed

- Reconciled the CMake project version, vcpkg manifest and README badge to 0.5.0.

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

- Loading the dependency catalog no longer rejects targets without a bundled ONNX Runtime
  download when ONNX Runtime is disabled or supplied through a compatible prefix/package manager.
  Eight CMake regression cases cover archive selection and offline prefix resolution.
- ONNX Runtime downloads now select archives from the target OS and architecture.
- TensorRT builds now carry the CUDA include path, use the valid NVIDIA archive URL, and compile
  cleanly under strict warnings.
- The GPU gate no longer reports false passes, records dependency provenance, and supports local
  execution without arming an unwanted shutdown watchdog.
- GPU score filtering, background handling, and CPU/CUDA mask borders now follow the same
  postprocessing contract.
- Image and video output paths are no longer hardcoded: `--output` selects either.
- Keypoint decoding accepts the active-first `[17]` schema via `--keypoint-counts` (with
  `--background-class-id none`). The official `RFDETRKeypointPreview` checkpoint is
  background-first and still decodes with the default `{0, 17}`, so the earlier "default export
  cannot decode" warning was wrong.

### Validation status

- The four pre/post combinations (`CPU/CPU`, `GPU-pre/CPU-post`, `CPU-pre/GPU-post`, `GPU/GPU`) pass
  end-to-end through a real TensorRT engine: CUDA postprocess matches CPU to the tight tolerance
  (scores `1e-3`, mask IoU ≥ 0.999) and DALI preprocess to the decode-aware bound
  (`integration_test_gpu_parity.cpp`).
- The parity fixtures pass: CPU determinism bit-identical, DALI resize within `8.8e-3`, no
  letterbox, and the dense fixture (200 detections) regresses correctly.
- A real RTX 3060 gate run completed with 9 passes, 0 failures, and 5 explicitly unrun checks.
- DALI resize + normalise matches the CPU path within `8.8e-3` on all three natural fixtures. The
  encoded path diverges by up to `0.069` on the tensor, but the delta is entirely the JPEG decode
  (nvJPEG vs stb) — the frame path isolates resize and shows it is not the source. This remains the
  open "DALI preprocessing parity" item; its end-to-end effect is bounded (one final score shifted
  by up to 0.0163).
- The 1000-frame `compute-sanitizer` run completes with **no findings** and no leak: run inside the
  NGC 25.12 (CUDA 13.1) container, whose `compute-sanitizer` 2025.4 pairs with the app; the locally
  installed sanitizer/toolkit pairing could not instrument it (`Unable to find injection library`).
  1000 frames produced, 0 error/leak findings, exit via the `--error-exitcode 99` gate.
- Per-stage benchmarks recorded with a real engine (RTX 3060 Laptop, TensorRT 10.13.3): CPU
  preprocess 6.75 ms (432) / 11.53 ms (560), DALI preprocess 4.51 ms (432) / 4.59 ms (576), H2D
  0.64 ms, GPU compute 11.61 ms, D2H 1.30 ms, segmentation postprocess 3483 ms CPU vs 570 ms GPU
  (1080p dense fixture). See [`specs/roadmap.md`](specs/roadmap.md).
- TensorRT, DALI, CUDA, ExecuTorch, and Docker runtime behavior is not fully exercised by CI.

### Known issues

- DALI's encoded preprocess does not bit-match the CPU tensor (up to `0.069` max |Δ|): nvJPEG and
  stb decode JPEG differently. Resize itself is within tolerance; closing this fully would require
  the CPU path to decode with the same JPEG decoder.

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

[Unreleased]: https://github.com/olibartfast/rf-detr-cpp-inference/compare/v0.5.0...develop
[v0.5.0]: https://github.com/olibartfast/rf-detr-cpp-inference/compare/v0.4.0...v0.5.0
[v0.4.0]: https://github.com/olibartfast/rf-detr-cpp-inference/compare/v0.3.0...v0.4.0
[v0.3.0]: https://github.com/olibartfast/rf-detr-cpp-inference/compare/v0.2.2...v0.3.0
[v0.2.2]: https://github.com/olibartfast/rf-detr-cpp-inference/compare/v0.2.1...v0.2.2
[v0.2.1]: https://github.com/olibartfast/rf-detr-cpp-inference/compare/v0.2.0...v0.2.1
[v0.2.0]: https://github.com/olibartfast/rf-detr-cpp-inference/compare/v0.1.3...v0.2.0
[v0.1.3]: https://github.com/olibartfast/rf-detr-cpp-inference/compare/v0.1.2...v0.1.3
[v0.1.2]: https://github.com/olibartfast/rf-detr-cpp-inference/compare/v0.1.1...v0.1.2
[v0.1.1]: https://github.com/olibartfast/rf-detr-cpp-inference/compare/v0.1.0...v0.1.1
[v0.1.0]: https://github.com/olibartfast/rf-detr-cpp-inference/releases/tag/v0.1.0
