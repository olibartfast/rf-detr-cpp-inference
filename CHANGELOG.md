# Changelog

All notable changes to this project. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Validation records live in `specs/features/`, not here.

## [Unreleased]

## [v0.6.1] - 2026-10-09

### Added
- `-DPROFILING=ON`: frame pointers and debug info on every target, for `perf`/Valgrind call graphs.
- CPU decode-path benchmarks (`tests/benchmark/bench_cpu_pipeline.cpp`).
- GPU gate on Google Colab (`scripts/colab/gpu_gate.ipynb`) and headless `--display` check (`scripts/check_display.sh`).
- `DALI_SOURCE=pip ./scripts/fetch_dali.sh` stages DALI without Docker.
- `run_gate.sh`: `CUDA_ARCH` defaults to the card's compute capability; `SKIP_BUILD_MATRIX=1`.

### Changed
- clang-tidy is enforced (`WarningsAsErrors: '*'`); all findings fixed or suppressed per line.
- CLI parsing moved to `rfdetr::cli::parse_cli` (`src/cli_options.*`) with 47 unit tests; `main()` split up.
- `std::endl` replaced with `'\n'`: stdout is no longer flushed per line when piped.

### Fixed
- `run_gate.sh` failed engine builds on a fresh machine (TensorRT libs not yet downloaded).
- `run_gate.sh` reported a missing DALI as `FAIL` instead of `UNRUN`.
- Documented clang-tidy command now excludes `tensorrt_backend.cpp`, as CI does.

### Known issues
- ONNX Runtime runs single-threaded (`SetIntraOpNumThreads(1)` is hardcoded).
- GPU-decoded JPEGs (nvJPEG) do not bit-match the CPU tensor (stb); within GPU-preprocess tolerances.
- `--display` unverified on the DALI build and on a real (non-Xvfb) screen.

## [v0.6.0] - 2026-10-07

### Breaking
- TensorRT 11 required; TensorRT 8.x–10.x and DALI 1.x dropped. Rebuild `.engine` files.
- `-DUSE_GPU_PIPELINE=ON` now builds CUDA preprocessing; add `-DUSE_DALI=ON` for DALI.
- `dockerfile.trt`: `GPU_PIPELINE=cuda` removed (use `post`); `on` = CUDA pre + post, `dali-on` = DALI pre + CUDA post.

### Added
- CUDA GPU preprocessing (`-DUSE_CUDA_PREPROCESS=ON`), the default GPU preprocessor, with nvJPEG decode for JPEGs.
- `gpu-pipeline-dali` CMake preset; `dockerfile.trt` `GPU_PIPELINE=pre|post|dali-on`.
- GPU parity tests and benchmarks for the CUDA preprocessor, greyscale and CMYK JPEGs.

### Changed
- GPU stack on NGC 26.08: TensorRT 11, CUDA 13, DALI 2.x.
- ONNX Runtime 1.21.0 → 1.28.0.
- Export tooling aligned with [rfdetr 1.11.2](https://github.com/roboflow/rf-detr/releases/tag/1.11.2); exported tensors unchanged.
- TensorRT backend rejects engines with non-float32 I/O.
- End-to-end GPU-preprocess parity tolerances: score `0.06`, box centre 1 % of the longer side, mask IoU `0.95`.
- `gpu-compile.yml` compiles six GPU configurations.
- Pinned versions stated only in `versions.env`; `check_version_sync.sh` also checks the README tables.

### Fixed
- Reconfiguring after an `ONNX_RUNTIME_VERSION` change reused the stale download.
- `benchmarks` target failed to compile with `-DUSE_CUDA_POSTPROCESS=ON` or without a GPU preprocessor.
- DALI 2.x staging now includes nvImageCodec (`dlopen libnvimgcodec.so failed!`).

## [v0.5.1] - 2026-09-19

### Changed
- Docs restructured: README is a quick start; full reference in `docs/advanced-usage.md` ([#13](https://github.com/olibartfast/rf-detr-cpp-inference/pull/13)).
- Maintainer procedures moved from `docs/` to `specs/`.

### Fixed
- README quick start did not install `python3-venv`.
- Docs: missing CMake cache variables, wrong `Config::resolution` default, wrong per-mode flag claims.

## [v0.5.0] - 2026-09-08

### Added
- TensorRT-only GPU pipeline: DALI preprocessing and CUDA segmentation postprocessing (`USE_DALI`, `USE_CUDA_POSTPROCESS`, `USE_GPU_PIPELINE`).
- Compile-only GPU CI matrix; `scripts/run_gate.sh` for hardware verification.
- CLI: resolution, max detections, mask threshold, background-class slot, `--keypoint-counts`, `--output`.
- Segmentation export in `deploy/export_executorch.py`.
- GPU parity fixtures, tests and per-stage benchmarks.

### Changed
- Export tooling aligned with rfdetr 1.9.1–[1.10.1](https://github.com/roboflow/rf-detr/releases/tag/1.10.1); decoding ranks the full query/class score grid.
- ExecuTorch v1.4.0 with optimized kernels.
- Third-party pins centralized in `versions.env`.
- `Dockerfile` replaced by `dockerfile.onnxrt`, `dockerfile.executorch`, `dockerfile.trt`.

### Fixed
- ONNX Runtime download selects the archive by target OS/arch; catalog no longer rejects unsupported targets when ONNX Runtime is off.
- TensorRT build: CUDA include path, archive URL, strict-warning cleanups.
- CPU/CUDA postprocessing contract mismatches (score filter, background, mask borders).
- Hardcoded image/video output paths.

### Known issues
- DALI's JPEG path does not bit-match the CPU tensor (nvJPEG vs stb).

## [v0.4.0] - 2026-08-04

### Added
- ExecuTorch backend for `.pte` (XNNPACK and portable delegates).
- Unified dependency resolver (apt, Conan, vcpkg, vendored, download, FetchContent).
- ThreadSanitizer and Valgrind targets.

### Changed
- Enabling more than one inference backend is a configure-time error.
- Export tooling on rfdetr 1.9.0.

## [v0.3.0] - 2026-07-03
- FFmpeg, SDL2 and stb replace OpenCV as the default media stack (`USE_OPENCV=ON` keeps OpenCV).
- Inference-backend × media-backend Docker matrix.

## [v0.2.2] - 2026-07-01
- Aligned with rfdetr 1.8.3.

## [v0.2.1] - 2026-07-01
- Decoded boxes clamped to image bounds (rfdetr 1.8.3 fix).

## [v0.2.0] - 2026-06-17
- Keypoint models: export, decoding, drawing, video.
- Export tooling on rfdetr 1.8.0; `--simplify` removed.

## [v0.1.3] - 2026-05-29
- Export tooling on rfdetr 1.7.0.

## [v0.1.2] - 2026-05-12
- `rfdetr[onnx]` 1.6.5.post0; export-device selection.

## [v0.1.1] - 2026-02-17
- Export tooling on rfdetr 1.4.3.

## [v0.1.0] - 2026-02-14
- Initial release.

[Unreleased]: https://github.com/olibartfast/rf-detr-cpp-inference/compare/v0.6.1...develop
[v0.6.1]: https://github.com/olibartfast/rf-detr-cpp-inference/compare/v0.6.0...v0.6.1
[v0.6.0]: https://github.com/olibartfast/rf-detr-cpp-inference/compare/v0.5.1...v0.6.0
[v0.5.1]: https://github.com/olibartfast/rf-detr-cpp-inference/compare/v0.5.0...v0.5.1
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
