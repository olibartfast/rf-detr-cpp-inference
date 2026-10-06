# Changelog

Notable user-visible changes to this project and compatibility updates for upstream
[`rfdetr`](https://github.com/roboflow/rf-detr).

## [Unreleased]

### Added

- GPU parity tests for greyscale and CMYK JPEGs (`GpuParityCudaPreprocess.GreyscaleJpegMatchesCpu`, `.CmykJpegTakesStbFallback`, fixtures `small_gray.jpg` / `small_cmyk.jpg`). nvJPEG decodes a greyscale JPEG to BGR within `0.016` of the CPU tensor, and a CMYK JPEG takes the stb fallback. Verified on an RTX 3060 Laptop, with greyscale and CMYK images end to end through the app as well.
- **CUDA GPU preprocessing** (`-DUSE_CUDA_PREPROCESS=ON`), now the default GPU preprocessor.
  A fused CUDA kernel does the bilinear stretch, BGR→RGB and ImageNet normalisation straight into
  the TensorRT input binding, matching the CPU preprocess to within `1.1e-6` (gate `1e-5`).
  Still images that are JPEGs decode on the GPU with nvJPEG, chosen from the file header, not
  its extension. Other formats decode with stb on the CPU and are uploaded. nvJPEG ships with
  the CUDA Toolkit, so there is nothing to stage and no new pin. On an RTX 3060 Laptop, from a
  1280×720 source, preprocessing a video frame took 0.55 ms against 6.8–11.9 ms for the CPU path
  plus tensor upload (432–576), and a JPEG image 4.0 ms against 20–25 ms. That is also faster
  than DALI on the same card: 0.56 vs 0.93–0.98 ms per frame, 4.2 vs 5.0 ms per JPEG.
  Verified on an RTX 3060 Laptop (sm_86, driver 610.43.02) in the NGC 26.08 TensorRT image
  (TensorRT 11.2.1.2, CUDA 13.4 in forward-compatibility mode). The run covered GPU parity tests,
  real-model detection, segmentation and keypoint runs, and `compute-sanitizer` over a
  1192-frame video. The record is in `specs/features/2026-09-25-cuda-preprocess/validation.md`.
- `--gpu-preprocess` uses whichever GPU preprocessor the build selected. **DALI stays as the
  alternative** (`-DUSE_DALI=ON`). The two are exclusive: enabling both is a configure-time error.
- `CMakePresets.json`: `gpu-pipeline` is now TensorRT + CUDA preprocessing + CUDA
  postprocessing, and the new `gpu-pipeline-dali` keeps the DALI variant.
- `dockerfile.trt` `GPU_PIPELINE` values `pre` (CUDA preprocessing), `post` (CUDA
  postprocessing) and `dali-on` (DALI preprocessing + CUDA postprocessing).
- GPU parity tests for the CUDA preprocessor: frame path, nvJPEG path, PNG fallback and header
  probe (unit), plus an end-to-end PNG-fallback case (integration). New benchmarks
  `BM_CudaPreprocessFrame`, `BM_CudaPreprocessEncoded`, `BM_DaliPreprocessFrame`, and CPU baselines
  that include the JPEG decode (`BM_CpuPreprocessEncoded`) or the tensor upload
  (`BM_CpuPreprocessUpload`).
- TensorRT 11.x support in the TensorRT backend. TensorRT 11 removed weak typing and
  `BuilderFlag::kFP16`, which made the backend fail to compile. An engine built from an
  `.onnx` takes the model's own precision, so an FP16 engine needs an FP16-converted ONNX — see
  "TensorRT 11 and FP16" in `docs/export.md`. Existing `.engine` files must be rebuilt after
  switching TensorRT versions.

### Removed

- **TensorRT 8.x–10.x and DALI 1.x support.** The backend now requires TensorRT 11
  (`NV_TENSORRT_MAJOR < 11` is a compile error, `TENSORRT_VERSION < 11` a configure error), and
  only the pinned `TENSORRT_VERSION`/`DALI_VERSION` are staged and compile-checked in CI.
  `export_trt.sh` no longer passes `--fp16`, and DALI staging (`fetch_dali.sh`, `dockerfile.trt`)
  expects the DALI 2.x layout.

### Changed

- **ONNX Runtime 1.21.0 → 1.28.0** (`ONNX_RUNTIME_VERSION`), the version in the NGC Triton container that `NGC_CONTAINER_TAG` pins (`tritonserver:26.08-py3` ships `libonnxruntime.so.1.28.0`). The official archives exist for all four automatic-download targets. No source change: the default build compiles under `-DWERROR=ON`, all 10 ctest entries and the 5 model-backed integration tests pass, and detection output on `data/dog.jpg` is identical to 1.21.0. One segmentation score moved in the sixth decimal and the masks are identical. `dockerfile.onnxrt` builds and runs with both `MEDIA_BACKEND` values; the OpenCV image's slightly different scores were already there on 1.21.0 and come from its own decode and resize.
- Aligned export tooling with [rfdetr 1.11.2](https://github.com/roboflow/rf-detr/releases/tag/1.11.2) (from 1.10.1, covering [1.11.0](https://github.com/roboflow/rf-detr/releases/tag/1.11.0) and [1.11.1](https://github.com/roboflow/rf-detr/releases/tag/1.11.1)). Exported tensors, opset 17 and C++ decoding are unchanged: detection, segmentation and keypoint exports give bit-identical outputs to 1.10.1. Upstream moved export internals into `Exporter` classes, but `deploy/` uses only `RFDETR.export()`, which did not change. Two pieces of export guidance changed. The `[executorch]` extra now caps ExecuTorch below 1.4, so `.pte` files are exported with 1.3.x; that `.pte` was verified to run on the pinned v1.4.0 C++ runtime, and `docs/export.md` no longer says to force-install the runtime's version. `export(format="tensorrt", fp16=True)` now builds a real FP16 engine on TensorRT 11 with float32 I/O, which the C++ backend accepts. See the [validation record](specs/features/2026-10-06-rfdetr-1.11.2-alignment/validation.md).
- **`-DUSE_GPU_PIPELINE=ON` now selects CUDA preprocessing**, not DALI; add `-DUSE_DALI=ON` for
  the DALI pipeline. Likewise `dockerfile.trt` `GPU_PIPELINE=on` now builds CUDA preprocessing +
  CUDA postprocessing (the old DALI + CUDA image is `dali-on`), and `GPU_PIPELINE=cuda` fails the
  build with a message naming its replacement, `post`.
- `gpu-compile.yml` installs `libnvjpeg-dev` and compiles six configurations: TensorRT alone,
  each GPU preprocessor, CUDA postprocessing, and both full pipelines. A step also checks that
  the two preprocessors are rejected together. `scripts/run_gate.sh` builds and checks both
  full pipelines.
- **End-to-end GPU parity tolerances for preprocessing** (`integration_test_gpu_parity.cpp`). A
  comparison where any GPU preprocessor produced the input now allows score `0.06`, box centre
  1% of the image's longer side and mask IoU `0.95`. Comparisons with an identical input tensor
  keep `1e-3` / 1 px / `0.999`. The old `0.03` / 1 px bound failed for DALI too on this stack.
  The engine amplifies input noise: nudging one element of the input tensor by `1e-6` moved
  `rfdetr-seg-medium`'s logits by up to 8.3, so no GPU preprocessor can meet the tight bound.
  Tensor-level parity is still gated tightly by the unit tests.
- **Pinned GPU stack moved to NGC 26.08** (`nvcr.io/nvidia/tensorrt:26.08-py3`): TensorRT
  10.13.3.9 → **11.2.1.2**, CUDA 13.0 → **13.3**, `NGC_CONTAINER_TAG` 25.12 → **26.08**, DALI
  1.51.2 → **2.2.0**. `dockerfile.trt` builds on the 26.08 images. Rebuild cached `.engine` files;
  an `.onnx` now builds an FP32 engine unless converted to FP16 first (TensorRT 11 has no FP16
  builder flag).
- The TensorRT download uses NVIDIA's 11.x archive naming,
  `TensorRT-Enterprise-<v>-Linux-x86_64-cuda-<cuda>-Release-external.tar.zst`. "Enterprise" is NVIDIA's name for standard TensorRT from 11.x, under the same
  free license.
- DALI staging (`dockerfile.trt` and `scripts/fetch_dali.sh`) also copies nvImageCodec and its
  codec libraries next to `libdali.so`: DALI 2.x loads them with `dlopen()`, and without them
  `--gpu-preprocess` fails with `dlopen libnvimgcodec.so failed!`. Re-run `fetch_dali.sh` into an
  empty directory to replace a DALI 1.x prefix.
- CI header staging takes the DALI wheel from the `cuda130` index and looks up its file name.
- Pinned versions are stated only in `versions.env` and restated only where a file cannot read it.
  `docs/` and `specs/` now name the variable (`TENSORRT_VERSION`, …) instead of repeating its value,
  and commands read it via `source scripts/versions.sh`; `check_version_sync.sh` now also verifies
  the README version tables, so a bump no longer needs hand-edited prose.
- The `trtexec` recipes in `docs/export.md` target TensorRT 11 (no `--fp16`).
- The TensorRT backend rejects an engine whose inputs or outputs are not float32, instead of
  copying float32-sized buffers into them. rfdetr exports are float32 throughout; this guards a
  reduced-precision ONNX converted without keeping its I/O types.
- `export_trt.sh` passes `trtexec --fp16` only when the container's `trtexec` still accepts it
  (TensorRT 11 removed the flag).

### Fixed

- Reconfiguring an existing build directory after `ONNX_RUNTIME_VERSION` changes no longer fails with `OnnxRuntime library not found at …/onnxruntime-linux-x64-<old>/lib/libonnxruntime.so.<new>`. The automatic download cached its extract dir in `ONNXRUNTIME_ROOTDIR`, and that cached root then took priority over the new pin. A cached root inside `DEPS_PROVIDED_DIR` that names a different version is now dropped and the pinned archive downloaded; a root you set yourself is still used as-is. `OnnxRuntimeCatalog-linux-x64-stale-download` covers it.
- The `benchmarks` target failed to compile with `-DUSE_CUDA_POSTPROCESS=ON` (it includes
  `gpu_test_utils.hpp`, which needs GoogleTest headers it never linked) and without any GPU
  preprocessor (`encode_jpeg` was unused under `-Werror`). Both now build.

## [v0.5.1] - 2026-09-19

### Changed

- Reconciled the CMake project version, vcpkg manifest and README badge to 0.5.1.

- Documentation restructured around a two-tier entry point
  ([#13](https://github.com/olibartfast/rf-detr-cpp-inference/pull/13)). `README.md` is now a quick start —
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
- `docs/advanced-usage.md` joined the list of prose restatements of `versions.env` in
  `specs/tech-stack.md`. The review finding to strip its version pins was declined: `AGENTS.md`
  requires prose version statements, and the page is inside the manual reconciliation step, not
  one of the four machine-checked locations.

### Fixed

Defects found in the [`#13`](https://github.com/olibartfast/rf-detr-cpp-inference/pull/13)
review thread and corrected before merge:

- The README quick start did not install `python3` or `python3-venv` before step 3's
  `python3 -m venv`, which fails on a clean Ubuntu box with `ensurepip is not available`.
- The advanced reference called itself the complete CMake option list while omitting real
  user-facing cache variables. It now documents `ONNXRUNTIME_ROOTDIR`, `TENSORRT_ROOTDIR`,
  `EXECUTORCH_ROOTDIR`, `DALI_ROOT`/`DALI_ROOTDIR`, `RFDETR_VERSIONS_ENV`,
  `VALGRIND_MEMCHECK_OPTS`, and `VALGRIND_PROFILE_ARGS`.
- The embedded `Config` reference described `resolution` as auto-detected when its real default
  is `560`; only `0` activates auto-detection, which the CLI supplies. Embedders following the
  old text would have silently requested a 560×560 input.
- The command-line reference claimed every flag works in every mode. `--gpu-postprocess` is
  rejected without `--segmentation`, both GPU flags require their build options, and `--display`
  applies to video only.
- The advanced reference's "Where to Go Next" sent readers to `docs/usage.md` for the `Config`
  reference, which this release had just moved into the advanced guide itself.
- The CI coverage note said "both push/PR workflows" when `ci.yml`, `lint.yml`, and
  `gpu-compile.yml` all trigger on push and pull request to `master` and `develop`; the same
  sentence in `specs/tech-stack.md` was corrected.
- The README versions table said CUDA Toolkit `13.x` rather than the exact `13.0` pin in
  `versions.env`.

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

[Unreleased]: https://github.com/olibartfast/rf-detr-cpp-inference/compare/v0.5.1...develop
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
