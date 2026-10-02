# Validation — Phase 6, CUDA preprocessing, DALI as the alternative

Every result is recorded here with the date and the hardware; anything not run is `UNRUN` with
the reason, never a pass. The removal-only checks of the original spec (DALI pins gone, `grep dali`
empty, `-DUSE_DALI=ON` rejected) were withdrawn with the 2026-10-02 scope change and are replaced
below by the checks for the alternative-preprocessor setup.

## Automated — no GPU (CI)

- [x] `./scripts/check_version_sync.sh` passes (2026-10-02)
- [x] `./scripts/check_dockerfile_parity.sh` passes (2026-10-02: 5 shared blocks agree)
- [x] Format: `find src tests -name '*.cpp' -o -name '*.hpp' | xargs clang-format-18 --dry-run --Werror` (2026-10-02)
- [x] Clang-tidy: `cmake -S . -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON` then
      `find src -name '*.cpp' ! -name 'tensorrt_backend.cpp' | xargs clang-tidy-18 -p build`
      (2026-10-02). It was also run with `-DUSE_TENSORRT -DUSE_CUDA_PREPROCESS -DUSE_CUDA_POSTPROCESS`
      on the changed sources, because the default configure compiles the GPU branches away. No
      finding falls on a line this phase added; the three `performance-avoid-endl` it first
      reported there were fixed. The warnings on unchanged lines predate this phase (CI's
      `WarningsAsErrors` is empty).
- [x] Cppcheck: `cppcheck --enable=all --std=c++20 --suppress=missingIncludeSystem --suppress=unmatchedSuppression --suppress=unusedFunction --error-exitcode=1 -I src src/` (2026-10-02; also clean with the GPU defines on the changed sources)
- [x] Default build and tests: `cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build build --parallel`,
      `ctest --test-dir build --output-on-failure -R UnitTests` (2026-10-02: all 10 ctest entries pass)
- [ ] `gpu-compile.yml` green on every matrix entry (runs on the PR; the same six configurations
      built locally, see the group C/D record): TensorRT alone, `+USE_CUDA_PREPROCESS`,
      `+USE_DALI`, `+USE_CUDA_POSTPROCESS`, full pipeline (CUDA), full pipeline (DALI) — all
      `-DWERROR=ON` on the pinned TensorRT — and its exclusive-configure step

## Compile-without-device

- [x] `-DUSE_ONNX_RUNTIME=OFF -DUSE_TENSORRT=ON -DUSE_GPU_PIPELINE=ON -DWERROR=ON` builds
      `inference_app`, `unit_tests`, `integration_tests` and `benchmarks` with the CUDA Toolkit and
      TensorRT (2026-10-02, builder image; compiling never touches the device)
- [x] With no device visible (`docker run --runtime=runc`, so no driver is injected), every
      `GpuParity*` and `GpuPostprocess*` test reports `SKIPPED` and none `FAILED`. The CUDA,
      DALI, CUDA-preprocess-only and CUDA-postprocess-only builds all ran with 0 failures
      (2026-10-02)
- [x] `-DUSE_DALI=ON -DUSE_CUDA_PREPROCESS=ON` fails at configure with a message naming both (2026-10-02)

## Automated — with a device (RTX 3060 Laptop, sm_86, builder image, `--gpus all`)

Tensor tolerances are max abs over the full `[1,3,432,432]` tensor against `*.preprocessed.bin`.

| Check | Tolerance | Result |
|-------|-----------|--------|
| Frame path (`preprocess_bgr_device`), `small`/`wide`/`tall` | ≤ `1e-5` | PASS 2026-09-27 (group A): `1.07e-6`/`1.07e-6`/`8.34e-7` |
| PNG fallback path, `small`/`wide`/`tall` | ≤ `1e-5` | PASS 2026-10-01 (group B): `1.07e-6`/`1.07e-6`/`8.34e-7` |
| Encoded path (nvJPEG + kernel), `small`/`wide`/`tall` | ≤ `1e-1` | PASS 2026-10-01 (group B): `4.85e-2`/`3.41e-2`/`3.13e-2` |
| No letterbox on `wide` and `tall` | pass | PASS 2026-09-27 frame path; 2026-10-01 encoded path |
| End-to-end, `cpupre-gpupost` vs `cpu-cpu` | score `1e-3`, centre 1 px, mask IoU ≥ `0.999` | PASS 2026-10-02, both builds: score Δ 0, centre 3e-5 px, IoU 1 |
| End-to-end, `gpupre-cpupost` (nvJPEG) vs `cpu-cpu` | score `0.06`, centre 1% of longer side, mask IoU ≥ `0.95` (was `0.03` / 1 px, see requirements.md) | PASS 2026-10-02: CUDA score Δ `0.041`, centre 1.0 px, IoU `0.972`; DALI `0.014`, 2.9 px, `0.980` |
| End-to-end, PNG fallback (`gpupre-cpupost` vs `cpu-cpu`, same PNG) | as the row above (was `1e-3` / `0.999`) | PASS 2026-10-02: score Δ `0.013`, centre 1.4 px, IoU `0.980` |
| End-to-end, `gpu-gpu` vs `gpupre-cpupost` | score `1e-3`, centre 1 px, mask IoU ≥ `0.999` | PASS 2026-10-02, both builds: score Δ 0, IoU 1 |
| `dense` fixture | 200 detections, count exact | PASS 2026-10-02 (`GpuParityRegression.DenseFixture`, CUDA and DALI builds) |
| Real model, all three tasks with `--gpu-preprocess` (and `--gpu-postprocess` for segmentation) | same detections and classes; integration tolerances above | PASS 2026-10-02, see the group C record |
| 1000-frame video, `--gpu-preprocess --gpu-postprocess`, `compute-sanitizer --tool memcheck` | 0 errors, 0 leaks | PASS 2026-10-02: 1192 frames, 0 bytes leaked; 2 reported API errors are TensorRT's own `createInferRuntime` IMEX probe, also present in a TensorRT-only build (group C/D record). 1080p, 596 frames, `--report-api-errors no`: 0 errors, 0 leaks |
| Same run, `compute-sanitizer --tool racecheck` on the preprocess kernel | 0 hazards | PASS 2026-10-02: 30 frames at 1080p, `--kernel-name` filtered to each repo kernel (`preprocess_bgr`, `decode_scores`, `select_and_decode`, `resize_threshold_masks`). racecheck: 0 hazards each; synccheck: 0 errors each. Unfiltered racecheck spends its time in TensorRT's own kernels and was stopped |

Record the measured max delta for each tensor row, not only pass/fail, and whether `-fmad=false`
was needed (requirements.md open question 1).

### Group A record (2026-09-27)

RTX 3060 Laptop (sm_86), driver 610.43.02, `rfdetr-trt-builder:26.08` (TensorRT 11.2.1.2, CUDA 13),
clang-18, `-DWERROR=ON`.

- **Open question 1 resolved: `-fmad=false` is not needed.** With nvcc's default FMA contraction the
  frame path is within `1.1e-6` of the golden CPU tensors — ~8000× tighter than DALI's `8.8e-3`.
- `GpuParityCudaPreprocess.EdgeGeometriesMatchCpu` (1×1, 1000×1, 7×333, exact 2× downscale,
  1920×1080→576, 640×480→560) within `1e-5`; `RejectsInvalidArguments` runs without a device.
- `compute-sanitizer --tool memcheck --leak-check full` on `GpuParityCudaPreprocess*`: 0 errors,
  0 bytes leaked. `--tool racecheck` on `FrameMatchesGoldenCpu`: 0 hazards.
- Builds: `USE_CUDA_PREPROCESS` alone, `USE_CUDA_PREPROCESS`+`USE_CUDA_POSTPROCESS`,
  `USE_GPU_PIPELINE` (DALI + both), TensorRT alone — all `-DWERROR=ON`, 0 warnings.
  `unit_tests` in the CUDA-only build: 61/61 pass.
- In the `USE_GPU_PIPELINE` build, `GpuParityPreprocess.EncodedPathBounded` and
  `.NoLetterboxBorders` (DALI encoded path) fail: that builder image predates `1d2b408` and its
  `/opt/dali` lacks nvImageCodec, so DALI's decode `dlopen` fails. Not caused by this change; the
  DALI frame-path test passes. Later DALI runs mount a working `/opt/dali` (group B benchmark record).

### Group B record (2026-10-01)

Same card and image as group A: RTX 3060 Laptop (sm_86), driver 610.43.02, `rfdetr-trt-builder:26.08`
(TensorRT 11.2.1.2, CUDA 13.4 in forward-compatibility mode, nvJPEG 13.2.2.35), GCC 13, `-DWERROR=ON`.

- `rfdetr::gpu::JpegDecoder` (`src/gpu/jpeg_decoder.{hpp,cpp}`), default nvJPEG backend,
  `NVJPEG_OUTPUT_BGRI` into a grow-only `DeviceBuffer`. `probe()` reads only the header and returns
  empty for anything that is not a one- or three-component JPEG with known subsampling, so the caller
  falls back to stb.
- The nvJPEG-vs-stb decoder gap measured `0.031`-`0.048`, below the `0.050`-`0.069` recorded for DALI's
  decode.
- `GpuParityCudaPreprocess.*`: 7/7 pass; `ProbeReadsJpegHeaderOnly` checks the header sizes against stb
  and rejects empty and PNG bytes. `compute-sanitizer --tool memcheck --leak-check full` on that
  group: 0 errors, 0 bytes leaked.
- Builds: `USE_CUDA_PREPROCESS` alone (all targets, `unit_tests` 64/64), and with
  `USE_CUDA_POSTPROCESS` (`unit_tests` 76/76). Both use `-DWERROR=ON` and produced no warnings.
- The PNG fallback is tested at component level (stb → upload → kernel). The orchestrator wiring that
  selects it at runtime lands in group C.
- Open question 3 (greyscale JPEG): **UNRUN**. No fixture is a one-component JPEG, and
  `stbi_write_jpg` always writes three components, so the test cannot generate one. `probe()` accepts
  `NVJPEG_CSS_GRAY`; check it against a real greyscale JPEG during the group F gate.
- Open question 2 (handle lifetime) carries into group C. The decoder documents one instance per
  thread, and `RFDETRInference` must own it the way it owns `dali_encoded_` today.

### Group C/D record (2026-10-02)

Same card and image as group A. DALI runs mount a working `/opt/dali` (with nvImageCodec), taken
from `rfdetr-trt-gate:ffmpeg-on`.

**Build matrix**, `-DWERROR=ON -DBENCHMARKS=ON`, all targets: `USE_GPU_PIPELINE` (CUDA),
`USE_GPU_PIPELINE`+`USE_DALI`, `USE_CUDA_PREPROCESS` alone, `USE_DALI` alone, `USE_CUDA_POSTPROCESS`
alone, TensorRT alone. All build with 0 warnings. The last two first failed in `bench_gpu_pipeline.cpp`
(`encode_jpeg` unused), an existing bug on `develop`, now fixed. `USE_CUDA_PREPROCESS`+`USE_DALI`
fails at configure with the exclusive-preprocessor message.

**Unit tests**, on the device: 76 / 72 / 64 / 60 / 69 / 56 pass in the same order, none skipped. The
DALI tests (`GpuParityPreprocess.*`) pass with the working DALI mounted.

**End-to-end** (`integration_test_gpu_parity`, `rfdetr-seg-medium` at 432 on `data/dog.jpg`): see
the table above. With the tolerances as first written, the CUDA build failed: PNG fallback score Δ
`0.013` against `1e-3`, and JPEG score Δ `0.041` against `0.03`. The DALI build failed too: box
centre 2.9 px against 1 px. A diagnostic (not committed) fed the TensorRT backend directly:

| Comparison | Max logit Δ |
|------------|-------------|
| CPU tensor, host path, run twice | 0 |
| CPU tensor, host path vs device path (`run_inference_device`) | 0 (boxes 0) |
| CPU tensor vs kernel tensor (input Δ `7.2e-7`) | 9.1 |
| CPU tensor vs the same with one element `+1e-6` | 8.3 |

So the device path is exact and the difference is the engine amplifying input noise. Score, box and
mask differences that large follow from any non-bit-identical tensor. The end-to-end bound for
GPU-preprocessed input was set from these measurements (requirements.md, Decisions).

**Real-model app runs** (`inference_app`, CUDA build), CPU path vs `--gpu-preprocess` (plus
`--gpu-postprocess` for segmentation). The same images go through both paths, JPEG and an ffmpeg-made
PNG of it:

| Task / model | Image | Detections CPU / GPU | Classes | Max score Δ | Max box Δ | Other |
|--------------|-------|----------------------|---------|-------------|-----------|-------|
| detection, `rfdetr-nano` | dog.jpg | 4 / 4 | same | 0.011 | 1.38 px | |
| detection | dog.png | 4 / 4 | same | 0.0008 | 0.07 px | |
| detection | bus.jpg | 5 / 5 | same | 0.0016 | 1.01 px | |
| detection | bus.png | 5 / 5 | same | 0 | 0 px | |
| segmentation, `rfdetr-seg-medium`, gpu-pre and gpu-gpu | dog.jpg | 4 / 4 | same | 0.041 | 2.88 px | mask pixels Δ ≤ 2.7 % |
| segmentation, gpu-pre and gpu-gpu | dog.png | 4 / 4 | same | 0.033 | 1.89 px | mask pixels Δ ≤ 0.11 % |
| keypoint, `rfdetr-keypoint` | bus.jpg | 3 / 3 | same | 0.0035 | 0.33 px | keypoints Δ ≤ 1.02 px |
| keypoint | bus.png | 3 / 3 | same | 0 | 0 px | keypoints Δ 0 |

**compute-sanitizer.** memcheck `--leak-check full` on 1192 frames (`people-walking.mp4` looped
once, 768×432) with `--segmentation --gpu-preprocess --gpu-postprocess`: run completed, 0 bytes
leaked, "2 errors". Both are `CUDA API Error: Fabric handle support is dependent on IMEX
channels`, from `cuDeviceGetAttribute` inside TensorRT's `createInferRuntime`, at startup. A
TensorRT-only build with no GPU pipeline reports the same two. They are API return codes from
TensorRT's own probe, not memory errors. The same check on the 1080p video with
`--report-api-errors no`: 596 frames, 0 errors, 0 leaks.

**1080p video.** `people-walking.mp4` upscaled to 1920×1080 (596 frames), `rfdetr-seg-medium`, CPU
path vs `--gpu-preprocess --gpu-postprocess`: both completed. Frames 200 and 350 (people present)
show the same detections and scores (0.95/0.92, 0.96/0.92) and clean masks.

`dog.jpg` has no person, so its keypoint run finds 0 / 0 detections; `bus.jpg` (810×1080) is the
keypoint input. The output images were compared side by side and look identical by eye. The only
visible difference is a small change in the bicycle mask outline in `seg dog.jpg`.

## Benchmarks (with a device)

`./build/benchmarks --benchmark_filter='Cuda|Cpu'`, 432 and 576, same card as Phase 4.

| Stage | Gate | Phase 4 DALI reference |
|-------|------|------------------------|
| `BM_CudaPreprocessFrame` (H2D + kernel) | ≤ DALI reference | 4.51 ms (432) / 4.59 ms (576) |
| `BM_CudaPreprocessEncoded` (nvJPEG + kernel) | ≤ DALI reference | 4.51 ms (432) / 4.59 ms (576) |
| CPU preprocess | unchanged ± 10 % | 6.75 ms / 11.53 ms |

A regression past the DALI reference is not a silent pass: record it and decide before merge.

### Benchmark record (2026-10-01, group B)

Same card and image as the group B record. A 1280×720 source; the encoded cases use a q95 JPEG. Every
row is the median of 5 repetitions. DALI was run in the same binary for a like-for-like comparison,
using a working `/opt/dali` (with nvImageCodec) taken from the `rfdetr-trt-gate:ffmpeg-on` image and
mounted over the builder's.

| Stage | 432 | 576 | DALI, same run (432 / 576) | Gate |
|-------|-----|-----|----------------------------|------|
| `BM_CudaPreprocessFrame` (H2D + kernel) | 0.564 ms | 0.577 ms | `BM_DaliPreprocessFrame` 0.975 / 0.930 ms | PASS |
| `BM_CudaPreprocessEncoded` (nvJPEG + kernel) | 4.26 ms | 4.19 ms | `BM_DaliPreprocess` 4.98 / 4.95 ms | PASS |
| `BM_CpuPreprocess` | 6.82 ms | 13.6 ms | — | 432 within ± 10 % of 6.75 ms; 576 had no prior 576 row (the old benchmark ran 560: 13.0 ms today) |

Speedup over the CPU path, from a second run on the same day. Each GPU case is compared with the CPU
work it replaces, including the transfer the CPU path still needs:

| Path | CPU baseline (432 / 576) | CUDA (432 / 576) | Speedup |
|------|--------------------------|------------------|---------|
| Frame: `BM_CpuPreprocessUpload` (CPU preprocess + H2D of the float tensor) vs `BM_CudaPreprocessFrame` | 6.81 / 11.9 ms | 0.552 / 0.558 ms | 12× / 21× |
| Still JPEG: `BM_CpuPreprocessEncoded` (stb decode from file + CPU preprocess) vs `BM_CudaPreprocessEncoded` | 20.4 / 25.4 ms | 4.02 / 3.98 ms | 5.1× / 6.4× |

`BM_CpuPreprocessEncoded` reads the JPEG from a file in the page cache, while the CUDA case starts from
bytes already in memory. It also leaves out the 0.5–0.9 ms tensor upload that the CPU path would still need,
so the still-JPEG speedup is understated, not inflated.
`BM_CpuPreprocess`/576 measured 13.6 ms in the first run and 11.0 ms in the second, so expect run-to-run
noise of roughly ±15 % on this laptop's CPU figures.

The encoded path is bound by the host: the default nvJPEG backend does Huffman decoding on the calling
thread, which accounts for ~4.2 ms of CPU time per image. DALI's ~0.7 ms "CPU" figure only counts the
calling thread; its Huffman stage runs in DALI's worker threads. On this metric DALI used the calling
thread less, not the machine.

## Manual

- [x] Detection, segmentation and keypoint output images from a real model look identical, by
      eye, between `--gpu-preprocess` and the CPU path on one JPEG and one PNG (2026-10-02, group
      C/D record)
- [ ] A 1080p video with `--gpu-preprocess --gpu-postprocess --display` plays without artefacts.
      **UNRUN for `--display`**: this session has no display. A 1920×1080 run without it (596 frames, upscaled
      `people-walking.mp4`) completed on both paths. Its output frames were compared CPU vs GPU and
      show the same people, the same scores and no artefacts (group C/D record)
- [x] The default ONNX Runtime (CPU) build's results are bit-identical to `develop` before this phase
      (2026-10-02: `rfdetr-nano` detection and `rfdetr-seg-medium` segmentation on `dog.jpg`,
      printed results and output image bytes identical, `develop` at `a32a2ff`)
- [x] Docker gate: `dockerfile.trt` builds for the new and changed `GPU_PIPELINE` values (`pre`,
      `on`, `dali-on`, at least one media backend each); `GPU_PIPELINE=cuda` fails with the
      message naming `post` (2026-10-02: `ffmpeg` × `off|pre|post|on|dali|dali-on` and `opencv` ×
      `on|dali-on` all built. `cuda` was rejected with the rename message. Six images ran with
      `--gpus all` on `dog.jpg`: CUDA images report "GPU preprocessing: CUDA", DALI images "DALI",
      4 instances each)
- [x] `README.md`, `AGENTS.md` and `docs/` present CUDA as the default GPU preprocessor and DALI as
      the configure-time alternative, with no instruction that enables both (2026-10-02)

## Definition of done

- [x] `CHANGELOG.md` `[Unreleased]` updated in house style, naming the verification hardware,
      driver, CUDA and TensorRT versions (2026-10-02). The file has no per-file tables in any
      entry, so the entries are prose like the rest.
- [ ] Roadmap Phase 6 items ticked and heading marked `(Complete)`
- [x] `specs/mission.md`/`specs/tech-stack.md` changes landed in the same commit as `README.md`/`AGENTS.md`
- [ ] Branch merged into `develop` and deleted
