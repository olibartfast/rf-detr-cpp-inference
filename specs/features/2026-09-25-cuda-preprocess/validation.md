# Validation — Phase 6, CUDA preprocessing, DALI removed

Every result is recorded here with the date and the hardware; anything not run is `UNRUN` with
the reason, never a pass.

## Automated — no GPU (CI)

- [ ] `./scripts/check_version_sync.sh` passes with the DALI pins gone
- [ ] `./scripts/check_dockerfile_parity.sh` passes
- [ ] Format: `find src tests -name '*.cpp' -o -name '*.hpp' | xargs clang-format-18 --dry-run --Werror`
- [ ] Clang-tidy: `cmake -S . -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON` then
      `find src -name '*.cpp' ! -name 'tensorrt_backend.cpp' | xargs clang-tidy-18 -p build`
- [ ] Cppcheck: `cppcheck --enable=all --std=c++20 --suppress=missingIncludeSystem --suppress=unmatchedSuppression --suppress=unusedFunction --error-exitcode=1 -I src src/`
- [ ] Default build and tests: `cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build build --parallel`,
      `ctest --test-dir build --output-on-failure -R UnitTests`
- [ ] `gpu-compile.yml` green on every matrix entry: TensorRT alone, `+USE_CUDA_PREPROCESS`,
      `+USE_CUDA_POSTPROCESS`, `+both` — all `-DWERROR=ON` on the pinned TensorRT
- [ ] `grep -rniI dali` over the tree returns only `CHANGELOG.md` history, completed spec
      directories, and the `USE_DALI` `FATAL_ERROR` stub

## Compile-without-device

- [ ] `-DUSE_ONNX_RUNTIME=OFF -DUSE_TENSORRT=ON -DUSE_GPU_PIPELINE=ON -DWERROR=ON` builds
      `inference_app`, `unit_tests`, `integration_tests` and `benchmarks` on a machine with the
      CUDA Toolkit and TensorRT but no device
- [ ] On that machine every `GpuParity*` and `GpuPostprocess*` test reports `SKIPPED`, none `FAILED`
- [ ] `-DUSE_DALI=ON` fails at configure with a message naming `-DUSE_CUDA_PREPROCESS=ON`

## Automated — with a device (RTX 3060 Laptop, sm_86, builder image, `--gpus all`)

Tensor tolerances are max abs over the full `[1,3,432,432]` tensor against `*.preprocessed.bin`.

| Check | Tolerance | Result |
|-------|-----------|--------|
| Frame path (`preprocess_bgr_device`), `small`/`wide`/`tall` | ≤ `1e-5` | PASS 2026-09-27 (group A): `1.07e-6`/`1.07e-6`/`8.34e-7` |
| PNG fallback path, `small`/`wide`/`tall` | ≤ `1e-5` | |
| Encoded path (nvJPEG + kernel), `small`/`wide`/`tall` | ≤ `1e-1` | |
| No letterbox on `wide` and `tall` | pass | PASS 2026-09-27, frame path (encoded path: group B) |
| End-to-end, `cpupre-gpupost` vs `cpu-cpu` | score `1e-3`, mask IoU ≥ `0.999` | |
| End-to-end, `gpupre-cpupost` (nvJPEG) vs `cpu-cpu` | score `0.03`, mask IoU ≥ `0.95` | |
| End-to-end, `gpu-gpu` vs `gpupre-cpupost` | score `1e-3`, mask IoU ≥ `0.999` | |
| `dense` fixture | 200 detections, count exact | |
| Real model, all three tasks with `--gpu-preprocess` (and `--gpu-postprocess` for segmentation) | integration tolerances above | |
| 1000-frame video, `--gpu-preprocess --gpu-postprocess`, `compute-sanitizer --tool memcheck` | 0 errors, 0 leaks | |
| Same run, `compute-sanitizer --tool racecheck` on the preprocess kernel | 0 hazards | |

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
  DALI frame-path test passes, and group D deletes these tests.

## Benchmarks (with a device)

`./build/benchmarks --benchmark_filter='Cuda|Cpu'`, 432 and 576, same card as Phase 4.

| Stage | Gate | Phase 4 DALI reference |
|-------|------|------------------------|
| `BM_CudaPreprocessFrame` (H2D + kernel) | ≤ DALI reference | 4.51 ms (432) / 4.59 ms (576) |
| `BM_CudaPreprocessEncoded` (nvJPEG + kernel) | ≤ DALI reference | 4.51 ms (432) / 4.59 ms (576) |
| CPU preprocess | unchanged ± 10 % | 6.75 ms / 11.53 ms |

A regression past the DALI reference is not a silent pass: record it and decide before merge.

## Manual

- [ ] Detection, segmentation and keypoint output images from a real model look identical, by
      eye, between `--gpu-preprocess` and the CPU path on one JPEG and one PNG
- [ ] A 1080p video with `--gpu-preprocess --gpu-postprocess --display` plays without artefacts
- [ ] The default ONNX Runtime (CPU) build's results are bit-identical to `develop` before this phase
- [ ] Docker gate: `dockerfile.trt` × `MEDIA_BACKEND=ffmpeg|opencv` × `GPU_PIPELINE=off|pre|post|on`
      build; `GPU_PIPELINE=dali` and `=cuda` fail with the replacement message
- [ ] `README.md`, `AGENTS.md` and `docs/` contain no DALI build or staging instruction

## Definition of done

- [ ] `CHANGELOG.md` `[Unreleased]` updated in house style (prose + per-file table), naming the
      verification hardware, driver, CUDA, TensorRT versions
- [ ] Roadmap Phase 6 items ticked and heading marked `(Complete)`
- [ ] `specs/mission.md`/`specs/tech-stack.md` changes landed in the same commit as `README.md`/`AGENTS.md`
- [ ] Branch merged into `develop` and deleted
