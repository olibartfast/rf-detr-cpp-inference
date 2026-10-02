# Plan — Phase 6, CUDA preprocessing, DALI as the alternative

Base: `develop` after `feature/tensorrt-11-support` merges. Branch
`feature/phase-6-cuda-preprocess`. The tree builds and CI stays green after every group.

Groups A and B were planned and done before the 2026-10-02 scope change (requirements.md: DALI
stays as a configure-time alternative). Groups C–F below are the post-change plan; the original
"remove DALI" group D is withdrawn.

## Group A — CUDA preprocessing kernel (no GPU to build; GPU to test)

1. `src/gpu/rfdetr_preprocess.hpp` (new) — `rfdetr::gpu::preprocess_bgr_device(const std::uint8_t *bgr,
   int height, int width, float *dst, int resolution, std::span<const float, 3> means,
   std::span<const float, 3> stds, StreamHandle stream)`, guarded by `USE_CUDA_PREPROCESS`.
2. `src/gpu/rfdetr_preprocess.cu` (new) — one thread per output pixel, all three channels; the
   arithmetic of `preprocess_bgr_image` (`src/media.cpp:202-247`) and `normalize_image`
   (`src/processing_utils.cpp:15-24`) line for line. Register in `CMakeLists.txt` beside
   `rfdetr_postprocess.cu` in the `RFDETR_SOURCES` GPU block (`:249-253`).
3. `CMakeLists.txt:114-131` — `option(USE_CUDA_PREPROCESS …)`; `USE_GPU_PIPELINE` sets it with
   `USE_CUDA_POSTPROCESS`; the TensorRT `FATAL_ERROR` covers both; `enable_language(CUDA)` and
   `find_dependency_unified(CUDAToolkit)` when either is on; `target_compile_definitions(… USE_CUDA_PREPROCESS)`
   beside `:362-363`. `USE_DALI` stays working in this group.
4. `src/gpu/gpu_context.hpp:3,79`, `src/video_pipeline.cpp:196,209,218`,
   `src/rfdetr_inference.cpp:571` — add `USE_CUDA_PREPROCESS` to the guards.
5. `tests/unit/test_gpu_parity.cpp` — new group `GpuParityCudaPreprocess`: frame path on
   `small`/`wide`/`tall` vs `*.preprocessed.bin`, max abs ≤ `1e-5`; no-letterbox check on
   `wide`/`tall`. Skips without a device.

## Group B — nvJPEG decode (no GPU to build; GPU to test)

6. `src/gpu/jpeg_decoder.hpp`/`.cpp` (new, plain C++) — `rfdetr::gpu::JpegDecoder`: owns
   `nvjpegHandle_t`/`nvjpegJpegState_t`; `std::optional<ImageSize> probe(std::span<const std::uint8_t>)`
   via `nvjpegGetImageInfo` (empty = not a JPEG nvJPEG accepts); `void decode(bytes, DeviceBuffer &dst,
   StreamHandle)` to `NVJPEG_OUTPUT_BGRI`. Non-copyable, non-movable, pimpl like `DaliPreprocessor`.
   Register in `RFDETR_SOURCES` under `USE_CUDA_PREPROCESS`.
7. `CMakeLists.txt:348-352` — link `CUDA::nvjpeg` under `USE_CUDA_PREPROCESS`.
8. `tests/unit/test_gpu_parity.cpp` — encoded path: nvJPEG + kernel vs `*.preprocessed.bin`, max
   abs ≤ `1e-1`; PNG fallback: write each fixture's stb-decoded image to PNG with
   `rfdetr::media::save_image`, run the fallback path, max abs ≤ `1e-5`.

## Group C — Orchestrator and CLI (no GPU to build; GPU to test)

9. `src/rfdetr_inference.hpp` — add `jpeg_decoder_` (`USE_CUDA_PREPROCESS`) beside the DALI members
   (`USE_DALI`); `encoded_bytes_`/`frame_device_` shared by both; `dali_pipeline_dir` kept.
10. `src/rfdetr_inference.cpp` — `gpu_preprocess_active()` and the GPU guards accept either
    preprocessor; `run_gpu_image()` reads the bytes, `probe()`s, nvJPEG-decodes into
    `frame_device_` or falls back to `load_image` + `copy_h2d`, then the kernel writes the input
    binding; `run_gpu_frame()` uploads and runs the kernel. `#elif defined(USE_DALI)` keeps the DALI
    bodies unchanged. "Built without" errors name both options.
11. `src/main.cpp` — `--gpu-preprocess` accepted with either option; help text names both;
    `--dali-pipeline-dir` documented as DALI-only.
12. `tests/integration/integration_test_gpu_parity.cpp` — guard on
    `(USE_DALI || USE_CUDA_PREPROCESS) && USE_CUDA_POSTPROCESS`; `.dali` skip only in DALI builds;
    new `PngFallbackMatchesCpu` for CUDA builds. Registered under the same condition in `CMakeLists.txt`.
13. `tests/benchmark/bench_gpu_pipeline.cpp` — `BM_CudaPreprocessFrame`, `BM_CudaPreprocessEncoded`,
    `BM_CpuPreprocessUpload`, `BM_CpuPreprocessEncoded`, `BM_DaliPreprocessFrame`; `benchmarks` links
    GTest when it includes `gpu_test_utils.hpp`.

## Group D — DALI as the alternative (no GPU)

14. `CMakeLists.txt` — `USE_DALI` + `USE_CUDA_PREPROCESS` → `FATAL_ERROR`; `USE_GPU_PIPELINE` selects
    CUDA preprocessing unless `USE_DALI` is given. `CMakePresets.json` — `gpu-pipeline` is the CUDA
    pipeline (no `DALI_ROOT`), new `gpu-pipeline-dali`.
15. `dockerfile.trt` — `GPU_PIPELINE=off|pre|post|on|dali|dali-on`, `cuda` rejected; DALI staged only
    for `dali|dali-on`. Run `./scripts/check_dockerfile_parity.sh`.
16. `.github/workflows/gpu-compile.yml` — `libnvjpeg-dev-<cuda>`; six matrix entries; a step on the
    TensorRT entry asserts the exclusive configure error.
17. DALI pins, staging scripts, `.dali` pipelines and `scripts/run_gate.sh` stay; the gate script
    gains the CUDA-preprocess build if it hard-codes the DALI one.

## Group E — Constitution and docs (no GPU)

18. `specs/gpu-pipeline.md` — architecture (CUDA default, DALI alternative), rule 8, correctness
    rules scoped per preprocessor.
19. `specs/mission.md`, `specs/tech-stack.md` (CMake options, constraints, CI coverage) — **same
    commit** as `README.md`, `AGENTS.md` and any open spec.
20. `docs/advanced-usage.md`, `docs/building.md`, `docs/docker.md`, `docs/usage.md`,
    `docs/architecture.md`, `.claude/skills/gpu-verify/SKILL.md`, `specs/rented-gpu-runbook.md` where
    they name the GPU options or `GPU_PIPELINE` values.
21. `CHANGELOG.md` `[Unreleased]` — Added (`USE_CUDA_PREPROCESS`, nvJPEG, presets, Docker values),
    Changed (`USE_GPU_PIPELINE`/`GPU_PIPELINE=on` now CUDA, `cuda` rejected), Fixed (`benchmarks` GTest
    link), with the per-file table.

## Group F — Gate (GPU)

22. Build matrix in the builder image, `-DWERROR=ON`: TensorRT alone, each preprocessor alone, CUDA
    postprocess alone, both full pipelines; both preprocessors together fails at configure.
23. Docker gate: `dockerfile.trt` for the changed `GPU_PIPELINE` values; `cuda` fails with the message.
24. [gpu-verify](../../../.claude/skills/gpu-verify/SKILL.md) on the RTX 3060 inside the builder
    image: everything in validation.md's "with a device" section, for the CUDA pipeline and a DALI
    regression run. Record results in validation.md.
25. Tick roadmap Phase 6, mark it `(Complete)`, open the PR to `develop`.
