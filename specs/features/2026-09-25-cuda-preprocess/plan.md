# Plan — Phase 6, CUDA preprocessing, DALI removed

Base: `develop` after `feature/tensorrt-11-support` merges. Branch
`feature/phase-6-cuda-preprocess`. The tree builds and CI stays green after every group.

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

9. `src/rfdetr_inference.hpp:66,183-184` — replace `dali_encoded_`/`dali_frame_` with
   `std::unique_ptr<rfdetr::gpu::JpegDecoder> jpeg_decoder_`; drop `dali_pipeline_dir`.
10. `src/rfdetr_inference.cpp:571-686` — `ensure_gpu_ready()` creates the decoder;
    `run_gpu_image()` reads the bytes, `probe()`s, then nvJPEG-decodes into `frame_device_` or
    falls back to `load_image` + `copy_h2d`; both then call `preprocess_bgr_device` into the input
    binding and `run_inference_device`. `run_gpu_frame()` keeps its H2D and calls the kernel.
    Error text for "built without" names `-DUSE_CUDA_PREPROCESS=ON`.
11. `src/main.cpp:147,178,253` — remove `--dali-pipeline-dir` and its help text.
12. `tests/integration/integration_test_gpu_parity.cpp:29,131-171` — guard on
    `USE_CUDA_PREPROCESS && USE_CUDA_POSTPROCESS`; drop the `.dali` existence skip; tolerances
    unchanged (`0.03`/`0.95` for the nvJPEG combination, `1e-3`/`0.999` for the rest).
13. `tests/benchmark/bench_gpu_pipeline.cpp:17-112` — `BM_CudaPreprocessFrame` (H2D + kernel) and
    `BM_CudaPreprocessEncoded` (nvJPEG + kernel), `Arg(432)->Arg(576)`.

## Group D — Remove DALI (no GPU)

14. Delete `src/gpu/dali_preprocessor.{hpp,cpp}`, `deploy/dali/`, `data/dali/`,
    `scripts/fetch_dali.sh` (with its nvImageCodec staging), `scripts/generate_dali_pipelines.sh`, `cmake/deps/packages/DALI.cmake`.
15. `CMakeLists.txt` — `USE_DALI` becomes `if(USE_DALI) message(FATAL_ERROR "…use -DUSE_CUDA_PREPROCESS=ON")`;
    remove `DALI_ROOT` handling, `Deps::DALI`, `_DALI_RPATH_DIRS` (`:335-337`) and
    `GPU_PARITY_DALI_DIR` (`:413,459`). `CMakePresets.json:59-67` drops `DALI_ROOT` and renames
    the description.
16. `cmake/deps/strategies/ProvidedPackageManager.cmake` — drop DALI-specific handling, if any.
17. `versions.env`, `cmake/versions.cmake`, `scripts/versions.sh` — remove `DALI_VERSION` and
    `TRITON_IMAGE`; `NGC_CONTAINER_TAG` stays for `TENSORRT_IMAGE` and the Docker base. `scripts/check_version_sync.sh` — drop the DALI README
    checks. Run it.
18. `scripts/ci/stage_gpu_headers.sh` — remove the DALI wheel staging; `gpu-compile.yml` — install
    `libnvjpeg-dev-<cuda>` beside `cuda-cudart-dev` (`:69`), matrix entries (`:32-38`) become the
    four `USE_CUDA_PREPROCESS`×`USE_CUDA_POSTPROCESS` combinations; validate the YAML.
19. `dockerfile.trt` — remove `dali-none`/`dali-fetch` (with its nvImageCodec copy)/`dali-selected` and `/opt/dali`;
    `GPU_PIPELINE=off|pre|post|on`, `dali`/`cuda` rejected with a message. Run
    `./scripts/check_dockerfile_parity.sh`.
20. `scripts/run_gate.sh` — drop DALI staging and pipeline checks.

## Group E — Constitution and docs (no GPU)

21. `specs/gpu-pipeline.md` — architecture diagram (nvJPEG / H2D → fused kernel), rule 8 rewrite,
    correctness rules and risks per requirements.md.
22. `specs/mission.md`, `specs/tech-stack.md` (GPU stack rows, CMake options table, constraints,
    CI coverage, loader consumers) — **same commit** as `README.md`, `AGENTS.md` (GPU Pipeline
    section: no DALI staging step, new option) and any open spec.
23. `docs/advanced-usage.md`, `docs/building.md`, `docs/architecture.md`, `docs/docker.md`,
    `docs/usage.md`, `docs/development.md`, `specs/rented-gpu-runbook.md`,
    `.claude/skills/gpu-verify/SKILL.md`, `tests/data/gpu_parity/README.md`.
24. `CHANGELOG.md` `[Unreleased]` — Removed (DALI, `--dali-pipeline-dir`, `USE_DALI`, the pins),
    Added (`USE_CUDA_PREPROCESS`, nvJPEG), Changed (`GPU_PIPELINE` values, tolerances), with the
    per-file table.

## Group F — Gate (GPU)

25. Docker gate: every `dockerfile.trt` `MEDIA_BACKEND` × `GPU_PIPELINE` combination builds;
    `dali`/`cuda` fail with the message.
26. [gpu-verify](../../../.claude/skills/gpu-verify/SKILL.md) on the RTX 3060 inside the builder
    image: everything in validation.md's "with a device" section. Record results in validation.md.
27. Tick roadmap Phase 6, mark it `(Complete)`, merge to `develop`, delete the branch.
