# Phase 6 — CUDA preprocessing, DALI removed

Implements [roadmap.md](../../roadmap.md) Phase 6. Replaces the DALI preprocessing half of the GPU
pipeline with our own CUDA code, so the GPU pipeline is CUDA end to end: a fused preprocessing
kernel, nvJPEG for JPEG decode, and the existing segmentation postprocessing kernels. DALI, its
staging, its serialized pipelines and its version pins are removed in the same phase.

**Based on `develop` after `feature/tensorrt-11-support` (PR #16, `a32a2ff`).** That branch
left one pinned stack — TensorRT 11 (`TENSORRT_VERSION`) and DALI 2.x (`DALI_VERSION`) — with no
compat or legacy versions, and made DALI 2.x staging copy nvImageCodec and its codec libraries,
which DALI `dlopen()`s at decode time (`scripts/fetch_dali.sh:37-47`, `dockerfile.trt:88-102`).
This phase removes all of that.

## Scope

### In

| Deliverable | Path |
|-------------|------|
| Fused preprocessing kernel: BGR interleaved `uint8` → RGB planar `float` NCHW, bilinear stretch, ImageNet normalise, written into the TensorRT input binding on the context stream | `src/gpu/rfdetr_preprocess.cu`, `src/gpu/rfdetr_preprocess.hpp` (new) |
| nvJPEG decoder: JPEG bytes → interleaved BGR on the device, on the context stream | `src/gpu/jpeg_decoder.cpp`, `src/gpu/jpeg_decoder.hpp` (new) |
| Orchestrator wiring: `run_gpu_image` (nvJPEG, or stb fallback + upload for non-JPEG) and `run_gpu_frame` (upload + kernel) | `src/rfdetr_inference.cpp:571-686`, `src/rfdetr_inference.hpp:66,183-184` |
| CLI: `--dali-pipeline-dir` removed | `src/main.cpp:147,178,253` |
| CMake: `USE_CUDA_PREPROCESS` option, `USE_GPU_PIPELINE` enabling both CUDA halves, `CUDA::nvjpeg` link, `USE_DALI` turned into a `FATAL_ERROR` that names the replacement | `CMakeLists.txt:114-203,249-253,335-363,395-459`, `CMakePresets.json:59-67` |
| DALI removed | `src/gpu/dali_preprocessor.{hpp,cpp}`, `deploy/dali/`, `data/dali/`, `scripts/fetch_dali.sh`, `scripts/generate_dali_pipelines.sh`, `cmake/deps/packages/DALI.cmake` |
| Pins: `DALI_VERSION` and the derived `TRITON_IMAGE` removed; README DALI rows and their sync checks removed | `versions.env`, `cmake/versions.cmake`, `scripts/versions.sh`, `scripts/check_version_sync.sh`, `README.md` |
| Docker: DALI and nvImageCodec staging stages removed; `GPU_PIPELINE` values renamed (see Decisions) | `dockerfile.trt` |
| CI: DALI header staging replaced by `libnvjpeg-dev`; matrix covers the four `USE_CUDA_PREPROCESS`/`USE_CUDA_POSTPROCESS` combinations on the pinned TensorRT | `scripts/ci/stage_gpu_headers.sh`, `.github/workflows/gpu-compile.yml` |
| Parity tests retargeted from DALI to the CUDA preprocessor, plus a PNG-fallback test | `tests/unit/test_gpu_parity.cpp`, `tests/integration/integration_test_gpu_parity.cpp`, `tests/unit/gpu_parity_fixtures.hpp`, `tests/data/gpu_parity/README.md` |
| Benchmarks: `BM_DaliPreprocess` replaced by CUDA frame preprocess and nvJPEG encoded preprocess | `tests/benchmark/bench_gpu_pipeline.cpp` |
| Gate script: DALI steps removed | `scripts/run_gate.sh` |
| Constitution and docs | `specs/gpu-pipeline.md`, `specs/mission.md`, `specs/tech-stack.md`, `specs/roadmap.md`, `specs/rented-gpu-runbook.md`, `.claude/skills/gpu-verify/SKILL.md`, `AGENTS.md`, `README.md`, `docs/*.md`, `CHANGELOG.md` |

### Out

- nvJPEG hardware backend (`NVJPEG_BACKEND_HARDWARE`) — the RTX 3060 verification card has no
  hardware JPEG decoder; the default backend is used. Revisit only on an A100/H100/Jetson target.
- GPU decode for PNG/BMP and other non-JPEG formats — they decode on the CPU with stb and are
  uploaded.
- Batched nvJPEG decode (`nvjpegDecodeBatched`) — batch size is fixed at 1 (roadmap Deferred).
- Stream-per-slot video overlap — the single-stream design and its risk row in
  `specs/gpu-pipeline.md` are unchanged.
- Any change to CUDA postprocessing (`src/gpu/rfdetr_postprocess.cu`) beyond build-guard renames.
- Detection/keypoint GPU postprocessing (roadmap Deferred).

## Decisions

| Decision | Reason |
|----------|--------|
| **DALI is removed in this phase**, not kept beside the CUDA path. | User decision (interview 2026-09-25). DALI brings a container-extracted C++ distribution, version-coupled `.dali` files, a pin, a derived container image and, since 2.x, a set of `dlopen()`ed codec libraries that no linker check sees (the `libnvimgcodec.so` abort fixed in `1d2b408`); the work it does is one kernel plus a JPEG decode. |
| **`USE_DALI=ON` is a configure `FATAL_ERROR`** naming `-DUSE_CUDA_PREPROCESS=ON`, for one release. | The interview chose "exclusive at configure" for the two preprocessors; with DALI gone the only remaining case is an old command line, which must fail loudly rather than build without GPU preprocessing. Remove the stub after the next release. |
| **Option `USE_CUDA_PREPROCESS`**, independent of `USE_CUDA_POSTPROCESS`; `USE_GPU_PIPELINE` enables both. Both require `USE_TENSORRT` (existing `FATAL_ERROR` kept). | Keeps the existing two-halves shape (`CMakeLists.txt:114-131`) and the four-combination parity matrix. Both halves now need nvcc, so `enable_language(CUDA)` moves to "either". |
| **nvJPEG for JPEG, stb + upload for everything else.** Format chosen by `nvjpegGetImageInfo` success, not file extension. | User decision. nvJPEG ships in the CUDA Toolkit (`CUDA::nvjpeg`), so no new pin; approved in the interview. Magic bytes are authoritative; extensions lie. |
| **nvJPEG decodes to `NVJPEG_OUTPUT_BGRI`**, feeding the same kernel as the frame path. | One kernel, one parity contract. The decode is the only difference between the image and frame paths, matching today's fixture split. |
| **Image dimensions come from `nvjpegGetImageInfo`**, not a host decode. | `run_gpu_image` currently decodes the whole image on the CPU just to read its size (`src/rfdetr_inference.cpp:609-617`). The header read removes that decode. |
| **Frame-path tensor tolerance: max abs `1e-5`** vs `preprocess_bgr_image`. | User decision. The kernel reproduces the CPU arithmetic exactly (`src/media.cpp:202-247`, `src/processing_utils.cpp:15-24`); only FMA contraction and division rounding may differ. Replaces DALI's `2e-2`. |
| **Encoded (nvJPEG) tolerances unchanged**: tensor max abs `1e-1`, end-to-end score `0.03`, mask IoU `0.95`. | User accepted the measured `0.050-0.069` decoder gap (nvJPEG vs stb IDCT/upsampling) as not worth chasing. It is a decoder property, not a DALI one. |
| **PNG fallback tolerance: max abs `1e-5`.** | stb decodes on both sides, so the tensor goes through the frame-path kernel from identical pixels. |
| **Kernel math mirrors the CPU order of operations**: `clamp_source_coord`, `y0 = min(int(src_y), h-1)`, the nested `(p00*(1-wx)+p01*wx)*(1-wy)+(…)*wy`, `/255.0f`, then `(v-mean)/std` with the plain ImageNet `means`/`stds`. No folded `mean*255`. | Rule 2 of the model contract; folding changes rounding and would cost the `1e-5` gate. |
| **`--dali-pipeline-dir` is removed**, not deprecated. `Config::dali_pipeline_dir` goes too. | Nothing reads it. An unknown flag already errors in `main.cpp`. |
| **`dockerfile.trt` `GPU_PIPELINE` values become `off\|pre\|post\|on`**; `dali` and `cuda` fail the build with a message naming the replacement. | `dali` names a removed library and `cuda` would now be ambiguous (both halves are CUDA). Failing is better than silently mapping `cuda` to post-only. |
| **One stream, no intermediate sync** — H2D (or nvJPEG), kernel, `enqueueV3`, postprocess all on `backend_->device_stream()`. | `specs/gpu-pipeline.md` Architecture. nvJPEG's `nvjpegDecode` takes the stream; its pinned host staging is owned by the decoder object. |
| **Fixtures stay as they are** (432, JPEG `small`/`wide`/`tall`, `dense`); no regeneration. | They are CPU-produced and independent of DALI. The "only resolution with a `.dali` pipeline" rationale in the README is rewritten; the resolution stays. |

## Context

- **Architectural commitments** (`specs/mission.md`): exactly one backend; GPU tests `GTEST_SKIP()`
  without a device; CPU and GPU stay numerically in step — the CPU preprocess is the reference and
  is **not** changed by this phase; a change to either side is mirrored.
- **Model contract** (`specs/gpu-pipeline.md`): rules 1 (no letterbox) and 2 (ImageNet
  normalisation) define the kernel. Rule 8 ("DALI hosts preprocessing only") is rewritten to say
  preprocessing and postprocessing are our own kernels in `src/gpu/`, with no external operator
  framework. The `daliOutputRelease` and `antialias=False` correctness rules and the DALI risk rows
  are removed; "resize is a tolerance gate" becomes the `1e-5` gate above. Any `mission.md` or
  `tech-stack.md` edit propagates to `README.md`, `AGENTS.md` and open specs in the same commit.
- **Patterns to follow:**
  - Kernel/host split and `CUDA_CHECK`: `src/gpu/rfdetr_postprocess.cu`, `src/gpu/cuda_check.hpp`.
  - Grow-only device buffer for the uploaded frame: `frame_device_` in `run_gpu_frame`
    (`src/rfdetr_inference.cpp:670-673`); reuse it for the nvJPEG output and the PNG fallback.
  - Device-less skips: `SKIP_WITHOUT_GPU` in `tests/unit/gpu_test_utils.hpp`.
  - Build guards: `gpu_context.hpp:3`, `video_pipeline.cpp:196-218` and `rfdetr_inference.cpp:571`
    test `USE_CUDA_POSTPROCESS || USE_DALI`; they become `USE_CUDA_POSTPROCESS || USE_CUDA_PREPROCESS`.
- **Constraints:** CI has no GPU (compile-only, stub `.so` files); `rfdetr_inference_lib` must
  compile with `-DWERROR=ON` against staged headers for the pinned TensorRT 11 (the only one; 10.x
  is rejected at configure).
  nvJPEG's header needs `libnvjpeg-dev-<cuda>` in `gpu-compile.yml`'s apt step
  (`.github/workflows/gpu-compile.yml:69`).
- **Verification hardware:** the local RTX 3060 Laptop (sm_86), run inside the `dockerfile.trt`
  builder image with `--gpus all` on the pinned NGC stack. `compute-sanitizer` comes from that
  image, as in Phase 4.
- **Open questions, resolve during implementation:**
  1. Whether `-fmad=false` (or `__fmul_rn`/`__fadd_rn`) is needed to hold `1e-5`. Try the default
     first and record the measured max delta.
  2. nvJPEG handle lifetime: one `nvjpegHandle_t` + `nvjpegJpegState_t` per `RFDETRInference`,
     created lazily like today's `dali_encoded_`. Confirm it is not shared across the video
     pipeline's threads.
  3. CMYK / greyscale JPEGs: nvJPEG converts to BGRI; confirm on one greyscale fixture that the
     CPU path (stb, `req_comp=3`) agrees within the encoded tolerance.
