# Phase 6 — CUDA preprocessing, DALI as the alternative

Implements [roadmap.md](../../roadmap.md) Phase 6. Adds our own CUDA preprocessing to the GPU
pipeline (a fused kernel plus nvJPEG for JPEG decode), so the default GPU pipeline is CUDA end to end
with the existing segmentation postprocessing kernels. DALI preprocessing stays in the tree as a
configure-time alternative to the CUDA preprocessor, never compiled alongside it.

**Scope change, 2026-10-02.** As first written (interview 2026-09-25), this phase removed DALI. The
user reversed that during implementation: "dali not to remove … but alternative". DALI, its staging,
its `.dali` pipelines, its pins and its Docker stages therefore stay. The "exclusive at configure"
decision from the interview now governs the two preprocessors. Groups A and B are unaffected. Group
C keeps the DALI code paths, group D becomes "DALI as the alternative", and the removal-only checks
in validation.md are replaced.

**Based on `develop` after `feature/tensorrt-11-support` (PR #16, `a32a2ff`).** That branch
left one pinned stack — TensorRT 11 (`TENSORRT_VERSION`) and DALI 2.x (`DALI_VERSION`) — with no
compat or legacy versions, and made DALI 2.x staging copy nvImageCodec and its codec libraries,
which DALI `dlopen()`s at decode time (`scripts/fetch_dali.sh:37-47`, `dockerfile.trt:88-102`).
All of that is unchanged by this phase.

## Scope

### In

| Deliverable | Path |
|-------------|------|
| Fused preprocessing kernel: BGR interleaved `uint8` → RGB planar `float` NCHW, bilinear stretch, ImageNet normalise, written into the TensorRT input binding on the context stream | `src/gpu/rfdetr_preprocess.cu`, `src/gpu/rfdetr_preprocess.hpp` (new) |
| nvJPEG decoder: JPEG bytes → interleaved BGR on the device, on the context stream | `src/gpu/jpeg_decoder.cpp`, `src/gpu/jpeg_decoder.hpp` (new) |
| Orchestrator: `run_gpu_image` (nvJPEG, or stb fallback + upload for non-JPEG) and `run_gpu_frame` (upload + kernel) under `USE_CUDA_PREPROCESS`, with the DALI paths kept under `USE_DALI` | `src/rfdetr_inference.cpp`, `src/rfdetr_inference.hpp` |
| CLI: `--gpu-preprocess` uses whichever preprocessor is compiled in; `--dali-pipeline-dir` kept for DALI builds | `src/main.cpp` |
| CMake: `USE_CUDA_PREPROCESS` option, `CUDA::nvjpeg` link, `USE_DALI` + `USE_CUDA_PREPROCESS` a configure `FATAL_ERROR`; `USE_GPU_PIPELINE` enables CUDA preprocessing unless `USE_DALI` is given | `CMakeLists.txt`, `CMakePresets.json` (`gpu-pipeline` → CUDA, new `gpu-pipeline-dali`) |
| Docker: `GPU_PIPELINE` values `off\|pre\|post\|on\|dali\|dali-on`; `cuda` rejected with a message naming `post` | `dockerfile.trt` |
| CI: `libnvjpeg-dev` installed; compile matrix covers TensorRT alone, each preprocessor, CUDA postprocess, and both full pipelines; a step asserts the exclusive configure error | `.github/workflows/gpu-compile.yml` |
| Parity tests for the CUDA preprocessor (frame, encoded, PNG fallback, probe), DALI tests kept; integration test runs against either preprocessor, plus a PNG-fallback end-to-end case for CUDA | `tests/unit/test_gpu_parity.cpp`, `tests/integration/integration_test_gpu_parity.cpp` |
| Benchmarks: `BM_CudaPreprocessFrame`, `BM_CudaPreprocessEncoded`, CPU baselines with decode / upload; DALI benchmarks kept | `tests/benchmark/bench_gpu_pipeline.cpp` |
| Constitution and docs | `specs/gpu-pipeline.md`, `specs/mission.md`, `specs/tech-stack.md`, `specs/roadmap.md`, `.claude/skills/gpu-verify/SKILL.md`, `AGENTS.md`, `README.md`, `docs/*.md`, `CHANGELOG.md` |

### Out

- Removing DALI (reversed 2026-10-02, see above).
- nvJPEG hardware backend (`NVJPEG_BACKEND_HARDWARE`) — the RTX 3060 verification card has no
  hardware JPEG decoder; the default backend is used. Revisit only on an A100/H100/Jetson target.
- nvJPEG GPU-hybrid Huffman decode (decoupled API). The default backend's host Huffman stage is
  ~4 ms of the encoded path's time; worth a separate phase only if still-image throughput matters.
- GPU decode for PNG/BMP and other non-JPEG formats — they decode on the CPU with stb and are
  uploaded.
- Batched nvJPEG decode (`nvjpegDecodeBatched`) — batch size is fixed at 1 (roadmap Deferred).
- Stream-per-slot video overlap — the single-stream design and its risk row in
  `specs/gpu-pipeline.md` are unchanged.
- Any change to CUDA postprocessing (`src/gpu/rfdetr_postprocess.cu`).
- Detection/keypoint GPU postprocessing (roadmap Deferred).

## Decisions

| Decision | Reason |
|----------|--------|
| **DALI stays as a configure-time alternative**, not removed. | User decision, 2026-10-02, reversing the 2026-09-25 removal. |
| **`USE_DALI` and `USE_CUDA_PREPROCESS` are exclusive at configure** (`FATAL_ERROR` naming both). | The interview's "exclusive at configure" choice. One preprocessor per binary keeps `run_gpu_image`/`run_gpu_frame` a compile-time branch and the parity matrix finite. |
| **CUDA is the default**: `USE_GPU_PIPELINE=ON` enables `USE_CUDA_PREPROCESS` + `USE_CUDA_POSTPROCESS`, or DALI + CUDA postprocess when `-DUSE_DALI=ON` is also given. | The CUDA path needs no container-extracted distribution, `.dali` files or `dlopen()`ed codecs, holds `1e-5` frame parity, and measured faster than DALI on both paths (validation.md). |
| **nvJPEG for JPEG, stb + upload for everything else.** Format chosen by `nvjpegGetImageInfo` success, not file extension. | User decision. nvJPEG ships in the CUDA Toolkit (`CUDA::nvjpeg`), so no new pin; approved in the interview. Magic bytes are authoritative; extensions lie. |
| **nvJPEG decodes to `NVJPEG_OUTPUT_BGRI`**, feeding the same kernel as the frame path. | One kernel, one parity contract. The decode is the only difference between the image and frame paths, matching today's fixture split. |
| **Image dimensions come from `nvjpegGetImageInfo`**, not a host decode. | The DALI path decodes the whole image on the CPU just to read its size; the header read removes that for the CUDA path. |
| **The nvJPEG decoder is created on the first still image**, owned per `RFDETRInference`. | nvJPEG decode state is not thread-safe, and a video run never decodes JPEG, so it never creates one (open question 2). |
| **Frame-path tensor tolerance: max abs `1e-5`** vs `preprocess_bgr_image`. | User decision. The kernel reproduces the CPU arithmetic exactly (`src/media.cpp:202-247`, `src/processing_utils.cpp:15-24`); only FMA contraction and division rounding may differ. DALI keeps its `2e-2`. |
| **Encoded (nvJPEG) tolerances unchanged**: tensor max abs `1e-1`, end-to-end score `0.03`, mask IoU `0.95`. | User accepted the measured decoder gap (nvJPEG vs stb IDCT/upsampling) as not worth chasing. It is a decoder property, shared by both preprocessors. |
| **PNG fallback tensor tolerance: max abs `1e-5`.** | stb decodes on both sides, so the tensor goes through the frame-path kernel from identical pixels. |
| **End-to-end, any GPU preprocessor vs CPU: score `0.06`, box centre 1% of the longer image side, mask IoU `0.95`**; identical-input comparisons keep `1e-3` / 1 px / `0.999`. Replaces the encoded `0.03` / 1 px and the PNG `1e-3` / `0.999` above. | Measured 2026-10-02, group C. The engine amplifies input noise: one tensor element nudged by `1e-6` moved `rfdetr-seg-medium`'s logits by up to 8.3, while host and device paths given the same tensor were bit-identical. A ~1e-6 tensor (the CUDA kernel) therefore moved scores by up to `0.013` and centres by 1.4 px. The prior bound also failed for DALI on this stack (centre 2.9 px). The tight tensor gates stay in the unit tests. |
| **Kernel math mirrors the CPU order of operations**: `clamp_source_coord`, `y0 = min(int(src_y), h-1)`, the nested `(p00*(1-wx)+p01*wx)*(1-wy)+(…)*wy`, `/255.0f`, then `(v-mean)/std` with the plain ImageNet `means`/`stds`. No folded `mean*255`. | Rule 2 of the model contract; folding changes rounding and would cost the `1e-5` gate. |
| **`dockerfile.trt` `GPU_PIPELINE` values become `off\|pre\|post\|on\|dali\|dali-on`**; `cuda` fails the build naming `post`. | `on` follows `USE_GPU_PIPELINE` (all-CUDA); `cuda` would be ambiguous now that both halves are CUDA, and failing beats silently mapping it. |
| **One stream, no intermediate sync** — H2D (or nvJPEG), kernel, `enqueueV3`, postprocess all on `backend_->device_stream()`. | `specs/gpu-pipeline.md` Architecture. |
| **Fixtures stay as they are** (432, JPEG `small`/`wide`/`tall`, `dense`); no regeneration. | They are CPU-produced and independent of either preprocessor. |

## Context

- **Architectural commitments** (`specs/mission.md`): exactly one backend; GPU tests `GTEST_SKIP()`
  without a device; CPU and GPU stay numerically in step — the CPU preprocess is the reference and
  is **not** changed by this phase; a change to either side is mirrored.
- **Model contract** (`specs/gpu-pipeline.md`): rules 1 (no letterbox) and 2 (ImageNet
  normalisation) define the kernel. Rule 8 ("DALI hosts preprocessing only") is rewritten: GPU
  preprocessing is either our own kernel + nvJPEG (default) or a DALI pipeline, chosen at configure,
  and postprocessing is our own kernels. The DALI correctness rules and risk rows stay and apply to
  DALI builds only; "resize is a tolerance gate" gains the CUDA path's `1e-5`. Any `mission.md` or
  `tech-stack.md` edit propagates to `README.md`, `AGENTS.md` and open specs in the same commit.
- **Patterns to follow:**
  - Kernel/host split and `CUDA_CHECK`: `src/gpu/rfdetr_postprocess.cu`, `src/gpu/cuda_check.hpp`.
  - Grow-only device buffer for the uploaded frame: `frame_device_` in `run_gpu_frame`
    (`src/rfdetr_inference.cpp:670-673`); reuse it for the nvJPEG output and the PNG fallback.
  - Device-less skips: `SKIP_WITHOUT_GPU` in `tests/unit/gpu_test_utils.hpp`.
  - Build guards: `gpu_context.hpp:3`, `video_pipeline.cpp:196-218` and `rfdetr_inference.cpp:571`
    test `USE_CUDA_POSTPROCESS || USE_CUDA_PREPROCESS || USE_DALI`.
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
     created lazily like `dali_encoded_`. Confirm it is not shared across the video
     pipeline's threads. **Resolved:** created on the first `run_gpu_image()`; the video path
     calls only `run_gpu_frame()`, which never touches it.
  3. CMYK / greyscale JPEGs: nvJPEG converts to BGRI; confirm on one greyscale fixture that the
     CPU path (stb, `req_comp=3`) agrees within the encoded tolerance.
