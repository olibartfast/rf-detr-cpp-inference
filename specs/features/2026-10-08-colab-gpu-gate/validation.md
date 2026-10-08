# Validation: GPU gate on Google Colab

Runs of `scripts/run_gate.sh` on Colab cards, driven by `scripts/colab/gpu_gate.ipynb`. Procedure
and Colab specifics: [rented-gpu-runbook.md](../../rented-gpu-runbook.md#google-colab).

## Tesla T4, 2026-10-08 — `develop` + Colab tooling (`31bbafe`)

Tesla T4 (sm_75, 15 GB), driver 580.82.07 (CUDA 13.0), CUDA toolkit 13.3.73 under minor-version
compatibility (no `cuda-compat`), TensorRT 11.2.1.2, DALI 2.2.0 **from pip** (`nvidia-dali-cuda130`,
not the NGC build). Ubuntu 24.04, 8 vCPUs, High-RAM. Model: rf-detr-seg-medium at 432, exported on
the runtime with rfdetr 1.11.2.

| Check | Result |
|-------|--------|
| Full builds, CUDA pre + CUDA post and DALI pre + CUDA post, sm_75, `-DWERROR=ON` | PASS |
| Four pre/post combinations through `inference_app` | PASS (smoke) |
| `GpuParityIntegration.FourCombinationsAgree` on both builds; `PngFallbackMatchesCpu` | PASS (executed, none skipped) |
| `compute-sanitizer --tool memcheck`, 1000 frames, `--gpu-preprocess --gpu-postprocess --segmentation` | PASS: 1000 frames, 0 errors (about 37 min) |
| UnitTests on the GPU build (`test_gpu_postprocess` on the device) | PASS |
| `check_display.sh`, plain and `--gpu-preprocess --gpu-postprocess --segmentation` | PASS: 768×576 window with masks, boxes and labels; `q` stopped the run at 86 and 87 of 1000 frames, exit 0 |

Benchmarks (432):

| Stage | GPU | CPU |
|-------|-----|-----|
| Preprocess, video frame | 0.64 ms | 4.3 ms (4.9 ms with upload) |
| Preprocess, JPEG | 2.5 ms (nvJPEG) | 14.2 ms |
| Segmentation postprocess | 306 ms | 2301 ms |

Not verified:
- Independent-halves builds and configure guards (`SKIP_BUILD_MATRIX=1`; they do not depend on the
  card).
- The default ONNX Runtime path and its bit-identical check.
- `SKIPPED` on a device-less host.
- The per-stage benchmark with a real engine.
- `--display` with GPU-accelerated drawing to a real screen (Xvfb renders in software).
- DALI from the NGC container.

Defects found by this run, fixed on the branch:
- `run_gate.sh` could not load `libnvinfer_builder_resource_sm75` on a fresh box: the TensorRT
  libs were not yet on `LD_LIBRARY_PATH`.
- Colab's own `CUDA_VERSION` replaced the pin in `setup_colab.sh`.
- Re-running the notebook deleted the kernel's working directory.

## NVIDIA L4, 2026-10-08 — PR #24 (`01b950b`) + Colab tooling

PR #24 (`feature/clang-tidy-enforcement-and-cli-refactor`) rewrites the code `--display` goes
through: CLI parsing moves to `src/cli_options.cpp`, and `video_reader.cpp`, `video_writer.cpp` and
`rfdetr_inference.cpp` change. The T4 run above covered `develop`, so this run checks the PR's own
source, with `scripts/` taken from `feature/colab-gpu-gate`.

NVIDIA L4 (sm_89, 23 GB), driver 580.82.07 (CUDA 13.0), CUDA toolkit 13.3, TensorRT 11.2.1.2
(pre-fetched with `wget`: the configure-time download timed out at 54 %), DALI 2.2.0 from pip.
Same model as the T4 run.

| Check | Result |
|-------|--------|
| Full builds, CUDA pre + CUDA post and DALI pre + CUDA post, sm_89, `-DWERROR=ON` | PASS |
| Four pre/post combinations through `inference_app` | PASS (smoke) |
| `GpuParityIntegration` on both builds | PASS (3 tests executed, none skipped) |
| `compute-sanitizer --tool memcheck`, 1000 frames, full GPU pipeline | PASS: 1000 frames, 0 errors |
| UnitTests on the GPU build | PASS |
| `check_display.sh`, plain and `--gpu-preprocess --gpu-postprocess --segmentation` | PASS: 768×576 window with image content; `q` stopped the run at 402 and 396 of 1000 frames, exit 0 |

Benchmarks (432): CUDA preprocessing 0.61 ms per frame against 4.1 ms on the CPU. Segmentation
postprocessing 266 ms on the GPU against 2125 ms on the CPU.

Not verified: the same items as the T4 run.
