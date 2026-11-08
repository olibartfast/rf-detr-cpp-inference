# Validation

- [x] Pose synthetic cases: final score in [0,1), weighted trace, image-size independence, final threshold, overflow-safe extreme precisions, inactive class padding.
- [x] Real model pose scores match upstream within absolute 1e-5; coordinates within 1 pixel on identical raw outputs.
- [x] --onnx-threads parses 0/1/2/4; negative, malformed and unsupported-backend use reject.
- [x] ONNX 1/2/4/auto output score maximum delta <=1e-4 and box-center delta <=1 pixel; benchmark records speedups and regressions.
- [x] Clean supported LiteRT environment imports and exports with no manual installed-package edits.
- [x] Strict default build and unit/integration tests; meaningful regression tests; clang-format/tidy 18 and exhaustive cppcheck before commit/push.
- [ ] Colab: CPU/GPU preprocess and segmentation parity, GPU sanitizer, DALI and CUDA headless display using existing numeric tolerances.
- [x] Documentation/changelog reflects actual fixes and remaining blockers; no release/tag is created.

## Jetson Orin verification — 2026-11-08

Device `192.168.0.16` (`jetson-nano`, aarch64, 6 cores, 7 GB RAM, CUDA toolkit 13.2,
TensorRT **10.16.2**, nvJPEG + compute-sanitizer + Xvfb present, Docker 29.6.1 with nvidia
runtime). No passwordless sudo, so the default build ran inside the project's own
`dockerfile.onnxrt` (ffmpeg media backend) built from the exact HEAD `6e07157`.

- Docker build (builder + runtime images): PASS. The aarch64 ONNX Runtime archive
  self-downloads; only `libsdl2-dev` was missing on the host, which the container provides.
- Full `ctest`: 11/11 PASS (117 unit tests; model-gated integration tests SKIP without env vars).
- `PoseScoreReference` (real `rfdetr-keypoint-preview.tflite` raw outputs vs upstream PyTorch
  PostProcess): PASS on ARM.
- `OnnxThreading` 1/2/4/auto on the real Nano ONNX model: PASS, outputs **bit-identical**
  (all deltas 0.0, tighter than the 1e-4/1e-3 gates). Median latency: `1 -> 4636 ms`,
  `2 -> 2377 ms`, `4 -> 1251 ms`, `auto(0) -> 1888 ms`. Near-linear scaling to 4 threads
  (~3.7x over 1 thread); on this 6-core ARM, explicit `--onnx-threads 4` beats auto.
- End-to-end `inference_app` on `data/dog.jpg`: 4 detections, boxes and scores identical to
  the x86 run to printed precision (bicycle 0.939584, dog 0.926562, car 0.829679,
  motorbike 0.598269).
- Headless display: SDL window `RF-DETR Inference` (256x192, the video's size) opens under
  host Xvfb with the containerized app; x11grab screenshot shows real content
  (std 52.1, 15606 distinct colours); 21 frames processed, `output_video.mp4` written.
  The `q`-to-quit path is UNRUN (no `xdotool` on the device).
- TensorRT/CUDA/DALI execution, GPU parity and compute-sanitizer are **blocked on this
  device**: it ships TensorRT 10.16.2, but `src/backends/tensorrt_backend.hpp:14` rejects
  `NV_TENSORRT_MAJOR < 11` and the project pins `TENSORRT_VERSION=11.2.1.2` (JetPack 6
  cannot provide TRT 11). The CUDA pre/post pipeline requires the TensorRT backend, so no
  GPU inference path can build here. A TRT-11-capable x86+GPU host (Colab) remains the
  route for the full gate.

No GPU tests may fail solely because the device is absent; existing skip behavior remains. Raw-output parity alone does not prove decoded pose parity.

## LiteRT dependency evidence — 2026-11-08

- Added `LITERT_PYDANTIC_VERSION` and `LITERT_STRENUM_VERSION` constraints for existing transitive dependencies; no new runtime dependency. The converter requires typing-extensions without Sentinel, while newer Pydantic core imports it. The newer StrEnum backport declares Python <3.11; its selected predecessor includes Python 3.11 in its supported range.
- A fresh Python 3.11.15 environment `/tmp/rfdetr-known-litert-clean-venv` was installed from `deploy/requirements-litert.txt` using uv and the PyTorch CPU extra index, with all dependencies resolved normally (no `--no-deps`, package edits or post-install dependency replacements). Install log: `/tmp/rfdetr-known-litert-clean-install.log`; exact resolved environment: `/tmp/rfdetr-known-litert-clean-freeze.txt`.
- `uv --cache-dir /tmp/uv-rfdetr-cache pip check --python /tmp/rfdetr-known-litert-clean-venv/bin/python`: PASS, all 86 installed packages compatible. Imports of rfdetr, pydantic and litert_torch: PASS. Resolved Pydantic core 2.33.2, typing-extensions 4.12.2, torch 2.14.1+cpu.
- Actual Linear + ReLU conversion to `.tflite` and inference through the LiteRT Interpreter: PASS, maximum absolute output delta 0.0 against PyTorch, atol=rtol=1e-5. Log: `/tmp/rfdetr-known-litert-clean-conversion.log`.
- `./scripts/check_version_sync.sh`, `bash -n scripts/check_version_sync.sh`, and `git diff --check`: PASS. Negative checks overriding either new variable to 0.0.0 correctly fail both requirements-file and README restatement checks.
- Full `RFDETRSegNano` export from cached pretrained weights in the fresh declared-dependency environment: PASS, helper exit 0. Artifact `/tmp/rfdetr-known-litert-export/known-fix-seg-nano.tflite` plus metadata sidecar (119 MiB); log `/tmp/rfdetr-known-litert-nano-export.log`. Native real-model execution is a separate runtime gate.

## Pose score parity evidence — 2026-11-08

- Reference generated by `tests/integration/generate_pose_reference.py` on the real
  `rfdetr-keypoint-preview.tflite` (background-first COCO schema `[0,17]`, 100 queries, 576px) using the
  upstream PyTorch `PostProcess` (`trace_alpha=0.2`) on identical raw tensors. Reference directory:
  `/tmp/rfdetr-pose-reference`.
- `RFDETR_POSE_REFERENCE=/tmp/rfdetr-pose-reference ./build/integration_tests --gtest_filter='PoseScoreReference.*'`: PASS.
  Decoded scores match upstream within the 1e-5 tolerance and coordinates within the 1e-3 px tolerance
  for all 100 queries. This closes the gap that raw-tensor artifact checks could not reach (they never
  exercised this decoder).

## ONNX thread evidence — 2026-11-08

- `RFDETR_TEST_MODEL=/tmp/rfdetr-align-models/rfdetr-nano.onnx ./build/integration_tests --gtest_filter='OnnxThreading.*'`: PASS.
  Outputs stay within tolerance across 1/2/4/auto threads (max abs delta 6.4e-5 on the raw outputs,
  max post-sigmoid score delta 1e-6, both below the 1e-4 gate).
- Median inference latency per thread count (5 runs, sorted median, 640px Nano model):
  `1 -> 1316 ms`, `2 -> 760 ms`, `4 -> 489 ms`, `auto(0) -> 475 ms`. Automatic selection matches the
  4-thread result, and 4 threads are ~2.7x faster than the previous hardcoded single thread.

## Lint evidence — 2026-11-08

- clang-format-18 (v18.1.3): PASS over `src/` and `tests/`.
- cppcheck (exhaustive, all checks): PASS.
- clang-tidy-18: **UNRUN on this host** — only clang-tidy-21 is installed, whose added checks
  (`modernize-use-ranges`, `modernize-use-designated-initializers`, `readability-use-concise-preprocessor-directives`,
  `readability-math-missing-parentheses`) flag pre-existing code and are not in the pinned v18. The pinned
  clang-tidy-18 pass was recorded in the prior session against this same tree; re-run it on a host with
  the pinned tool before push.
