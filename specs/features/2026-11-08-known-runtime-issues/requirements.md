# Correct known runtime defects

## Scope
User explicitly requested technical fixes for known issues on 2026-11-08, after withdrawing v0.7.1. Work branches from develop; master remains v0.7.0. This is corrective work, not a new release or roadmap phase.

| In | Paths |
|---|---|
| ONNX thread control | src/backends/inference_backend.*, src/backends/onnx_runtime_backend.*, src/cli_options.*, src/main.cpp, src/rfdetr_inference.* |
| Upstream-correct pose scoring and threshold | src/rfdetr_inference.cpp, src/processing_utils.*, tests/unit/test_rfdetr_inference.cpp |
| Reproducible LiteRT export dependencies | versions.env, deploy/, scripts/check_version_sync.sh, docs/export.md, README.md |
| Reproducible runtime checks | tests/integration/, scripts/colab/, this spec |
| Accurate issue resolution | CHANGELOG.md, docs/advanced-usage.md, docs/usage.md |

## Decisions
- Preserve the inference-backend interface; runtime configuration goes through the factory, not backend conditionals in the orchestrator.
- ONNX default uses its supported automatic intra-op thread selection, with an explicit nonnegative CLI override and a way to retain one thread.
- Pose scores must mirror RF-DETR PostProcess: findability-weighted normalized covariance trace in log space, sigmoid normalization and threshold on the final score. Pixel covariance remains for drawing only. Detection and segmentation are unaffected.
- Fix Pydantic compatibility through a reproducible supported export environment, never by mutating installed packages.
- Investigate JPEG parity against the model contract; decoder differences within the established tolerance are not a proven algorithm bug. A behavior change must be specified and tested before claiming a fix.
- User requested Colab for real GPU verification; use an accessible connected runtime. Do not rent hardware or invent hardware passes.
- CPU-only LiteRT is supported scope, not a defect demanding unapproved GPU/quantized support.

## Context
ONNX hardcodes one thread (src/backends/onnx_runtime_backend.cpp:25). Existing profiling measured a 2.90x four-thread counterfactual; thread behavior must be remeasured locally. Pose C++ currently derives fusion from pixel covariance and does not normalize/filter fused scores, unlike upstream models/postprocess.py. Prior artifact checks compared raw tensors, not this decoder, so they did not prove pose score parity. Colab runbook and notebook already exist. Local host has CUDA toolkit but no NVIDIA device/driver CLI; Brev is logged out.
