# RF-DETR 1.12.0 alignment

## Scope
Align RFDETR_VERSION and export tooling with official [1.12.0](https://github.com/roboflow/rf-detr/releases/tag/1.12.0), fulfilling the roadmap standing obligation.

| In | Paths |
|---|---|
| Pins and guidance | versions.env, deploy/requirements.txt, README.md, docs/export.md, docs/advanced-usage.md |
| Preserve experimental LiteRT pose | deploy/litert_keypoint_export.py, tests/python/test_litert_keypoint_export.py |
| ExecuTorch guidance | deploy/export_executorch.py |
| Alignment evidence | CHANGELOG.md, this directory |

Out: new Python inference features, quantization, new models, metadata ingestion in C++, and runtime/backend redesign.

## Contract classification and decisions
- Inputs: default RGB NCHW float32, ImageNet normalization and non-antialiased resize unchanged. Optional Python antialias is not a new default.
- Outputs: raw dets/labels/masks/keypoints order and semantics unchanged. Final segmentation masks unchanged; only auxiliary eval masks removed. No CPU/CUDA postprocess change is indicated by source review (mission: one contract).
- Operators: ONNX optimizer removed; XNNPACK decomposes attention and folds constants; LiteRT uses pixel-row sampling. Re-export and validate runtime behavior before claiming compatibility.
- Public export APIs: existing keyword exports remain; integer batch size validation already satisfied by helpers. New embedded/sidecar metadata must retain actual keypoint output names.
- LiteRT rejects keypoints before the existing adapter hook. Bypass only that refusal in the opt-in adapter while retaining upstream conversion and sampling, float32 contract and numerical rewrite checks.
- No new dependency or runtime pin; metadata is additive and unused by C++.

## Context
Baseline origin/develop ddc2a90, tracking RFDETR_VERSION=1.11.2. Local develop diverges; work occurs in an isolated checkout without rewriting it. Upstream tag 1.12.0 is 639eac4555ceb89ef399ccda3fde716bbcfe34cb. Mission commitments: exclusive backends, CPU/GPU parity, model-free unit tests. Actual export environments and device availability determine which runtime checks can execute; missing checks remain UNRUN.
