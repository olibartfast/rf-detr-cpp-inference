# Requirements: RF-DETR 1.10.0 Alignment

The standing obligation in [roadmap.md](../../roadmap.md) requires an alignment pass for
[RF-DETR 1.10.0](https://github.com/roboflow/rf-detr/releases/tag/1.10.0). The official tags
compared are 1.9.4 at `9b009fa928d6218320439803d1da01869a85c072` and 1.10.0 at
`0f432b6c7ace44d9494bcd7b92928e437e7fa7c5`.

## Classification

| Question | Answer |
|----------|--------|
| Exported model inputs change? | **No.** Shape, dtype, normalization, and resolution contracts are unchanged. The uint8 transfer optimization is in Python `predict()`, outside exported models. |
| Exported outputs change? | **No.** Names, order, count, shapes, and detection semantics remain unchanged. Reference-decoder optimizations are Python/NumPy-side and output-identical. |
| Required runtime operators change? | **No expected change.** ONNX opset 17 and the ExecuTorch runtime/delegate policy remain unchanged; artifact inspection is the validation gate. |
| Public Python APIs used by `deploy/` change? | **Yes, additively.** `RFDETR.export()` adds `output_name`. Default ExecuTorch names gain a delegate suffix and TensorRT names gain a precision suffix. |
| Training/dataset-only change? | **Mostly.** Optimizer, matcher, validation, loader, dataset, metric, and loss changes do not reach native C++ inference. |

Opset: upstream 1.10.0 defaults `opset_version=17` (`detr.py:1580`, `exporter.py:170`, `main.py:313`, `docs/learn/export.md:76`); this repo's `ONNX_OPSET_VERSION=17` matches that default, so no opset change is required.

## In scope

- Pin `rfdetr[onnx]` 1.10.0 in `versions.env` and `deploy/requirements.txt`.
- Pass explicit `output_name` values from the four export scripts so the repository's documented
  ONNX and PTE filenames remain stable.
- Treat the path returned by `model.export()` as authoritative and reject a missing artifact.
- Document upstream's new default ExecuTorch (`*_xnnpack.pte`) and TensorRT (`*_fp16.trt`)
  suffixes.
- Synchronize current-version statements, CHANGELOG, and validation evidence.

## Out of scope

- No changes to `src/`, backend validation, CPU/CUDA postprocessing, CMake, DALI pipelines, or
  TensorRT/ExecuTorch runtime pins unless artifact validation disproves the classification.
- No mirroring of Python `predict()`, training, NumPy decoder, or segmentation allocation
  optimizations; none executes in this C++ runtime.
- No re-export of checked-in TensorRT engines as part of the source change.

## Decisions

- Local scripts use upstream's `output_name` instead of reproducing upstream suffix logic. This
  preserves stable project paths while allowing upstream defaults to evolve.
- A non-null, existing path returned by `model.export()` is required. Guessing a filename after a
  successful call would hide an upstream API drift.
- Historical 1.9.4 statements remain historical; only live package/version guidance moves.

## Verified findings

- Default ONNX opset is 17 across 1.10.0, identical to this project's pinned `ONNX_OPSET_VERSION`:
  `src/rfdetr/detr.py:1580` (`opset_version: int = 17`),
  `src/rfdetr/export/_onnx/exporter.py:170` (`opset_version: int = 17`),
  `src/rfdetr/export/main.py:313` (`opset_version=args.opset_version`), and
  `docs/learn/export.md:76` (table default `17`). Upstream tests pin 17. No opset bump reaches the
  runtime, so `ONNX_OPSET_VERSION=17` in `versions.env` stays unchanged.

## Acceptance

- Hermetic exporter tests prove output-name forwarding, returned-path reporting, missing-path
  rejection, stable defaults, and keypoint compatibility-copy behavior.
- A clean 1.10.0 environment exports at least one ONNX model; its input/output signature is
  compared with 1.9.4 and it runs through the C++ ONNX backend.
- Default build and unit tests pass.
- TensorRT/ExecuTorch artifact checks are run when their exporters and runtimes are available;
  otherwise they are recorded as `UNRUN`, never inferred.
