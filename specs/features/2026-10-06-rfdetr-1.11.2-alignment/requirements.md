# RF-DETR 1.11.2 alignment

Implements the upstream-alignment obligation in ../../roadmap.md. Moves `RFDETR_VERSION` from
1.10.1 across three releases: 1.11.0, 1.11.1 and 1.11.2.

Official comparison: https://github.com/roboflow/rf-detr/compare/1.10.1...1.11.2
Release notes: [1.11.0](https://github.com/roboflow/rf-detr/releases/tag/1.11.0),
[1.11.1](https://github.com/roboflow/rf-detr/releases/tag/1.11.1),
[1.11.2](https://github.com/roboflow/rf-detr/releases/tag/1.11.2)

## Contract classification

| Boundary | Finding |
|----------|---------|
| Inputs | No shape, dtype, normalisation or resolution change. |
| Outputs | Names, order, shapes and dtypes unchanged. The graph changes: 1.11.2 replaces the segmentation head's `Einsum` with a `MatMul` and drops two `Concat` patterns that ONNX Runtime's CoreML provider rejects, and 1.11.0 reshapes the keypoint decode (`repeat` in place of broadcasting). These are refactors: for the same weights the outputs are bit-identical to 1.10.1. |
| Runtime operators | ONNX opset stays 17. ExecuTorch: the `[executorch]` extra is capped at `>=1.3,<1.4` in 1.11.2, so `.pte` files are exported with 1.3.x while the C++ runtime stays at `EXECUTORCH_VERSION` (v1.4.0). This needs a runtime check, not a pin change. |
| Python export API | 1.11.0 moves export internals into per-format `Exporter` classes and removes `optimize_for_inference()`. `deploy/` calls only `RFDETR.export()`, whose signature is unchanged. |
| TensorRT | 1.11.0 makes `export(format="tensorrt", fp16=True)` cast the ONNX to FP16 on TensorRT 11, with float32 boundary casts. `docs/export.md` said this falls back to FP32. Dynamic-batch engines are new and not consumed here (batch 1 is deferred). |
| Training/other | LoRA checkpoint loading, crowd-region mAP, `predict()` thread safety, `class_names` validation, YOLO dataset layout, CUDA-graph training, new OpenVINO/LiteRT/Core AI exports. None reach C++. |

## Scope

In: bump `RFDETR_VERSION` and its restatements, correct the ExecuTorch version guidance and the
TensorRT 11 FP16 guidance in `docs/export.md` and the `deploy/export_executorch.py` note, and
validate fresh exports of all three tasks on the default runtime, plus ExecuTorch and TensorRT.

Out: C++ or CUDA changes (none needed), dynamic batch, the new export formats, and a release tag.
