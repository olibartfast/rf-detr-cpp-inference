# RF-DETR 1.10.1 alignment

Implements the upstream-alignment obligation in ../../roadmap.md.

Official comparison: https://github.com/roboflow/rf-detr/compare/1.10.0...1.10.1

## Contract classification

| Boundary | Finding |
|----------|---------|
| Inputs | No shape, dtype, normalization, or resolution change. |
| Outputs | No exported output or decoding change. |
| Runtime operators | No export-graph or ONNX opset change; retain opset 17 and runtime pins. |
| Python export API | No change to APIs used by deploy/. |
| Training | CUDA multi-scale compilation, segmentation loss memory, and experimental XLA fixes only. Nearest point sampling retains the existing CPU/CUDA kernel. |

Scope: update RFDETR_VERSION and live export guidance; validate a fresh ONNX export with the default runtime. Preserve historical validation records and the compact changelog. C++/CUDA changes, keypoint schema repair, and release tagging are outside this alignment packet.
