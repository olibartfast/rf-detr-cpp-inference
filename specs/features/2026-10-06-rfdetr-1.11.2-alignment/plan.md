# Plan

1. Read the 1.11.0, 1.11.1 and 1.11.2 release notes and the 1.10.1...1.11.2 diff of `src/rfdetr/export`, `src/rfdetr/models` and `pyproject.toml`; classify the export boundaries (no GPU).
2. Export detection, segmentation and keypoint ONNX with 1.10.1 and 1.11.2 at the same resolutions; compare signatures and outputs (CPU).
3. Run the C++ ONNX Runtime app and the integration tests on the 1.11.2 exports (CPU).
4. Export a `.pte` with the extra's ExecuTorch 1.3.1 and run it on the v1.4.0 C++ runtime (CPU).
5. Cast ONNX to FP16 with 1.11.2's TensorRT converter and run it in the C++ TensorRT 11 backend (GPU, Docker image).
6. Update `versions.env`, `deploy/requirements.txt`, the README table, `docs/export.md` and the `deploy/export_executorch.py` note; run version sync, the lint gate and the tests; add a CHANGELOG entry.
