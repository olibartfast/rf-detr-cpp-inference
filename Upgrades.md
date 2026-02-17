# RF-DETR Upgrade Tracking

## 1.4.2 -> 1.4.3

**Upstream release**: https://github.com/roboflow/rf-detr/releases/tag/1.4.3

### What changed in this repo

| File | What changed |
|------|-------------|
| `deploy/requirements.txt` | `rfdetr[onnxexport]` 1.4.2 -> 1.4.3 |
| `docs/export.md` | Version bump |
| `README.md` | Version bump |

### Why

Patch release with no model or API changes affecting C++ inference. Upstream changes:

1. **Segmentation export fix** - resolved `deploy_to_roboflow` issue for segmentation model export (#578).
2. **MD5 checksum validation** - added checksum verification for downloaded pretrained weights (#679).
3. **COCO benchmarks** - added segmentation model benchmarks and updated inference thresholds (#678, #684).

No changes required to C++ source code, build system, or model postprocessing.

---

## 1.3.0 -> 1.4.2

**Upstream release**: https://github.com/roboflow/rf-detr/releases/tag/1.4.2

### What changed in this repo

| File | What changed |
|------|-------------|
| `deploy/requirements.txt` | `rfdetr[onnxexport]` 1.3.0 -> 1.4.2 |
| `deploy/export_detection.py` | Removed deprecated `RFDETRBase`, default `medium`, added `xlarge`/`2xlarge` |
| `deploy/export_segmentation.py` | Replaced `RFDETRSegPreview` with sized classes + added `--model_type` arg |
| `docs/export.md` | Version bump + updated Python API examples to new class names |
| `README.md` | Version bump, gcc alternative, minimal OpenCV install note, Ninja optional |
| `CMakeLists.txt` | `find_package(OpenCV)` now specifies required components (`core`, `imgproc`, `imgcodecs`) |
| `src/backends/onnx_runtime_backend.cpp` | Fixed `get_output_count()` returning 0 before inference (used `ort_output_tensors_` instead of `output_name_strings_`) |

### Why

1. **Seg ONNX export was broken** - upstream fixed it in #626. `RFDETRSegPreview` is gone, replaced by `RFDETRSegNano/Small/Medium/Large/XLarge/2XLarge`.
2. **`RFDETRBase` deprecated** - upstream no longer lists it. Use `RFDETRNano/Small/Medium/Large` instead.
3. **XL/2XL models added** - require `pip install rfdetr[plus]` (PML 1.0 license, not Apache).
4. **ONNX Runtime output count bug** - `get_output_count()` checked `ort_output_tensors_` which is only populated after `run_inference()`, but the constructor validates output count before that. Fixed to use `output_name_strings_` (populated during `initialize()`).
