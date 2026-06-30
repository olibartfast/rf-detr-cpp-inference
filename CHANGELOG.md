# CHANGELOG

Tracks upstream `rfdetr` version changes that affect this C++ inference project.

---

## v0.2.1

**Upstream release**: https://github.com/roboflow/rf-detr/releases/tag/1.8.3 (partial backport)

### Fixed

| File | Change |
|------|--------|
| `src/processing_utils.hpp` | New `clamp_box()` declaration. |
| `src/processing_utils.cpp` | Implement `clamp_box()` using `std::clamp` to constrain boxes to `[0, max_w] x [0, max_h]`. |
| `src/rfdetr_inference.cpp` | Apply `clamp_box()` after `scale_box()` in detection, segmentation, and keypoint postprocess paths. |
| `tests/unit/test_rfdetr_inference.cpp` | 4 `ClampBox` unit tests + 1 end-to-end `PostprocessTest.BoxesClampedToImageBounds` test. |

### Why

Ports the box-clamping fix from upstream rf-detr v1.8.3 (`PostProcess._postprocess_boxes()`, #1168). Predicted boxes are now guaranteed within `[0,width] x [0,height]`, so objects at image edges no longer produce negative `x1/y1` or out-of-frame `x2/y2` coordinates. Partial backport: the fp16 `scale_fct` dtype-cast is N/A in C++; training/export-side 1.8.3 changes do not affect postprocessing, so no model re-export is required.

---

## v0.2.0

**Upstream release**: https://github.com/roboflow/rf-detr/releases/tag/1.8.0

### Added

| File | Change |
|------|--------|
| `deploy/export_keypoint.py` | **New.** Keypoint model export script using `RFDETRKeypointPreview`. |
| `src/rfdetr_inference.hpp` | `ModelType::KEYPOINT`, `KeypointResult` struct, keypoint Config fields (`keypoint_counts`, `skeleton`, `keypoint_uncertainty_alpha`, `draw_uncertainty`, `keypoint_color`). Declarations for `postprocess_keypoint_outputs()`, `draw_keypoints()`, `get_label_name()`. |
| `src/rfdetr_inference.cpp` | Keypoint postprocessing pipeline: 3-output validation, background-offset class mapping, per-query class selection, bbox decode, image-relative keypoint coordinate decode, sigmoid findability/visibility, Precision Cholesky → pixel covariance, uncertainty-weighted score fusion. `draw_keypoints()`: boxes + labels, keypoint circles (radius ∝ findability), skeleton lines, optional uncertainty ellipses. `get_label_name()` helper. |
| `src/main.cpp` | `--keypoint` CLI flag, dispatch to keypoint postprocessing and drawing. Keypoint result output (coordinates, findability, visibility per keypoint). |
| `src/video_pipeline.hpp` | `keypoints` field in `FrameSlot`. |
| `src/video_pipeline.cpp` | `draw_keypoint_on_frame()` helper. KEYPOINT dispatch in `infer_postprocess_stage()` and `draw_write_stage()`. |
| `tests/unit/test_rfdetr_inference.cpp` | 7 keypoint postprocessing unit tests: 3-output validation, class selection + bbox decode, keypoint coordinate decode, scale factor application, no-detection threshold, Cholesky→covariance math, background column skipping. |
| `tests/integration/integration_test_rfdetr_inference.cpp` | Keypoint E2E integration test with `RFDETR_KEYPOINT_MODEL` env var. |
| `docs/export.md` | Keypoint model export section. |

### Changed

| File | Change |
|------|--------|
| `deploy/requirements.txt` | `rfdetr[onnx]` 1.7.0 → 1.8.0 |
| `deploy/export_detection.py` | Removed `--simplify` arg and deprecation warnings. |
| `deploy/export_segmentation.py` | Same as detection export script. |
| `docs/export.md` | Version bumps 1.7.0→1.8.0. Removed `--simplify` from options table. Removed TRT re-export note (1.7.0-specific). |
| `README.md` | Version bump, keypoint usage examples, keypoint model in ONNX download section. |
| `tests/integration/integration_test_rfdetr_inference.cpp` | Skip message 1.7.0→1.8.0. |
| `.github/workflows/ci.yml` | Add `develop` branch to push/PR triggers. |
| `README.md` | CI section updated to reflect `develop` branch. |

### Why

RF-DETR 1.8.0 introduces keypoint detection models via `RFDETRKeypointPreview`. The `--simplify` flag (deprecated since 1.7.0) is removed entirely in this release.

---

## v0.1.3

**Upstream release**: https://github.com/roboflow/rf-detr/releases/tag/1.7.0

### Changed

| File | Change |
|------|--------|
| `deploy/requirements.txt` | `rfdetr[onnx]` 1.6.5.post0 → 1.7.0 |
| `deploy/export_detection.py` | `--simplify` deprecated warning; print variant ONNX filename |
| `deploy/export_segmentation.py` | Same as detection export script |
| `deploy/inspect_model.py` | `RFDETRSegPreview` → `RFDETRSegMedium` |
| `docs/export.md` | Version bump, variant ONNX filenames, simplify deprecation, TRT re-export note |
| `README.md` | Version bump + variant ONNX filename examples + integration test model env var |
| `export_trt.sh` | Example path `inference_model.onnx` → `rfdetr-medium.onnx` |
| `Dockerfile.tensorrt` | NGC TRT Docker tag 25.09 → 25.12; variant ONNX examples in usage comments |
| `tests/integration/integration_test_rfdetr_inference.cpp` | Resolve model from `RFDETR_TEST_MODEL` or variant filenames; skip E2E when absent |

### Why

No C++ source changes required. ONNX tensor names and postprocessing unchanged.

Notable upstream changes between 1.6.5.post0 and 1.7.0:

1. **Variant ONNX filenames** (1.7.0) — exports named e.g. `rfdetr-medium.onnx`, `rfdetr-seg-medium.onnx` instead of generic `inference_model.onnx`.
2. **`simplify` deprecated** (1.7.0) — `model.export(simplify=True)` is a no-op; ONNX simplification no longer runs during export.
3. **ONNX/TRT dynamic batch fix** (#950) — re-export with 1.7.0 before building TensorRT engines if you rely on dynamic batch shapes.
4. **Import path deprecations** — `rfdetr.util.*` → `rfdetr.utilities.*`, `rfdetr.deploy.*` → `rfdetr.export.*` (removal in v1.8).
5. **Class deprecations** — `RFDETRBase`, `RFDETRSegPreview` emit warnings; use sized variant classes instead.

---

## v0.1.2

**Upstream release**: https://github.com/roboflow/rf-detr/releases/tag/1.6.5.post0

### Changed

| File | Change |
|------|--------|
| `deploy/requirements.txt` | `rfdetr[onnxexport]` 1.4.3 → `rfdetr[onnx]` 1.6.5.post0 |
| `deploy/export_detection.py` | Added `--device` flag to allow explicit device selection (e.g. `--device cpu`) |
| `deploy/export_segmentation.py` | Added `--device` flag (same as above) |
| `docs/export.md` | Extra rename note, version bump, TRT Docker tag 25.09 → 25.12 |
| `README.md` | Version bump + extra rename |
| `export_trt.sh` | NGC TRT Docker tag 25.09 → 25.12 |

### Why

The `onnxexport` optional extra was renamed to `onnx` in rfdetr 1.6.0. No C++ source changes required.

Notable upstream changes between 1.4.3 and 1.6.5.post0:

1. **Extra rename** (1.6.0) — `rfdetr[onnxexport]` → `rfdetr[onnx]`.
2. **Non-square inference shapes** (1.6.2) — `export()` and `predict()` now accept `(height, width)` tuples; ONNX model output format unchanged.
3. **Fine-tuned model export fix** (1.6.3) — `reinitialize_detection_head` now replaces `nn.Linear` instead of mutating weights, so custom-class-count models export correctly.
4. **`torch.export.export` fix** (1.6.4) — `spatial_shapes_hw` was not threaded through decoder layers; fixed for models using multi-scale deformable attention.
5. **PTL pin** (1.6.5.post0) — pins PyTorch Lightning ≤ 2.6.1.

---

## v0.1.1

[v0.1.1](https://github.com/olibartfast/rf-detr-cpp-inference/commit/f9028533ad96d79117da2a74a5aa121fd80277c1)

**Upstream release**: https://github.com/roboflow/rf-detr/releases/tag/1.4.3

### Changed

| File | Change |
|------|--------|
| `deploy/requirements.txt` | `rfdetr[onnxexport]` 1.4.2 → 1.4.3 |
| `docs/export.md` | Version bump |
| `README.md` | Version bump |

### Why

Patch release with no model or API changes affecting C++ inference. Upstream changes:

1. **Segmentation export fix** — resolved `deploy_to_roboflow` issue for segmentation model export (#578).
2. **MD5 checksum validation** — added checksum verification for downloaded pretrained weights (#679).
3. **COCO benchmarks** — added segmentation model benchmarks and updated inference thresholds (#678, #684).

---

## v0.1.0

[v0.1.0](https://github.com/olibartfast/rf-detr-cpp-inference/commit/5ba569b7f7454a2b0fbe3e56ee885d9dad46fc70)

**Upstream release**: https://github.com/roboflow/rf-detr/releases/tag/1.4.2

### Changed

| File | Change |
|------|--------|
| `deploy/requirements.txt` | `rfdetr[onnxexport]` 1.3.0 → 1.4.2 |
| `deploy/export_detection.py` | Removed deprecated `RFDETRBase`, default `medium`, added `xlarge`/`2xlarge` |
| `deploy/export_segmentation.py` | Replaced `RFDETRSegPreview` with sized classes + added `--model_type` arg |
| `docs/export.md` | Version bump + updated Python API examples to new class names |
| `README.md` | Version bump, gcc alternative, minimal OpenCV install note, Ninja optional |
| `CMakeLists.txt` | `find_package(OpenCV)` now specifies required components (`core`, `imgproc`, `imgcodecs`) |
| `src/backends/onnx_runtime_backend.cpp` | Fixed `get_output_count()` returning 0 before inference |

### Why

1. **Seg ONNX export was broken** — upstream fixed it in #626. `RFDETRSegPreview` is gone, replaced by `RFDETRSegNano/Small/Medium/Large/XLarge/2XLarge`.
2. **`RFDETRBase` deprecated** — upstream no longer lists it. Use `RFDETRNano/Small/Medium/Large` instead.
3. **XL/2XL models added** — require `pip install rfdetr[plus]` (PML 1.0 license, not Apache).
4. **ONNX Runtime output count bug** — `get_output_count()` checked `ort_output_tensors_` which is only populated after `run_inference()`, but the constructor validates output count before that. Fixed to use `output_name_strings_` (populated during `initialize()`).
