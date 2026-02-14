# RF-DETR Upgrade Tracking

## 1.3.0 -> 1.4.2

**Upstream release**: https://github.com/roboflow/rf-detr/releases/tag/1.4.2

### What changed in this repo

| File | What changed |
|------|-------------|
| `deploy/requirements.txt` | `rfdetr[onnxexport]` 1.3.0 -> 1.4.2 |
| `deploy/export_detection.py` | Removed deprecated `RFDETRBase`, default `medium`, added `xlarge`/`2xlarge` |
| `deploy/export_segmentation.py` | Replaced `RFDETRSegPreview` with sized classes + added `--model_type` arg |
| `docs/export.md` | Version bump + updated Python API examples to new class names |
| `README.md` | Version bump |

### Why

1. **Seg ONNX export was broken** - upstream fixed it in #626. `RFDETRSegPreview` is gone, replaced by `RFDETRSegNano/Small/Medium/Large/XLarge/2XLarge`.
2. **`RFDETRBase` deprecated** - upstream no longer lists it. Use `RFDETRNano/Small/Medium/Large` instead. The detection export script still accepts `base` for backward compat.
3. **XL/2XL models added** - require `pip install rfdetr[plus]` (PML 1.0 license, not Apache).

### What did NOT change

- No C++ code changes needed. Model ONNX outputs are the same format.
- No build system changes.
