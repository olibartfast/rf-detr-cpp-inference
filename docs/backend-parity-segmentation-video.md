# Cross-Backend Segmentation Video Test

Manual test run verifying that all three inference backends — ONNX Runtime, TensorRT, and
ExecuTorch — process the same instance-segmentation model over the same video and produce
equivalent detections.

CI does not cover TensorRT or ExecuTorch (see `AGENTS.md`), so this is the record of a manual
cross-backend check. Run date: 2026-08-04, branch `develop`, at v0.4.0.

---

## Why This Test Exists

The three backends share everything downstream of the model: `RFDETRInference` postprocessing,
the ring-buffer video pipeline, and the drawing code. Only tensor execution differs. A
segmentation model over video exercises the widest path — 3 outputs instead of 2, per-instance
mask resize, and the video pipeline — so agreement across backends is a strong signal that the
backend abstraction is behaving.

---

## Test Inputs

| Item | Value |
|------|-------|
| Model | `RFDETRSegMedium`, 432×432 input |
| Outputs | `dets` `[1,200,4]`, `labels` `[1,200,91]`, `masks` `[1,200,108,108]` |
| Labels | `data/coco-labels-91.txt` |
| Video | [Pexels 18662635](https://www.pexels.com/video/a-group-of-people-walking-through-a-crowded-city-street-18662635/) — crowded street, 1920×1080, 60fps, 320 frames (5.4s), 4.3 MB |
| Threshold | 0.5 (default) |

The clip is a dense pedestrian scene, chosen so the frame carries many overlapping `person`
instances plus vehicles — the case where per-instance masks are actually distinguishable from a
single semantic blob.

> [!NOTE]
> Pexels blocks direct page scraping (Cloudflare 403), but `https://www.pexels.com/download/video/<id>/`
> 302-redirects to the underlying file and works with `curl -L`. Query parameters on that URL cause a 404.

---

## Commands

Each backend is compiled into its own build tree (exactly one backend per binary — enabling two is
a configure-time error). `main.cpp` hard-codes the video output to `output_video.mp4` **relative to
the current working directory**, so each run needs its own directory or the three results overwrite
each other.

```bash
# ONNX Runtime
mkdir -p run/onnx && cd run/onnx
../../build/inference_app \
    output_seg/rfdetr-seg-medium.onnx \
    street_1080p.mp4 \
    data/coco-labels-91.txt \
    --segmentation

# TensorRT
mkdir -p run/trt && cd run/trt
../../build-trt/inference_app \
    output_seg/rfdetr-seg-medium.engine \
    street_1080p.mp4 \
    data/coco-labels-91.txt \
    --segmentation

# ExecuTorch
mkdir -p run/et && cd run/et
../../build-et/inference_app \
    output_seg/rfdetr-seg-medium.pte \
    street_1080p.mp4 \
    data/coco-labels-91.txt \
    --segmentation
```

---

## Results

All three completed 320 frames with exit code 0, auto-detected 432×432, and reported the expected
three outputs.

| Backend | Model file | Wall time | Device |
|---------|-----------|-----------|--------|
| TensorRT | `rfdetr-seg-medium.engine` | 1m 14s | GPU |
| ONNX Runtime | `rfdetr-seg-medium.onnx` | 17m 45s | CPU |
| ExecuTorch (xnnpack) | `rfdetr-seg-medium.pte` | 19m 02s | CPU |

> [!WARNING]
> **These timings are not a benchmark.** ONNX Runtime and ExecuTorch ran on CPU while the TensorRT
> run used the GPU, and the runs overlapped in wall-clock time, so the CPU backends contended for
> cores with each other. Use `-DBENCHMARKS=ON` and the `benchmarks` target for real numbers.

### Detection parity

Scores for the seven above-threshold instances on frame 150:

| Backend | Instance scores |
|---------|-----------------|
| ONNX Runtime | 0.51, 0.59, 0.74, 0.66, 0.54, 0.52, 0.56 |
| ExecuTorch | 0.51, 0.59, 0.74, 0.66, 0.54, 0.52, 0.56 |
| TensorRT | 0.50, 0.58, 0.73, 0.65, 0.54, 0.52, 0.56 |

ExecuTorch and ONNX Runtime agree exactly. TensorRT is ~0.01 lower on four of the seven and
identical on the rest — expected engine build precision drift, not a postprocessing difference:
the same instances are found, with the same class assignments and visually identical masks.

Single-image sanity check on `data/dog.jpg` through the ExecuTorch backend, for a scene with
mixed classes rather than only `person`:

```
Found 4 instances above threshold 0.5
  bicycle   0.946   55085 mask px
  dog       0.866   35495 mask px
  car       0.806   12784 mask px
  motorbike 0.572    1754 mask px
```

---

## Gap Found: No Segmentation `.pte` Export

**`deploy/export_executorch.py` cannot export a segmentation model.** Its `--model_type` choices
cover only the detection classes (`nano`, `small`, `medium`, `large`, `xlarge`, `2xlarge`) and it
instantiates `RFDETRNano`…`RFDETR2XLarge` — none of the `RFDETRSeg*` classes. `docs/export.md`
correspondingly documents the `.pte` output layout as `dets` + `labels` only.

This is a limitation of this project's tooling, **not** of upstream `rfdetr` or of the C++ runtime:

- `rfdetr` 1.9.0 exports `RFDETRSegMedium` to a `.pte` with no error; there is no seg/ExecuTorch
  block in `RFDETRBase.export()`.
- `ExecuTorchBackend::validate_output_order()` only inspects outputs 0 and 1, so a third `masks`
  output passes the check untouched.
- `RFDETRInference::postprocess_segmentation_outputs()` addresses `output_data_cache_[2]`
  positionally, so it is backend-agnostic — nothing about it is ONNX- or TensorRT-specific.

The `.pte` used for this test was produced with a standalone script equivalent to:

```python
from rfdetr import RFDETRSegMedium

model = RFDETRSegMedium(resolution=432)
model.export(format="executorch", backend="xnnpack", output_dir="output_seg")
```

`resolution=432` matches the existing ONNX/TensorRT segmentation models, so all three backends run
an identically-shaped model. The resulting `.pte` is 132 MB.

**Follow-up:** wire the `RFDETRSeg*` classes into `deploy/export_executorch.py` and document the
3-output segmentation layout in `docs/export.md#executorch-model-export`.

---

## Notes

- **Output path collision.** `main.cpp` writes video output to `output_video.mp4` in the current
  working directory with no override flag. Comparing backends requires per-run directories.
- **`.gitignore` coverage.** `*.onnx` and `*.engine` are ignored, but `*.pte` is not, and `*.mp4`
  is ignored only as the exact root-level `output_video.mp4`. Exported `.pte` models and collected
  result videos show up as untracked files.
