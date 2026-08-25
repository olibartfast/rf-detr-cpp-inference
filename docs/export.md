# **RF-DETR Export Instructions**  

Follow the procedure listed at https://rfdetr.roboflow.com/learn/deploy/
## Requirements

> [!IMPORTANT]
> - Python version: **3.10+** (upstream `rfdetr` 1.9.4; Python 3.11 venv still recommended here)
> - Starting with RF-DETR 1.6.0, the export extra was renamed: use `pip install rfdetr[onnx]`
> - **Tested version**: `rfdetr[onnx]==1.9.4`
> - Starting with RF-DETR 1.7.0, ONNX exports use variant filenames (e.g. `rfdetr-medium.onnx`, `rfdetr-seg-medium.onnx`) instead of the generic `inference_model.onnx`
> - The `--simplify` flag was removed in 1.8.0 (already deprecated in 1.7.0). Export scripts no longer accept it.
> - RF-DETR 1.8.x adds keypoint model export support via `RFDETRKeypointPreview`.
> - 1.9.0 relaxes the `onnxsim` pin to `>=0.7.0`. Earlier releases pinned `<0.6.0`, which had no prebuilt wheels for CPython 3.11/3.13 and made `pip install rfdetr[onnx]` hang building it from source — relevant here because this guide recommends a 3.11 venv.
> - 1.9.0 fixes ONNX export at a non-native resolution, which previously crashed.
> - 1.9.0 adds ExecuTorch (`.pte`) export via `rfdetr[executorch]`; see [ExecuTorch Model Export](#executorch-model-export).
> - 1.9.2 changes hierarchical COCO label mapping: unannotated grouping categories no longer consume a class slot, and all splits share the training split mapping. Retrain pre-1.9.2 checkpoints when adopting the re-filtered label space; otherwise per-class metrics can be misaligned. Exported model I/O is unchanged.
> - 1.9.1 gates the extras to interpreters that ship wheels: `[onnx]` pins `onnxruntime<1.24` on Python 3.10 (newer releases dropped cp310 wheels), and `[executorch]` installs empty on Python 3.14 (ExecuTorch ships cp310–cp313 only). On the recommended 3.11 venv both resolve exactly as before.
> - 1.9.1 makes the exported models' own inference helpers resize the way `predict()` does (bilinear, half-pixel centers, `antialias=False`) instead of PIL's antialiased filters. This C++ project already resizes that way — see [Preprocessing parity](#preprocessing-parity) — so exported ONNX/`.pte` models score here exactly as they did before; only upstream's Python-side ONNX/TFLite inference and INT8 calibration change. Re-export INT8 TFLite models if you ship any (not consumed by this project).
> - 1.9.1 speeds up ExecuTorch/XNNPACK export ~2.5× by recombining undelegated `addmm` into `aten.linear`. **This changes which kernels the `.pte` needs at run time** — see [ExecuTorch Model Export](#executorch-model-export).
> - 1.9.2 changes nothing on the export or inference path — it is a training/dataset release. One item to know before retraining: hierarchical COCO datasets (Roboflow exports carrying a synthetic unannotated root category) no longer let that root consume a label slot, and `train`/`valid`/`test` now share one label mapping derived from `train`. A model retrained after that change has a different class layout than one trained before it, so **the label file you pass to `inference_app` must come from the same side of the change as the checkpoint**.
> - 1.9.3 makes exported-model decoding multi-label. Its ONNX/TFLite reference decoders had been taking a per-query `argmax`, so any query scoring above threshold on more than one class silently lost the rest; they now rank the flattened *(query, class)* grid the way `PostProcess._select_topk` does. `PostProcess` itself switched from `torch.topk` to a stable `argsort`, fixing the tie order to descending score, then ascending flattened index. **This project's C++ postprocessing mirrors both changes** — see [How detections are selected](../README.md#how-detections-are-selected).
> - 1.9.3 also fixes `SegmentationHead`'s `skip_blocks` branch to apply the learned `spatial_features_proj`. That is a training-loss fix only: the export path already applied the projection unconditionally, so exported models are byte-identical and no re-export is needed.
> - 1.9.4 makes the background logit slot explicit, via a `background_class_id` argument on the same reference decoders (`-1` default, `None` to keep every slot, `0` for background-first checkpoints). Mirrored here as `--background-class-id`. Note upstream's `-1` default mis-decodes the official pretrained COCO weights — a real foreground category occupies their final slot — which is why the C++ default is `0`; see the flag's README entry.
> - 1.9.4 also stops the TFLite helper from treating any lone rank-4 output as a segmentation mask (a keypoint export's `pred_keypoints` could be upsampled into `Detections.mask`). Not reachable here: this project selects output tensors by position under an explicit `--segmentation` / `--keypoint` mode, never by guessing from rank.
> - The remaining 1.9.2–1.9.4 fixes are training, augmentation, dataset, and TFLite-conversion changes with no bearing on an exported detection/segmentation/keypoint model.

### Setup Virtual Environment

```bash
# install python3.11 on Ubuntu 24.04
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.11

sudo apt install python3.11-venv python3.11-distutils -y

# Create virtual environment with Python 3.11
python3.11 -m venv rfdetr_venv
source rfdetr_venv/bin/activate

# Install RF-DETR with export dependencies (tested version)
pip install rfdetr[onnx]==1.9.4
```

---

## Preprocessing parity

An exported model only reproduces `predict()`'s numbers if the caller resizes the way `predict()`
does: **bilinear, half-pixel centers, no antialias filter**, on the raw tensor, then ImageNet
mean/std normalization. 1.9.1 added `rfdetr/export/_resize.py` and routed upstream's own ONNX/TFLite
inference helpers, INT8 calibration, and benchmark paths through it, replacing
`PIL.Image.resize(BILINEAR)` — which low-pass filters on downscale and drifts from training.

This project's C++ preprocessing (`media.cpp::preprocess_bgr_image`) uses exactly that convention:
`src = (dst + 0.5) * scale - 0.5`, clamped into the source extent, with a plain 2×2 bilinear tap and
no area averaging. Segmentation mask upsampling (`resize_threshold_mask`, and the CUDA kernel behind
`--gpu-postprocess`) uses the same formula.

Writing the convention down exposed one divergence, fixed in this release: the C++ clamped the
derived *sample index* rather than the *source coordinate*, so output pixels mapping before the first
source pixel kept a negative interpolation weight and extrapolated past the edge instead of holding
it. Mask upsampling always upscales, so this affected the leading rows and columns of every mask.
The convention is now locked in by `PreprocessFrame.ResizeIsAntialiasFree`,
`MaskResize.HalfPixelCenterBilinear`, `MaskResize.LeadingEdgeClampsInsteadOfExtrapolating`, and
`PreprocessFrame.UpscaleDoesNotExtrapolatePastEdge`.

---

## Detection Model Export

### ONNX Export for ONNX Runtime

RF-DETR supports exporting detection models to the ONNX format, which enables interoperability with various inference frameworks and can improve deployment efficiency.

```python
from rfdetr import RFDETRMedium  # or RFDETRNano/Small/Medium/Large

model = RFDETRMedium(pretrain_weights=<CHECKPOINT_PATH>)

model.export()
```

**Model Outputs:**
- `dets`: Bounding boxes `[batch, num_queries, 4]` in cxcywh format (normalized)
- `labels`: Class logits `[batch, num_queries, num_classes]`

This command saves the ONNX model to the `output` directory as `rfdetr-medium.onnx` (filename includes the model variant).

---

## Segmentation Model Export

### ONNX Export for Instance Segmentation

For instance segmentation, use the sized `RFDETRSeg*` model classes or the provided export script.

#### Using Python Script

```bash
python deploy/export_segmentation.py --model_type medium --input_size 432
```

**Available Options:**
- `--model_type`: Model type: `nano`, `small`, `medium`, `large`, `xlarge`, `2xlarge` (default: medium)
- `--output_dir`: Path to save exported model (default: `output`)
- `--opset_version`: ONNX opset version (default: 17)
- `--batch_size`: Batch size for export (default: 1)
- `--input_size`: Input image size (default: 640)

#### Using Python API

```python
from rfdetr import RFDETRSegMedium  # or RFDETRSegNano/Small/Large/XLarge/2XLarge

model = RFDETRSegMedium(pretrain_weights=<CHECKPOINT_PATH>)

model.export(
    opset_version=17,
    batch_size=1
)
```

**Model Outputs:**
- `dets`: Bounding boxes `[batch, num_queries, 4]` in cxcywh format (normalized)
- `labels`: Class logits `[batch, num_queries, num_classes]`
- `masks`: Segmentation masks `[batch, num_queries, mask_h, mask_w]` (e.g., 108x108)

This command saves the ONNX segmentation model to the `output` directory as `rfdetr-seg-medium.onnx`.

---

## Keypoint Model Export

### ONNX Export for Keypoint Detection

RF-DETR 1.8.x adds keypoint detection via `RFDETRKeypointPreview`. Export with the provided script:

```bash
python deploy/export_keypoint.py
```

**Available Options:**
- `--output_dir`: Path to save exported model (default: `output`)
- `--opset_version`: ONNX opset version (default: 17)
- `--batch_size`: Batch size for export (default: 1)
- `--input_size`: Input image size (default: 576, model resolution). Must be divisible by 24 (`patch_size=12` × `num_windows=2`).
- `--device`: Device for export, e.g. `cpu` or `cuda` (default: RF-DETR auto)

**Model Outputs:**
- `dets`: Bounding boxes `[batch, num_queries, 4]` in cxcywh format (normalized)
- `labels`: Class logits `[batch, num_queries, num_keypoint_classes]` (preview model: 2 classes for COCO person)
- `keypoints`: Keypoints `[batch, num_queries, C*K_max, 8]` where C = keypoint classes, K_max = max keypoints per class

This command saves the ONNX keypoint model to the `output` directory as `rfdetr-keypoint-preview.onnx`. The export script also writes a compatibility copy as `rfdetr-keypoint.onnx`.

> [!NOTE]
> `RFDETRKeypointPreview` is a Preview API — tensor layout may change in future releases.

---

## ExecuTorch Model Export

RF-DETR 1.9.0 adds ExecuTorch (`.pte`) export for on-device inference. The C++ side runs these
through the ExecuTorch backend (`-DUSE_EXECUTORCH=ON`).

```bash
pip install 'rfdetr[executorch]==1.9.4'
pip show executorch   # confirm the runtime version it resolved
```

> [!IMPORTANT]
> Pinning `rfdetr` is not enough. The extra only constrains ExecuTorch to `>=1.3,<2.0`, so the
> resolved runtime moves as ExecuTorch publishes releases — `rfdetr[executorch]==1.9.0` resolved
> **1.3.1**, the same pin resolves **1.4.0** today — and `.pte` schema compatibility across
> ExecuTorch releases is not guaranteed. This project pins the C++ runtime to **v1.4.0** in
> `cmake/deps/packages/ExecuTorch.cmake`; check `pip show executorch` after installing and pin it
> explicitly (`pip install 'executorch==1.4.0'`) if it differs from the runtime you build against.

> [!WARNING]
> **A 1.9.1+ `.pte` needs the optimized kernel set.** 1.9.1 recombines the `addmm` ops XNNPACK
> leaves un-delegated back into `aten.linear` — 6 such calls in an `RFDETRNano` export at 384×384,
> which is where its ~2.5× XNNPACK speedup comes from. `aten::linear.out` is registered only by
> ExecuTorch's optimized kernels; the portable set registers `addmm.out` and `mm.out` but not
> `linear.out`, so a portable-only runtime fails at load with an unregistered-operator error.
> Build the ExecuTorch prefix with `-DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON` (it defaults to `OFF`);
> see the README section [Building the ExecuTorch install
> prefix](../README.md#building-the-executorch-install-prefix). The C++ build links
> `optimized_native_cpu_ops_lib` when the prefix has it and warns at configure time when it does not.

### Using Python Script

```bash
python deploy/export_executorch.py --model_type medium --input_size 640
```

**Available Options:**
- `--model_type`: Model type: `nano`, `small`, `medium`, `large`, `xlarge`, `2xlarge` (default: medium)
- `--output_dir`: Path to save exported model (default: `output`)
- `--batch_size`: Batch size for export (default: 1)
- `--input_size`: Input image size (default: 640)
- `--backend`: `xnnpack` (CPU, fp32), `coreml` (Apple, fp16), or `qnn` (Qualcomm HTP, fp16) (default: xnnpack)
- `--soc`: Target SoC identifier, required when `--backend qnn` (e.g. `SM8650`)

> [!IMPORTANT]
> **Only `--backend xnnpack` produces a `.pte` this project can run.** The delegate is baked into
> the program and must be linked into the runtime to self-register, but `-DEXECUTORCH_DELEGATE`
> accepts only `xnnpack` or `portable` — there is no `coreml` or `qnn` delegate in this build. A
> CoreML or QNN `.pte` therefore fails at load with `Backend ... is not registered`. Those options
> exist for exporting to a *different* ExecuTorch runtime; the script warns before and after export.
> Note also that `portable` is only useful for a `.pte` exported with no delegate at all — every
> `rfdetr` ExecuTorch export applies one, so `xnnpack` is the setting that matches this workflow.

### Using Python API

```python
from rfdetr import RFDETRMedium  # or RFDETRNano/Small/Large/XLarge/2XLarge

model = RFDETRMedium(pretrain_weights=<CHECKPOINT_PATH>)

model.export(format="executorch", backend="xnnpack")
```

**Model Outputs** (same layout and order as the ONNX detection export):
- `dets`: Bounding boxes `[batch, num_queries, 4]` in cxcywh format (normalized)
- `labels`: Class logits `[batch, num_queries, num_classes]`

This saves the model to the `output` directory as `rfdetr-medium.pte`.

> [!IMPORTANT]
> The delegate is baked into the `.pte` at export time, and the C++ build must link the matching
> delegate library for it to self-register. Export with `--backend xnnpack` (the default) and build
> with `-DEXECUTORCH_DELEGATE=xnnpack` (also the default). A mismatch fails at run time, not at link
> time. The ExecuTorch backend checks at load that the program returns `dets` before `labels` and
> refuses to run otherwise, rather than decoding class logits as box coordinates.

> [!NOTE]
> ExecuTorch's `format="executorch"` with `backend="coreml"` produces a `.pte` and is distinct from
> `format="coreml"`, which produces a native `.mlpackage`. This project consumes `.pte` only.

> [!WARNING]
> **`FileNotFoundError: [Errno 2] No such file or directory: 'flatc'`** — `.pte` serialization shells
> out to the `flatc` FlatBuffers compiler, which ships in the venv's `bin/`. ExecuTorch resolves it
> from `PATH`, so invoking the interpreter by absolute path (`/path/to/venv/bin/python export.py`)
> without activating the venv fails. Either `source <venv>/bin/activate` first, or put the venv's
> `bin` on `PATH` for the command.

### Verified ONNX / ExecuTorch parity

`rf-detr-nano.pth` exported at 384×384 to both formats and run through `inference_app` produced
identical detections — same count, classes, and order at thresholds 0.5 and 0.05, with a maximum
box delta of 1e-4 px and score delta of 1e-6. The ExecuTorch backend requires no preprocessing or
postprocessing differences from the ONNX Runtime path.

Re-run for 1.9.1 (`rfdetr[onnx,executorch]==1.9.1`, ExecuTorch runtime v1.4.0 built with
`EXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON`, `data/dog.jpg`): both backends found the same 3 detections
with boxes equal to every printed digit and scores differing by at most 1e-6 (`car` 0.854236 vs
0.854237). Against the previous portable-only prefix the same `.pte` refused to run:

```
E executorch:method.cpp:819] Missing operator: [29] aten::linear.out
E executorch:method.cpp:1125] There are 6 instructions don't have corresponding operator registered
Error: ExecuTorch forward() failed with error code 20
```

---

## TensorRT Export (Optional)

For GPU deployment, you can convert the ONNX model to TensorRT format for optimized performance.

### Native Export (RF-DETR 1.9.0+)

1.9.0 can build the engine itself, without a separate `trtexec` step:

```python
from rfdetr import RFDETRMedium

model = RFDETRMedium(pretrain_weights=<CHECKPOINT_PATH>)
model.export(format="tensorrt", fp16=True)  # alias: format="trt"
```

Requires `pip install 'rfdetr[tensorrt]==1.9.4'`, which provides `tensorrt` + `polygraphy`. The engine is built in-process through the polygraphy API rather than by shelling out to `trtexec`, so no `trtexec` binary is needed, and it is built for the local GPU architecture. Pass `fp16=False` on TensorRT builds that do not expose the FP16 builder flag.

Note that `[tensorrt]` no longer installs `pycuda` as of 1.9.0 — that moved to the separate `[tensorrt-bench]` extra and is only needed for `TRTInference`'s async benchmarking mode. The standard export-to-engine path is unaffected.

The `trtexec` recipes below remain valid and are what `export_trt.sh` in this repo uses; prefer them when you want to build an engine on a machine without the Python package installed.

### Detection or Segmentation Models

```bash
trtexec --onnx=/path/to/model.onnx \
        --saveEngine=/path/to/model.engine \
        --memPoolSize=workspace:4096 \
        --fp16 \
        --useCudaGraph \
        --useSpinWait \
        --warmUp=500 \
        --avgRuns=1000 \
        --duration=10
```

### Using TensorRT Docker Container

```bash
export NGC_TAG_VERSION=25.12

docker run --rm -it --gpus=all \
    -v $(pwd)/exports:/exports \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v $(pwd)/model.onnx:/workspace/model.onnx \
    -w /workspace \
    nvcr.io/nvidia/tensorrt:${NGC_TAG_VERSION}-py3 \
    /bin/bash -cx "trtexec --onnx=model.onnx \
                            --saveEngine=/exports/model.engine \
                            --memPoolSize=workspace:4096 \
                            --fp16 \
                            --useCudaGraph \
                            --useSpinWait \
                            --warmUp=500 \
                            --avgRuns=1000 \
                            --duration=10"
```

> [!NOTE]
> TensorRT optimization works for both detection and segmentation models. The C++ inference engine supports ONNX Runtime, TensorRT, and ExecuTorch backends with compile-time backend selection — exactly one is compiled in.