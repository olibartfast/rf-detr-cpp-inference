# **RF-DETR Export Instructions**  

Follow the procedure listed at https://rfdetr.roboflow.com/learn/deploy/
## Requirements

> [!IMPORTANT]
> - Python version: **3.10+** (upstream `rfdetr` 1.9.0; Python 3.11 venv still recommended here)
> - Starting with RF-DETR 1.6.0, the export extra was renamed: use `pip install rfdetr[onnx]`
> - **Tested version**: `rfdetr[onnx]==1.9.0`
> - Starting with RF-DETR 1.7.0, ONNX exports use variant filenames (e.g. `rfdetr-medium.onnx`, `rfdetr-seg-medium.onnx`) instead of the generic `inference_model.onnx`
> - The `--simplify` flag was removed in 1.8.0 (already deprecated in 1.7.0). Export scripts no longer accept it.
> - RF-DETR 1.8.x adds keypoint model export support via `RFDETRKeypointPreview`.
> - 1.9.0 relaxes the `onnxsim` pin to `>=0.7.0`. Earlier releases pinned `<0.6.0`, which had no prebuilt wheels for CPython 3.11/3.13 and made `pip install rfdetr[onnx]` hang building it from source — relevant here because this guide recommends a 3.11 venv.
> - 1.9.0 fixes ONNX export at a non-native resolution, which previously crashed.
> - 1.9.0 adds ExecuTorch (`.pte`) export via `rfdetr[executorch]`; see [ExecuTorch Model Export](#executorch-model-export).

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
pip install rfdetr[onnx]==1.9.0
```

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
pip install 'rfdetr[executorch]==1.9.0'
```

> [!IMPORTANT]
> Pin the version. `rfdetr[executorch]` does not constrain ExecuTorch itself, and `.pte` schema
> compatibility across ExecuTorch releases is not guaranteed. `rfdetr[executorch]==1.9.0` resolves
> ExecuTorch **1.3.1**, which is the C++ runtime version this project pins in
> `cmake/deps/packages/ExecuTorch.cmake`. An unpinned install can pull a newer exporter whose `.pte`
> the runtime cannot load.

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

Requires `pip install 'rfdetr[tensorrt]==1.9.0'`, which provides `tensorrt` + `polygraphy`. The engine is built in-process through the polygraphy API rather than by shelling out to `trtexec`, so no `trtexec` binary is needed, and it is built for the local GPU architecture. Pass `fp16=False` on TensorRT builds that do not expose the FP16 builder flag.

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