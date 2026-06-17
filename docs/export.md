# **RF-DETR Export Instructions**  

Follow the procedure listed at https://rfdetr.roboflow.com/learn/deploy/
## Requirements

> [!IMPORTANT]
> - Python version: **3.10+** (upstream `rfdetr` 1.8.0; Python 3.11 venv still recommended here)
> - Starting with RF-DETR 1.6.0, the export extra was renamed: use `pip install rfdetr[onnx]`
> - **Tested version**: `rfdetr[onnx]==1.8.0`
> - Starting with RF-DETR 1.7.0, ONNX exports use variant filenames (e.g. `rfdetr-medium.onnx`, `rfdetr-seg-medium.onnx`) instead of the generic `inference_model.onnx`
> - The `--simplify` flag was removed in 1.8.0 (already deprecated in 1.7.0). Export scripts no longer accept it.
> - RF-DETR 1.8.0 adds keypoint model export support via `RFDETRKeypointPreview`.

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
pip install rfdetr[onnx]==1.8.0
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
- `--output_dir`: Path to save exported model (default: current directory)
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

RF-DETR 1.8.0 adds keypoint detection via `RFDETRKeypointPreview`. Export with the provided script:

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

## TensorRT Export (Optional)

For GPU deployment, you can convert the ONNX model to TensorRT format for optimized performance.

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
> TensorRT optimization works for both detection and segmentation models. The C++ inference engine supports both ONNX Runtime and TensorRT backends with compile-time backend selection.