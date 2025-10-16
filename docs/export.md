# **RF-DETR Export Instructions**  

Follow the procedure listed at https://rfdetr.roboflow.com/learn/deploy/
## Requirements

> [!IMPORTANT]
> - Python version: **3.11 or lower** (onnxsim currently requires Python <= 3.11)
> - Starting with RF-DETR 1.2.0, you must run `pip install rfdetr[onnxexport]` before exporting
> - **Tested version**: `rfdetr[onnxexport]==1.3.0`

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
pip install rfdetr[onnxexport]==1.3.0
```

---

## Detection Model Export

### ONNX Export for ONNX Runtime

RF-DETR supports exporting detection models to the ONNX format, which enables interoperability with various inference frameworks and can improve deployment efficiency.

```python
from rfdetr import RFDETRBase

model = RFDETRBase(pretrain_weights=<CHECKPOINT_PATH>)

model.export()
```

**Model Outputs:**
- `dets`: Bounding boxes `[batch, num_queries, 4]` in cxcywh format (normalized)
- `labels`: Class logits `[batch, num_queries, num_classes]`

This command saves the ONNX model to the `output` directory.

---

## Segmentation Model Export

### ONNX Export for Instance Segmentation

For instance segmentation, use the `RFDETRSegPreview` model class or the provided export script.

#### Using Python Script

```bash
python deploy/export_segmentation.py --simplify --input_size 432
```

**Available Options:**
- `--output_dir`: Path to save exported model (default: current directory)
- `--opset_version`: ONNX opset version (default: 17)
- `--simplify`: Simplify ONNX model using onnxsim
- `--batch_size`: Batch size for export (default: 1)
- `--input_size`: Input image size (default: 640)

#### Using Python API

```python
from rfdetr import RFDETRSegPreview

model = RFDETRSegPreview(pretrain_weights=<CHECKPOINT_PATH>)

model.export(
    opset_version=17,
    simplify=True,
    batch_size=1
)
```

**Model Outputs:**
- `dets`: Bounding boxes `[batch, num_queries, 4]` in cxcywh format (normalized)
- `labels`: Class logits `[batch, num_queries, num_classes]`
- `masks`: Segmentation masks `[batch, num_queries, mask_h, mask_w]` (e.g., 108x108)

This command saves the ONNX segmentation model to the `output` directory.

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
export NGC_TAG_VERSION=25.09

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
> TensorRT optimization works for both detection and segmentation models. The C++ inference engine now supports both ONNX Runtime and TensorRT backends. Backend selection is done at compile time using CMake flags (see docs/backends.md and docs/COMPILE_TIME_BACKEND.md).
 