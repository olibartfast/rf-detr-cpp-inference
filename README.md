# RF-DETR C++ Inference

[![C++](https://img.shields.io/badge/language-C++20-blue.svg)](https://en.cppreference.com/w/cpp)
[![CMake](https://img.shields.io/badge/build%20system-CMake-blue.svg)](https://cmake.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

C++ project for performing object detection and instance segmentation inference using the RF-DETR model with **multiple inference backends** (ONNX Runtime and TensorRT) and OpenCV.

**🎯 Features:**
- 🔄 **Multiple Backends**: ONNX Runtime (CPU/GPU) and TensorRT 10.x (GPU)
- 🎨 **Strategy Pattern**: Clean architecture with compile-time backend selection
- 📦 **Flexible Build**: Choose backend at compile time for optimal performance
- 🚀 **High Performance**: TensorRT 10.x optimization for NVIDIA GPUs (8.x+ also supported)
- 🔧 **Auto-detection**: Automatic input resolution and output tensor detection
- 🏗️ **Clean Architecture**: Organized namespace structure (`rfdetr::backend`) 
---

## Table of Contents
- [Dependencies](#dependencies)
- [Model Setup](#model-setup)
- [Installation](#installation)
- [Backend Selection](#backend-selection)
- [Usage](#usage)
- [Building](#building)
- [Acknowledgements](#acknowledgements)

---

## Dependencies

### Required (All Backends)
- **C++20 Compiler**: Clang++ 15 or compatible (e.g., `clang++-15`)
- **CMake**: Version 3.12 or higher
- **OpenCV**: Version 4.x (e.g., install via `sudo apt-get install libopencv-dev` on Ubuntu)
- **Google Test**: Version 1.12.1 (automatically fetched during build)
- **Ninja**: Optional but recommended (`sudo apt-get install ninja-build`)

### Backend-Specific Dependencies

#### ONNX Runtime Backend (Default)
- **ONNX Runtime**: Version 1.21.0 (automatically downloaded during build)
- **Platform**: Linux, Windows, macOS
- **Acceleration**: CPU and GPU (CUDA/DirectML)

#### TensorRT Backend (Optional)
- **TensorRT**: Version 10.x (8.x+ also supported)
- **CUDA Toolkit**: Version 12.x or later (11.x+ also supported)
- **Platform**: Linux with NVIDIA GPU
- **Acceleration**: NVIDIA GPU only

See [backends documentation](docs/backends.md) for detailed installation instructions.

---

## Model Setup

This project supports both RF-DETR detection and segmentation models from Roboflow.

1. **Visit the RF-DETR Repository**:
   - Go to the [RF-DETR GitHub repository](https://github.com/roboflow/rf-detr) for model details.
   - Read the [Roboflow blog](https://blog.roboflow.com/rf-detr/) for an overview.

2. **Download the ONNX Model**:
   - Follow instructions in the [export documentation](docs/export.md) to export models in ONNX format.
   - **Tested with**: `rfdetr[onnxexport]==1.3.0` (Python ≤ 3.11 required)
   - **Detection models**: Export with standard configuration (outputs: `dets`, `labels`)
   - **Segmentation models**: Export with segmentation configuration (outputs: `dets`, `labels`, `masks`)
   - Place the model (e.g., `inference_model.onnx`) in a chosen directory.

3. **Prepare the COCO Labels**:
   - Create a `coco-labels-91.txt` file with one label per line:
     ```
     person
     bicycle
     car
     motorbike
     aeroplane
     ...
     ```

---

## Installation

### Install Dependencies (Ubuntu)

```bash
sudo apt-get update
sudo apt-get install -y clang-15 libopencv-dev ninja-build cmake
```

Ensure `clang++-15` is available as your compiler.

---

## Backend Selection

This project uses **compile-time backend selection**. Choose the backend when building, not at runtime:

| Backend | Best For | Pros | Cons |
|---------|----------|------|------|
| **ONNX Runtime** | Development, CPU inference | Cross-platform, easy setup | Slower than TensorRT on GPU |
| **TensorRT** | Production on NVIDIA GPUs | Maximum performance | GPU-only, requires CUDA/TensorRT |

**Key Advantage**: Compile-time selection results in smaller binaries, faster startup, and no runtime overhead.

See [backends documentation](docs/backends.md) and [compile-time backend guide](docs/COMPILE_TIME_BACKEND.md) for detailed information.

---

## Usage

### Prepare Input Files

- The RF-DETR ONNX model file (e.g., `inference_model.onnx`)
- An input image (e.g., `image.jpg`)
- A COCO labels file (e.g., `coco-labels-91.txt`)

### Run Inference

After building the project (see below), run the inference application. **Note**: The backend is selected at compile time, not runtime.

#### Object Detection

```bash
./build/inference_app /path/to/model.onnx /path/to/image.jpg /path/to/coco-labels-91.txt
```

#### Instance Segmentation

```bash
./build/inference_app /path/to/model.onnx /path/to/image.jpg /path/to/coco-labels-91.txt --segmentation
```

**Note**: The backend (ONNX Runtime or TensorRT) was selected when you built the executable. To use a different backend, rebuild the project with different CMake flags.

**Features:**
- The output image is saved as `output_image.jpg`
- Detection/segmentation results (bounding boxes, labels, scores, and mask pixels) are printed to the console
- Input resolution is automatically detected from the model (supports 432x432, 560x560, etc.)
- Segmentation mode draws colored masks with transparency overlays
- Uses top-k selection (default: 300 detections) for efficient processing

---

## Configuration

The inference engine supports various configuration options that can be modified in `src/main.cpp`:

- **Model Type**: `ModelType::DETECTION` or `ModelType::SEGMENTATION`
- **Resolution**: Set to `0` for auto-detection from model, or specify manually (e.g., `432`, `560`)
- **Confidence Threshold**: Default `0.5` (adjustable in `Config::threshold`)
- **Max Detections**: Default `300` for top-k selection (adjustable in `Config::max_detections`)
- **Mask Threshold**: Default `0.0` for binary mask generation (adjustable in `Config::mask_threshold`)
- **Normalization**: ImageNet mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`

### Example Custom Configuration

```cpp
Config config;
config.resolution = 0;              // Auto-detect
config.threshold = 0.6f;            // Higher confidence threshold
config.max_detections = 100;        // Fewer detections
config.mask_threshold = 0.5f;       // More conservative masks
config.model_type = ModelType::SEGMENTATION;
```

---

## Building 

### Quick Start: Build with ONNX Runtime (Default)

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/usr/bin/clang-15 \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++-15

cmake --build build --parallel
```

This builds an executable with ONNX Runtime backend compiled in.

### Build with TensorRT Backend

```bash
cmake -S . -B build -G Ninja \
  -DUSE_ONNX_RUNTIME=OFF \
  -DUSE_TENSORRT=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/usr/bin/clang-15 \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++-15

cmake --build build --parallel
```

**Note**: Requires TensorRT and CUDA installed. See [backends documentation](docs/backends.md) for setup.

> **🆕 TensorRT 10.x Support**: The project now supports TensorRT 10.x with automatic API compatibility. See [TensorRT 10.x Migration Guide](docs/TENSORRT_10_MIGRATION.md) for installation and upgrade instructions.

### Build Options

- `-DUSE_ONNX_RUNTIME=ON/OFF` - Enable ONNX Runtime backend (default: ON)
- `-DUSE_TENSORRT=ON/OFF` - Enable TensorRT backend (default: OFF)
- `-DCMAKE_BUILD_TYPE=Release/Debug` - Build configuration

**Important**: Only ONE backend can be enabled at a time. The backend is compiled into the binary for optimal performance.

For detailed build instructions and troubleshooting, see:
- [Backends Documentation](docs/backends.md) - Backend comparison and setup
- [Compile-Time Backend Guide](docs/COMPILE_TIME_BACKEND.md) - Understanding compile-time selection
- [TensorRT 10.x Migration Guide](docs/TENSORRT_10_MIGRATION.md) - TensorRT 10.x support and migration

---

## Technical Details

### Model Outputs

#### Detection Model
- **dets**: `float32[batch, num_queries, 4]` - Bounding boxes in `cxcywh` format (normalized)
- **labels**: `float32[batch, num_queries, num_classes]` - Class logits

#### Segmentation Model
- **dets**: `float32[batch, num_queries, 4]` - Bounding boxes in `cxcywh` format (normalized)
- **labels**: `float32[batch, num_queries, num_classes]` - Class logits
- **masks**: `float32[batch, num_queries, mask_h, mask_w]` - Segmentation masks (e.g., 108x108)

### Processing Pipeline

1. **Preprocessing**:
   - Resize image to model input resolution (auto-detected)
   - Convert BGR to RGB
   - Normalize with ImageNet statistics
   - Convert to CHW format

2. **Inference**:
   - Run ONNX Runtime session
   - Auto-detect output tensor names from model

3. **Postprocessing**:
   - **Detection**: Select predictions above confidence threshold
   - **Segmentation**: 
     - Apply sigmoid to class logits
     - Top-k selection across all classes and queries
     - Resize masks to original image dimensions using bilinear interpolation
     - Apply threshold to create binary masks
   - Convert bounding boxes from `cxcywh` to `xyxy` format
   - Scale coordinates to original image size

4. **Visualization**:
   - Draw bounding boxes with class labels
   - Overlay segmentation masks with transparency (alpha = 0.5)
   - Use deterministic colors based on class IDs

---

## Acknowledgements

- The RF-DETR model used in this project is sourced from **Roboflow**, special thanks to the Roboflow team — check out their [GitHub repository](https://github.com/roboflow/rf-detr) and [site](https://blog.roboflow.com/rf-detr/).
- **Postprocessing implementation** is based on Roboflow's reference implementations:
  - Detection postprocessing: [benchmark_rfdetr.py](https://github.com/roboflow/single_artifact_benchmarking/blob/main/sab/models/benchmark_rfdetr.py)
  - Instance segmentation postprocessing: [benchmark_rfdetr_seg.py](https://github.com/roboflow/single_artifact_benchmarking/blob/main/sab/models/benchmark_rfdetr_seg.py)
