# RF-DETR C++ Inference

[![C++](https://img.shields.io/badge/language-C++20-blue.svg)](https://en.cppreference.com/w/cpp)
[![CMake](https://img.shields.io/badge/build%20system-CMake-blue.svg)](https://cmake.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

C++ project for performing object detection and instance segmentation inference using the RF-DETR model with ONNX Runtime and OpenCV. 
---

## Table of Contents
- [Dependencies](#dependencies)
- [Model Setup](#model-setup)
- [Installation](#installation)
- [Usage](#usage)
- [Building](#building)
- [Acknowledgements](#acknowledgements)

---

## Dependencies

- **C++20 Compiler**: Clang++ 15 or compatible (e.g., `clang++-15`)
- **CMake**: Version 3.12 or higher
- **ONNX Runtime**: Version 1.21.0 (automatically downloaded during build)
- **OpenCV**: Version 4.x (e.g., install via `sudo apt-get install libopencv-dev` on Ubuntu)
- **Google Test**: Version 1.12.1 (automatically fetched during build)
- **Ninja**: Optional but recommended (`sudo apt-get install ninja-build`)

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

## Usage

### Prepare Input Files

- The RF-DETR ONNX model file (e.g., `inference_model.onnx`)
- An input image (e.g., `image.jpg`)
- A COCO labels file (e.g., `coco-labels-91.txt`)

### Run Inference

After building the project (see below), run:

#### Object Detection

```bash
./build/inference_app /path/to/model.onnx /path/to/image.jpg /path/to/coco-labels-91.txt
```

#### Instance Segmentation

```bash
./build/inference_app /path/to/model.onnx /path/to/image.jpg /path/to/coco-labels-91.txt --segmentation
```

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

### Configure the Build

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug \
-DCMAKE_C_COMPILER=/usr/bin/clang-15 \
-DCMAKE_CXX_COMPILER=/usr/bin/clang++-15
```

### Build the Project

```bash
# Using cmake (works with any generator)
cd build && cmake --build . --parallel

# Or using ninja directly (if configured with -G Ninja)
ninja -C build
```

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
- If you use RF-DETR in your research or projects, please cite the original authors as described in their [repository](https://github.com/roboflow/rf-detr#citation).
