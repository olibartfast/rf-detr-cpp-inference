# RF-DETR C++ Inference

[![C++](https://img.shields.io/badge/language-C++20-blue.svg)](https://en.cppreference.com/w/cpp)
[![CMake](https://img.shields.io/badge/build%20system-CMake-blue.svg)](https://cmake.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

C++ project for performing object detection and instance segmentation inference using the RF-DETR model with **multiple inference backends** (ONNX Runtime and TensorRT) and OpenCV.

---

## Table of Contents
- [Dependencies](#dependencies)
- [Model Setup](#model-setup)
- [Installation](#installation)
- [Building](#building)
- [Usage](#usage)
- [Configuration](#configuration)
- [Technical Details](#technical-details)
- [Acknowledgements](#acknowledgements)
- [Code Quality Tools](#code-quality-tools)

---

## Dependencies

### Required (All Backends)
- **C++20 Compiler**: Clang 15+ or GCC 12+ (e.g., `clang++-15` or `g++-12`)
- **CMake**: Version 3.12 or higher
- **OpenCV**: Version 4.x (modules: core, imgproc, imgcodecs)
- **Google Test**: Version 1.12.1 (automatically fetched during build)
- **Ninja**: Optional but recommended (`sudo apt-get install ninja-build`)

### Backend-Specific Dependencies

#### ONNX Runtime Backend (Default)
- **ONNX Runtime**: Version 1.21.0 (automatically downloaded during build)
- **Platform**: Linux, Windows, macOS
- **Acceleration**: CPU and GPU (CUDA/DirectML)

#### TensorRT Backend (Optional)
- **TensorRT**: Version 10.x or 8.x+ (automatically downloaded during build if not found)
- **CUDA Toolkit**: Used version 13.x  - **must be installed manually**
- **Platform**: Linux with NVIDIA GPU
- **Acceleration**: NVIDIA GPU only
- **Note**: TensorRT libraries are automatically configured with RPATH, no LD_LIBRARY_PATH needed

---

## Model Setup

This project supports both RF-DETR detection and segmentation models from Roboflow.

1. **Visit the RF-DETR Repository**:
   - Go to the [RF-DETR GitHub repository](https://github.com/roboflow/rf-detr) for model details.
   - Read the [Roboflow blog](https://blog.roboflow.com/rf-detr/) for an overview.

2. **Download the ONNX Model**:
   - Follow instructions in the [export documentation](docs/export.md) to export models in ONNX format.
   - **Tested with**: `rfdetr[onnxexport]==1.4.3` (Python ≤ 3.11 required)
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
sudo apt-get install -y cmake

# Compiler - either clang or gcc (any C++20-capable version):
sudo apt-get install -y clang-15
# or use the system default gcc (no install needed if already present)

# OpenCV (the full -dev package is needed for CMake integration):
sudo apt-get install -y libopencv-dev

# Optional (faster incremental builds):
sudo apt-get install -y ninja-build

# Optional (linting and formatting — use version 18+ for GCC 14 compatibility):
sudo apt-get install -y clang-format-18 clang-tidy-18
```

---

## Building 

### Backend Selection

This project uses **compile-time backend selection**. Choose your backend when building:

| Backend | Best For | Pros | Cons |
|---------|----------|------|------|
| **ONNX Runtime** | Development, CPU inference | Cross-platform, easy setup | Slower than TensorRT on GPU |
| **TensorRT** | Production on NVIDIA GPUs | Maximum performance | GPU-only, requires CUDA/TensorRT |

**Important**: Only ONE backend can be enabled at a time. The backend is compiled into the binary for optimal performance and smaller binary size.

### Format Code (Optional)

If you have `clang-format-18` installed, you can check and auto-format all source files:

```bash
# Check for formatting issues (no changes made):
find src tests -name '*.cpp' -o -name '*.hpp' | xargs clang-format-18 --dry-run --Werror

# Auto-format in place:
find src tests -name '*.cpp' -o -name '*.hpp' | xargs clang-format-18 -i
```

### Static Analysis (Optional)

If you have `clang-tidy-18` installed, you can run static analysis using the compile commands database:

```bash
# Generate compile_commands.json first:
cmake -S . -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

# Run clang-tidy on project sources:
find src -name '*.cpp' | xargs clang-tidy-18 -p build
```

### Cppcheck (Optional)

If you have `cppcheck` installed, you can run additional static analysis:

```bash
# Install cppcheck:
sudo apt-get install -y cppcheck

# Run manually:
cppcheck --enable=all --std=c++20 \
  --suppress=missingIncludeSystem \
  --suppress=unmatchedSuppression \
  --error-exitcode=1 \
  -I src src/
```

This is also run automatically on every commit via pre-commit (see [Pre-commit](#pre-commit-optional)).

### Sanitizers (Optional)

Build with AddressSanitizer and UndefinedBehaviorSanitizer enabled via `-DSANITIZERS=ON`. Use a separate build directory to keep sanitizer and release builds independent:

```bash
cmake -S . -B build-san \
  -DCMAKE_BUILD_TYPE=Debug \
  -DSANITIZERS=ON

cmake --build build-san --parallel

# Run tests under sanitizers:
./build-san/unit_tests
./build-san/integration_tests
```

Sanitizers catch memory errors, use-after-free, undefined behaviour, and integer overflow at runtime with minimal code changes.

### Pre-commit (Optional)

[pre-commit](https://pre-commit.com/) runs `clang-format` and `cppcheck` automatically on every commit:

```bash
pip install pre-commit
pre-commit install           # install the git hook
pre-commit run --all-files   # run manually on all files
```

### Strict Compilation (Optional)

To treat all compiler warnings as errors (as CI does), pass `-DWERROR=ON`:

```bash
cmake -S . -B build -DWERROR=ON
cmake --build build
```

### Build with ONNX Runtime (Default)

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/usr/bin/clang-15 \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++-15

cmake --build build --parallel
```

Using the system default compiler (typically gcc):

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build --parallel
```

If you don't have Ninja installed, drop `-G Ninja` to use Make instead:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build --parallel
```

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

**What happens**:
- TensorRT 10.13.3.9 is automatically downloaded if not found
- Libraries are configured with RPATH - no need to set `LD_LIBRARY_PATH`
- The executable will use TensorRT for inference
- Requires CUDA 12.x (or 11.x+) installed manually
- Pre-built `.engine` or `.trt` files are loaded directly, skipping ONNX-to-TensorRT conversion

### Build Options

- `-DUSE_ONNX_RUNTIME=ON/OFF` - Enable ONNX Runtime backend (default: ON)
- `-DUSE_TENSORRT=ON/OFF` - Enable TensorRT backend (default: OFF)
- `-DCMAKE_BUILD_TYPE=Release/Debug` - Build configuration
- `-DSANITIZERS=ON/OFF` - Enable AddressSanitizer + UndefinedBehaviorSanitizer (default: OFF)
- `-DWERROR=ON/OFF` - Treat compiler warnings as errors (default: OFF)

---

## Usage

### Prepare Input Files

- The RF-DETR model file (`.onnx` for ONNX Runtime, `.onnx`/`.engine`/`.trt` for TensorRT)
- An input image (e.g., `image.jpg`)
- A COCO labels file (e.g., `coco-labels-91.txt`)

### Run Inference

After building the project, run the inference application:

#### Object Detection

```bash
./build/inference_app /path/to/model.onnx /path/to/image.jpg /path/to/coco-labels-91.txt
```

#### Instance Segmentation

```bash
./build/inference_app /path/to/model.onnx /path/to/image.jpg /path/to/coco-labels-91.txt --segmentation
```

#### Using Pre-built TensorRT Engine

If you have a pre-built TensorRT engine file (`.engine` or `.trt`), use it directly:

```bash
./build/inference_app /path/to/model.engine /path/to/image.jpg /path/to/coco-labels-91.txt --segmentation
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

## Code Quality Tools

| Tool | Purpose | How to run |
|------|---------|------------|
| `clang-format-18` | Code formatting | `find src tests -name '*.cpp' -o -name '*.hpp' \| xargs clang-format-18 -i` |
| `clang-tidy-18` | Static analysis (AST-based) | `find src -name '*.cpp' \| xargs clang-tidy-18 -p build` |
| `cppcheck` | Static analysis (flow-based) | `cppcheck --enable=all --std=c++20 -I src src/` |
| ASan + UBSan | Runtime memory/UB detection | `-DSANITIZERS=ON` at configure time |
| pre-commit | Automates format + cppcheck on commit | `pre-commit install` |

---

## Acknowledgements

- The RF-DETR model used in this project is sourced from **Roboflow**, special thanks to the Roboflow team — check out their [GitHub repository](https://github.com/roboflow/rf-detr) and [site](https://blog.roboflow.com/rf-detr/).
- **Postprocessing implementation** is based on Roboflow's reference implementations:
  - Detection postprocessing: [benchmark_rfdetr.py](https://github.com/roboflow/single_artifact_benchmarking/blob/main/sab/models/benchmark_rfdetr.py)
  - Instance segmentation postprocessing: [benchmark_rfdetr_seg.py](https://github.com/roboflow/single_artifact_benchmarking/blob/main/sab/models/benchmark_rfdetr_seg.py)
