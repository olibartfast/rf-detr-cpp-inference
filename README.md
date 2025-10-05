# RFDETR Inference

[![C++](https://img.shields.io/badge/language-C++20-blue.svg)](https://en.cppreference.com/w/cpp)
[![CMake](https://img.shields.io/badge/build%20system-CMake-blue.svg)](https://cmake.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

C++ project for performing object detection inference using the RF-DETR model with ONNX Runtime and OpenCV. 
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

This project uses the RF-DETR model from Roboflow.

1. **Visit the RF-DETR Repository**:
   - Go to the [RF-DETR GitHub repository](https://github.com/roboflow/rf-detr) for model details.
   - Read the [Roboflow blog](https://blog.roboflow.com/rf-detr/) for an overview.

2. **Download the ONNX Model**:
   - Follow instructions in the RF-DETR repo to get the model in ONNX format.
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

```bash
./build/inference_app /path/to/inference_model.onnx /path/to/image.jpg /path/to/coco-labels-91.txt
```

- The output image is saved as `output_image.jpg`
- Detection results (bounding boxes, labels, and scores) are printed to the console.

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
ninja -C build
```

---

## Acknowledgements

- The RF-DETR model used in this project is sourced from **Roboflow**, special thanks to the Roboflow team — check out their [GitHub repository](https://github.com/roboflow/rf-detr) and [site](https://blog.roboflow.com/rf-detr/).
- If you use RF-DETR in your research or projects, please cite the original authors as described in their [repository](https://github.com/roboflow/rf-detr#citation).
