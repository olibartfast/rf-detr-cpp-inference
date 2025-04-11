# RFDETR Inference

[![C++](https://img.shields.io/badge/language-C++20-blue.svg)](https://en.cppreference.com/w/cpp)
[![CMake](https://img.shields.io/badge/build%20system-CMake-blue.svg)](https://cmake.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)](https://github.com/<your-username>/rfdetr_inference/actions)

**`rfdetr_inference`** is a C++ project for performing object detection inference using the RF-DETR model with ONNX Runtime and OpenCV. It provides a pipeline to preprocess images, run inference, post-process results, and visualize detections. The project includes unit and integration tests using Google Test.

---

## Table of Contents
- [Dependencies](#dependencies)
- [Model Setup](#model-setup)
- [Installation](#installation)
- [Usage](#usage)
- [Building and Testing](#building-and-testing)
- [Project Structure](#project-structure)
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

## Building and Testing

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

This builds:

- `inference_app`
- `unit_tests`
- `integration_tests`

### Run Unit Tests

```bash
./build/unit_tests
```

### Run Integration Tests

```bash
./build/integration_tests
```

> Note: Ensure the ONNX model path in `integration_test_rfdetr_inference.cpp` is correct.

### Run All Tests

```bash
ninja -C build run_tests
```

---

## Project Structure

```
rfdetr_inference/
├── CMakeLists.txt
├── README.md
├── src/
│   ├── rfdetr_inference.hpp
│   ├── rfdetr_inference.cpp
│   └── main.cpp
├── tests/
│   ├── unit/
│   │   └── test_rfdetr_inference.cpp
│   └── integration/
│       └── integration_test_rfdetr_inference.cpp
├── data/
│   ├── test_image.jpg
│   ├── test_labels.txt
│   └── test_output.jpg
└── build/
```

---

## Acknowledgements

- The RF-DETR model used in this project is sourced from **Roboflow**, special thanks to the Roboflow team — check out their [GitHub repository](https://github.com/roboflow/rf-detr) and [site](https://blog.roboflow.com/rf-detr/).
