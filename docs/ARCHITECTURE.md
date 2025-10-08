# Architecture Overview

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Application Layer                        │
│                         (main.cpp)                               │
│  - Command-line parsing                                         │
│  - Result visualization                                         │
│  Note: Backend selected at compile time                         │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                      RFDETRInference                            │
│                   (rfdetr_inference.cpp)                        │
│  - Image preprocessing (resize, normalize)                      │
│  - Postprocessing (NMS, mask resize)                           │
│  - Visualization (bounding boxes, masks)                       │
│  - Backend-agnostic inference coordination                      │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│            rfdetr::backend::InferenceBackend                    │
│              (backends/inference_backend.hpp)                   │
│                    [Interface/Strategy]                         │
│                                                                 │
│  + initialize(model_path, input_shape)                         │
│  + run_inference(input_data, input_shape)                      │
│  + get_output_data(index, data, size)                          │
│  + get_output_shape(index)                                     │
│  + get_backend_name()                                          │
└────────────────────┬───────────────────────────────────────────┘
                     │
                     │ (Compile-time selection via #ifdef)
                     │
        ┌────────────┴────────────┐
        ▼                         ▼
┌──────────────────┐      ┌──────────────────┐
│ OnnxRuntime      │      │  TensorRT        │
│ Backend          │      │  Backend         │
│ (backends/onnx_  │      │  (backends/      │
│  runtime_backend │      │   tensorrt_      │
│  .cpp)           │      │   backend.cpp)   │
│                  │      │                  │
│ - Cross-platform │      │ - NVIDIA GPU     │
│ - CPU/GPU        │      │ - Optimized      │
│ - Easy setup     │      │ - FP16 support   │
└────────┬─────────┘      └────────┬─────────┘
         │                         │
         ▼                         ▼
┌──────────────────┐      ┌──────────────────┐
│  ONNX Runtime    │      │   TensorRT       │
│  Library         │      │   Library        │
│  (1.21.0)        │      │   (10.x)         │
└──────────────────┘      └──────────────────┘
```

## Design Pattern: Strategy

### Components

#### Context (`RFDETRInference`)
- Maintains reference to a strategy object (`InferenceBackend`)
- Delegates inference operations to the strategy
- Performs backend-agnostic pre/post-processing

#### Strategy Interface (`InferenceBackend`)
- Declares operations common to all inference backends
- Pure virtual methods for initialization, inference, output retrieval

#### Concrete Strategies
- **OnnxRuntimeBackend**: ONNX Runtime implementation
- **TensorRTBackend**: TensorRT implementation

#### Factory (`create_backend()`)
- Creates appropriate backend at **compile time** using preprocessor directives
- No runtime overhead - backend determined during compilation
- Returns `std::unique_ptr<InferenceBackend>` to the selected backend

## Data Flow

### 1. Initialization
```
main() 
  → Config{}
  → RFDETRInference(config)
    → create_backend()  // Compile-time selection via #ifdef
      → #ifdef USE_TENSORRT: new TensorRTBackend()
      → #elif USE_ONNX_RUNTIME: new OnnxRuntimeBackend()
        → backend->initialize(model_path)
          → [TensorRT]: Build/Load TensorRT engine, Allocate CUDA buffers
          → [ONNX Runtime]: Create session, Setup providers
```

### 2. Inference
```
main()
  → inference.preprocess_image(image_path)
    → Load image with OpenCV
    → Resize, normalize, convert to CHW
    → Return preprocessed data
  → inference.run_inference(data)
    → backend->run_inference(data)
      → [ONNX Runtime]: Create tensor, run session
      → [TensorRT]: Copy to GPU, execute, copy from GPU
    → Cache output tensors
  → inference.postprocess_outputs(...)
    → Apply sigmoid to logits
    → Top-k selection
    → Convert bounding boxes
    → Resize masks (if segmentation)
    → Return results
```

### 3. Visualization
```
main()
  → inference.draw_detections(image, boxes, classes, scores)
    → Draw bounding boxes
    → Draw class labels
  → inference.draw_segmentation_masks(image, ...)
    → Apply colored masks with alpha blending
    → Draw bounding boxes and labels
  → inference.save_output_image(image, path)
    → Save with OpenCV
```

## Compilation Flow

### Conditional Compilation (Compile-Time Backend Selection)

```
CMakeLists.txt
  ├─ USE_ONNX_RUNTIME=ON (Default)
  │   ├─ Download ONNX Runtime
  │   ├─ Link libonnxruntime.so
  │   ├─ Define USE_ONNX_RUNTIME
  │   └─ Compile backends/onnx_runtime_backend.cpp
  │
  ├─ USE_TENSORRT=ON (Optional, mutually exclusive)
  │   ├─ Find TensorRT installation
  │   ├─ Link libnvinfer.so, libnvonnxparser.so, libcudart.so
  │   ├─ Define USE_TENSORRT
  │   └─ Compile backends/tensorrt_backend.cpp
  │
  └─ Always compile:
      ├─ rfdetr_inference.cpp
      └─ backends/inference_backend.cpp (factory with #ifdef)

Note: Only ONE backend can be enabled at a time.
This results in smaller binaries and eliminates runtime overhead.
```

### Build Targets

```
inference_app (executable)
  └─ rfdetr_inference_lib (static library)
      ├─ rfdetr_inference.o
      ├─ inference_backend.o
      ├─ [onnx_runtime_backend.o] (if USE_ONNX_RUNTIME)
      └─ [tensorrt_backend.o] (if USE_TENSORRT)
```

## Memory Management

### ONNX Runtime Backend
```
Host Memory:
  ├─ Input tensor (std::vector<float>)
  ├─ Output tensors (std::vector<Ort::Value>)
  └─ Cached output data (std::vector<std::vector<float>>)

GPU Memory (if using GPU provider):
  ├─ Managed internally by ONNX Runtime
  └─ Automatic device memory allocation
```

### TensorRT Backend
```
Host Memory:
  ├─ Input tensor (std::vector<float>)
  ├─ Output buffers (std::vector<std::vector<float>>)
  └─ Cached output data

GPU Memory:
  ├─ Device input buffer (cudaMalloc)
  ├─ Device output buffers (cudaMalloc)
  └─ TensorRT engine workspace (managed)

Copies:
  Host → Device (input)
  Device → Host (outputs)
```

## Error Handling

```
┌─────────────────┐
│ main()          │
│                 │
│ try {           │
│   RFDETRInfer   │  ← May throw if backend unavailable
│   preprocess()  │  ← May throw if image not found
│   run_inference │  ← May throw if inference fails
│   postprocess() │  ← May throw if output shape mismatch
│   draw()        │
│   save()        │  ← May throw if save fails
│ }               │
│ catch (...)     │  ← Catch and report errors
│                 │
└─────────────────┘
```

## Extension Points

### Adding a New Backend

1. **Create header**: `src/new_backend.hpp`
```cpp
#pragma once
#ifdef USE_NEW_BACKEND
#include "inference_backend.hpp"

class NewBackend : public InferenceBackend {
    // Implement interface
};
#endif
```

2. **Implement**: `src/new_backend.cpp`
```cpp
#ifdef USE_NEW_BACKEND
#include "new_backend.hpp"
// Implementation
#endif
```

3. **Update factory**: `src/inference_backend.cpp`
```cpp
#ifdef USE_NEW_BACKEND
#include "new_backend.hpp"
#endif

std::unique_ptr<InferenceBackend> create_backend(BackendType type) {
    switch (type) {
        case BackendType::NEW_BACKEND:
#ifdef USE_NEW_BACKEND
            return std::make_unique<NewBackend>();
#else
            throw std::runtime_error("Backend not available");
#endif
    }
}
```

4. **Update CMake**: `CMakeLists.txt`
```cmake
option(USE_NEW_BACKEND "Enable new backend" OFF)

if(USE_NEW_BACKEND)
    find_package(NewBackendLib REQUIRED)
    list(APPEND RFDETR_SOURCES src/new_backend.cpp)
    target_link_libraries(rfdetr_inference_lib ${NEW_BACKEND_LIBS})
    target_compile_definitions(rfdetr_inference_lib PUBLIC USE_NEW_BACKEND)
endif()
```

## Thread Safety

### Current Implementation
- ❌ Not thread-safe
- Single-threaded execution assumed
- Each `RFDETRInference` instance should be used by one thread

### Future Enhancement
```cpp
class ThreadSafeRFDETRInference : public RFDETRInference {
    std::mutex inference_mutex_;
    
    void run_inference(std::span<const float> data) override {
        std::lock_guard<std::mutex> lock(inference_mutex_);
        RFDETRInference::run_inference(data);
    }
};
```

## Performance Profiling Points

```
┌──────────────────────────────────────────────────────┐
│                   Timing Breakdown                   │
├──────────────────────────────────────────────────────┤
│ 1. Image Loading       │ TBD   │ OpenCV imread   │
│ 2. Preprocessing       │ TBD  │ Resize/normalize│
│ 3. Inference           │ TBD │ Backend-specific│
│    ├─ ONNX Runtime CPU │ TBD│                 │
│    ├─ ONNX Runtime GPU │ TBD │                 │
│    └─ TensorRT GPU     │ TBD  │                 │
│ 4. Postprocessing      │ TBD  │ Sigmoid/top-k   │
│ 5. Mask Resize (seg)   │ TBD  │ Bilinear interp │
│ 6. Visualization       │ TBD  │ Drawing         │
│ 7. Image Saving        │ TBD   │ OpenCV imwrite  │
└──────────────────────────────────────────────────────┘
```

## Dependency Graph

```
Application (main)
    ↓
RFDETRInference
    ↓
InferenceBackend ←──────┐
    ↓                   │
    ├─→ ONNX Runtime    │
    │       ↓           │
    │   onnxruntime.so  │
    │                   │
    └─→ TensorRT        │
            ↓           │
        libnvinfer.so   │
        libnvonnxparser │
        libcudart.so    │
                        │
OpenCV (all paths) ─────┘
    ↓
libopencv_*.so
```

