# Compile-Time Backend Selection

## Overview

The RF-DETR inference project uses **compile-time backend selection** instead of runtime selection. This means the backend (ONNX Runtime or TensorRT) is determined when you compile the project, not when you run the executable.

## Why Compile-Time Selection?

1. **Smaller Binaries**: Only one backend is compiled into the executable, reducing binary size
2. **No Runtime Overhead**: No conditional checks or function pointer indirection at runtime
3. **Cleaner Dependencies**: You only need to install the backend you're actually using
4. **Simpler Deployment**: The executable is self-contained with its chosen backend

## Building with Different Backends

### ONNX Runtime (Default)

```bash
mkdir build && cd build
cmake -DUSE_ONNX_RUNTIME=ON -DUSE_TENSORRT=OFF ..
make -j$(nproc)
```

The executable will use ONNX Runtime for all inference operations.

### TensorRT

```bash
mkdir build && cd build
cmake -DUSE_ONNX_RUNTIME=OFF -DUSE_TENSORRT=ON ..
make -j$(nproc)
```

The executable will use TensorRT for all inference operations.

**Note**: You cannot enable both backends simultaneously. The CMake configuration will enforce this.

## Usage

After building, the usage is the same regardless of backend:

```bash
# Detection
./inference_app ./model.onnx ./image.jpg ./coco_labels.txt

# Segmentation
./inference_app ./model.onnx ./image.jpg ./coco_labels.txt --segmentation
```

The backend is already "baked into" the executable during compilation.

## Implementation Details

### Backend Factory Pattern

The backend selection uses a compile-time factory function in `src/backends/inference_backend.cpp`:

```cpp
std::unique_ptr<InferenceBackend> create_backend() {
#ifdef USE_ONNX_RUNTIME
    return std::make_unique<OnnxRuntimeBackend>();
#elif defined(USE_TENSORRT)
    return std::make_unique<TensorRTBackend>();
#else
    #error "No backend enabled. Build with -DUSE_ONNX_RUNTIME=ON or -DUSE_TENSORRT=ON"
#endif
}
```

This ensures that only the code for the selected backend is compiled into the binary.

### Namespace Organization

All backend code is organized under the `rfdetr::backend` namespace:

```
src/backends/
├── inference_backend.hpp       # Abstract base class
├── inference_backend.cpp       # Factory function
├── onnx_runtime_backend.hpp    # ONNX Runtime implementation
├── onnx_runtime_backend.cpp
├── tensorrt_backend.hpp        # TensorRT implementation
└── tensorrt_backend.cpp
```

### Conditional Compilation Guards

Each backend implementation is wrapped in preprocessor guards:

```cpp
#ifdef USE_ONNX_RUNTIME
// ONNX Runtime specific code
#endif

#ifdef USE_TENSORRT
// TensorRT specific code
#endif
```

This allows the code to compile even if a backend's dependencies are not installed, as long as that backend is not selected.

## Deployment Scenarios

### Scenario 1: Development Machine with Both Backends

Build two separate executables:

```bash
# Build ONNX Runtime version
mkdir build-onnx && cd build-onnx
cmake -DUSE_ONNX_RUNTIME=ON -DUSE_TENSORRT=OFF ..
make -j$(nproc)
cp inference_app ../inference_app_onnx

# Build TensorRT version
cd ..
mkdir build-tensorrt && cd build-tensorrt
cmake -DUSE_ONNX_RUNTIME=OFF -DUSE_TENSORRT=ON ..
make -j$(nproc)
cp inference_app ../inference_app_tensorrt
```

### Scenario 2: Production Deployment

1. Choose the appropriate backend for your hardware
2. Build once with that backend
3. Deploy the single executable with its runtime dependencies

For ONNX Runtime:
- Deploy `inference_app` + `libonnxruntime.so.1.21.0`

For TensorRT:
- Deploy `inference_app` + TensorRT libraries (libnvinfer, libcudart, etc.)

## Benefits Over Runtime Selection

| Aspect | Compile-Time | Runtime |
|--------|-------------|---------|
| Binary Size | Smaller (one backend) | Larger (all backends) |
| Startup Time | Faster | Slower (backend detection) |
| Dependencies | Minimal | All backends required |
| Runtime Overhead | None | Function pointer calls |
| Configuration | Build-time CMake flag | Command-line argument |
| Error Detection | Compile-time | Runtime |

## Troubleshooting

### "No backend enabled" Error

If you see this error during compilation:

```
#error "No backend enabled. Build with -DUSE_ONNX_RUNTIME=ON or -DUSE_TENSORRT=ON"
```

Make sure you've specified at least one backend in your CMake command:

```bash
cmake -DUSE_ONNX_RUNTIME=ON ..
```

### "Backend not available" Error (Old Code)

This error no longer exists! The old runtime backend selection has been completely removed.

### Missing Backend Libraries

If linking fails due to missing backend libraries:
- For ONNX Runtime: The library is automatically downloaded during CMake configuration
- For TensorRT: Install TensorRT according to the main README instructions

## Migration from Runtime Selection

If you have old code using the `--backend` parameter:

**Before:**
```bash
./inference_app model.onnx image.jpg labels.txt --backend onnx
./inference_app model.onnx image.jpg labels.txt --backend tensorrt
```

**After:**
```bash
# Build ONNX version
cmake -DUSE_ONNX_RUNTIME=ON ..
make
./inference_app model.onnx image.jpg labels.txt

# Build TensorRT version (separate executable)
cmake -DUSE_TENSORRT=ON ..
make
./inference_app model.onnx image.jpg labels.txt
```

The `--backend` parameter has been completely removed from the codebase.
