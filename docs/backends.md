# Building with Different Inference Backends

This project supports multiple inference backends using the Strategy Pattern with **compile-time backend selection**. You choose which backend to use when building the project, resulting in a smaller binary with optimal performance.

## Backend Options

### 1. ONNX Runtime (Default)
- **Platform**: Cross-platform (CPU and GPU)
- **Requirements**: None (automatically downloaded during build)
- **Best for**: Development, CPU inference, maximum compatibility

### 2. TensorRT
- **Platform**: NVIDIA GPUs only
- **Requirements**: CUDA, TensorRT libraries installed
- **Best for**: Production deployment on NVIDIA GPUs, maximum performance

## Build Configurations

### Build with ONNX Runtime only (Default)

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/usr/bin/clang-15 \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++-15

cmake --build build --parallel
```

Or explicitly:
```bash
cmake -S . -B build -G Ninja \
  -DUSE_ONNX_RUNTIME=ON \
  -DUSE_TENSORRT=OFF \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build --parallel
```

### Build with TensorRT only

```bash
cmake -S . -B build -G Ninja \
  -DUSE_ONNX_RUNTIME=OFF \
  -DUSE_TENSORRT=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/usr/bin/clang-15 \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++-15

cmake --build build --parallel
```

**Note**: You need to have TensorRT and CUDA installed on your system. Set `TENSORRT_ROOT` environment variable if TensorRT is installed in a non-standard location:

```bash
export TENSORRT_ROOT=/path/to/tensorrt
cmake -S . -B build -G Ninja -DUSE_TENSORRT=ON ...
```

### Note: Only One Backend Per Build

**Important**: You cannot enable both backends simultaneously. The backend is compiled into the binary for optimal performance.

To use both backends, build two separate executables:

```bash
# Build ONNX Runtime version
mkdir build-onnx && cd build-onnx
cmake .. -G Ninja -DUSE_ONNX_RUNTIME=ON -DUSE_TENSORRT=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build . --parallel
cp inference_app ../inference_app_onnx

# Build TensorRT version
cd ..
mkdir build-tensorrt && cd build-tensorrt
cmake .. -G Ninja -DUSE_ONNX_RUNTIME=OFF -DUSE_TENSORRT=ON -DCMAKE_BUILD_TYPE=Release
cmake --build . --parallel
cp inference_app ../inference_app_tensorrt
```

See [COMPILE_TIME_BACKEND.md](COMPILE_TIME_BACKEND.md) for more details on the compile-time selection approach.

## Runtime Usage

The backend is compiled into the executable, so there's no `--backend` flag. Simply run the executable you built:

### Detection

```bash
./build/inference_app model.onnx image.jpg coco-labels-91.txt
```

### Segmentation

```bash
./build/inference_app model.onnx image.jpg coco-labels-91.txt --segmentation
```

**For TensorRT**: On first run, the engine will be built from the ONNX model, which may take a few minutes. The engine is cached as `model.trt` for subsequent runs.

# TensorRT
./build/inference_app model.onnx image.jpg coco-labels-91.txt --segmentation --backend tensorrt
```

## TensorRT Installation

### Ubuntu/Debian

1. Install CUDA Toolkit (12.x recommended, 11.x+ supported):
```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get install cuda
```

2. Download and install TensorRT 10.x from [NVIDIA's website](https://developer.nvidia.com/tensorrt)
   (TensorRT 8.x+ also supported)

3. Extract and set environment variables:
```bash
tar -xzvf TensorRT-10.*.tar.gz
export TENSORRT_ROOT=/path/to/TensorRT-10.*
export LD_LIBRARY_PATH=$TENSORRT_ROOT/lib:$LD_LIBRARY_PATH
```

### Verify Installation

```bash
# Check CUDA
nvcc --version

# Check TensorRT headers
ls $TENSORRT_ROOT/include/NvInfer.h

# Check TensorRT libraries
ls $TENSORRT_ROOT/lib/libnvinfer.so
```

## Performance Comparison

| Backend | Platform | Latency | Throughput | Setup Time |
|---------|----------|---------|------------|------------|
| ONNX Runtime | CPU | TBD | Medium | Instant |
| ONNX Runtime | GPU | TBD | High | Instant |
| TensorRT | GPU | TBD | High | First run: 1-5 min (cached) |

## Troubleshooting

### ONNX Runtime Issues

**Q**: ONNX Runtime download fails  
**A**: Check internet connection or manually download from [ONNX Runtime releases](https://github.com/microsoft/onnxruntime/releases)

### TensorRT Issues

**Q**: "TensorRT not found" error  
**A**: Set `TENSORRT_ROOT` environment variable or install TensorRT in standard location

**Q**: "CUDA runtime library not found"  
**A**: Install CUDA Toolkit and ensure `/usr/local/cuda/lib64` is in `LD_LIBRARY_PATH`

**Q**: "TensorRT backend not available" at runtime  
**A**: Rebuild with `-DUSE_TENSORRT=ON`

**Q**: TensorRT engine build is slow  
**A**: This is normal for the first run. The engine is cached as `.trt` file for reuse

### Build Issues

**Q**: "At least one backend must be enabled" error  
**A**: Enable at least one backend: `-DUSE_ONNX_RUNTIME=ON` or `-DUSE_TENSORRT=ON`

## Architecture Overview

The project uses the **Strategy Pattern** for backend abstraction:

```
InferenceBackend (Interface)
    ├── OnnxRuntimeBackend (ONNX Runtime implementation)
    └── TensorRTBackend (TensorRT implementation)
```

Each backend implements:
- `initialize()` - Load and configure the model
- `run_inference()` - Execute inference
- `get_output_data()` - Retrieve results
- `get_output_shape()` - Get tensor dimensions

