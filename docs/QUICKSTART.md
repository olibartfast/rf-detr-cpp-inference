# Quick Start Guide: Backend Selection

> **Note**: This project uses **compile-time backend selection**. Choose your backend when building, not at runtime.

## TL;DR

```bash
# ONNX Runtime (works anywhere)
cmake -S . -B build -G Ninja && cmake --build build
./build/inference_app model.onnx image.jpg labels.txt

# TensorRT (NVIDIA GPU only, maximum performance)
cmake -S . -B build -G Ninja -DUSE_ONNX_RUNTIME=OFF -DUSE_TENSORRT=ON && cmake --build build
./build/inference_app model.onnx image.jpg labels.txt
```

## Choose Your Backend

### 🖥️ I want CPU inference (works everywhere)
```bash
cmake -S . -B build -G Ninja
cmake --build build
./build/inference_app model.onnx image.jpg labels.txt
```

**Result**: Executable with ONNX Runtime backend compiled in.

### 🚀 I have an NVIDIA GPU and want maximum performance
```bash
# Install TensorRT first (see below)
cmake -S . -B build -G Ninja -DUSE_ONNX_RUNTIME=OFF -DUSE_TENSORRT=ON
cmake --build build
./build/inference_app model.onnx image.jpg labels.txt
```

**Result**: Executable with TensorRT backend compiled in.

### 🔀 I need to switch between backends
Build two separate executables:

```bash
# Build ONNX Runtime version
mkdir build-onnx && cd build-onnx
cmake .. -G Ninja -DUSE_ONNX_RUNTIME=ON -DUSE_TENSORRT=OFF
cmake --build .
cp inference_app ../inference_app_onnx

# Build TensorRT version
cd ..
mkdir build-tensorrt && cd build-tensorrt
cmake .. -G Ninja -DUSE_ONNX_RUNTIME=OFF -DUSE_TENSORRT=ON
cmake --build .
cp inference_app ../inference_app_tensorrt

# Now use whichever you need
cd ..
./inference_app_onnx model.onnx image.jpg labels.txt
./inference_app_tensorrt model.onnx image.jpg labels.txt
```

## Installing TensorRT (For GPU Users)

### Ubuntu 22.04

```bash
# 1. Install CUDA
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get install cuda

# 2. Download TensorRT from NVIDIA
# Visit: https://developer.nvidia.com/tensorrt
# Download: TensorRT 10.x GA for Linux x86_64 (tar package)
#          (TensorRT 8.x+ also supported)

# 3. Extract and setup
tar -xzvf TensorRT-10.*.tar.gz
export TENSORRT_ROOT=$(pwd)/TensorRT-10.*
export LD_LIBRARY_PATH=$TENSORRT_ROOT/lib:$LD_LIBRARY_PATH

# 4. Verify
ls $TENSORRT_ROOT/include/NvInfer.h  # Should exist
ls $TENSORRT_ROOT/lib/libnvinfer.so  # Should exist

# 5. Build with TensorRT
cmake -S . -B build -G Ninja -DUSE_TENSORRT=ON
cmake --build build
```

### Docker (Recommended for TensorRT)

```dockerfile
FROM nvcr.io/nvidia/tensorrt:23.10-py3

# Install dependencies
RUN apt-get update && apt-get install -y \
    clang-15 \
    cmake \
    ninja-build \
    libopencv-dev

# Clone and build
WORKDIR /workspace
COPY . .
RUN cmake -S . -B build -G Ninja -DUSE_TENSORRT=ON && \
    cmake --build build

CMD ["./build/inference_app", "model.onnx", "image.jpg", "labels.txt", "--backend", "tensorrt"]
```

## Comparison Table

| Aspect | ONNX Runtime | TensorRT |
|--------|--------------|----------|
| **Setup** | ✅ Automatic | ⚠️ Manual install |
| **Platforms** | ✅ Linux, Windows, macOS | ⚠️ Linux + NVIDIA GPU only |
| **CPU Support** | ✅ Yes | ❌ No |
| **GPU Support** | ✅ CUDA, DirectML | ✅ CUDA only |
| **First Run** | ✅ Instant | ⚠️ 1-5 min (builds engine) |
| **Subsequent Runs** | ✅ Fast | ✅ Very fast |
| **Latency (GPU)** | ~10-30ms | ~5-15ms |
| **Memory Usage** | Medium | Low |
| **Binary Size** | ~50MB | ~200MB |

## Common Issues

### "TensorRT not found"
```bash
# Set TENSORRT_ROOT
export TENSORRT_ROOT=/path/to/TensorRT-10.x
cmake -S . -B build -G Ninja -DUSE_TENSORRT=ON
```

### "CUDA runtime library not found"
```bash
# Add CUDA to library path
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### Wrong backend compiled in
```bash
# The backend is compiled into the binary. To use TensorRT, rebuild:
cmake -S . -B build -G Ninja -DUSE_ONNX_RUNTIME=OFF -DUSE_TENSORRT=ON
cmake --build build

# Or build both versions (see "I need to switch between backends" above)
```

### TensorRT engine build is slow
This is normal on first run. The engine is cached as `model.trt` and reused on subsequent runs.

## Performance Tips

### For ONNX Runtime
- Use GPU provider: Install CUDA for GPU acceleration
- Reduce resolution: Lower input size = faster inference
- Adjust threads: Set `OMP_NUM_THREADS` for CPU

### For TensorRT
- FP16 mode: Automatically enabled if GPU supports it
- Keep `.trt` file: Reused across runs (skip rebuild)
- Batch inference: Process multiple images together (modify code)

## Examples

### Detection
```bash
./build/inference_app model.onnx dog.jpg coco-labels-91.txt
```

### Segmentation
```bash
./build/inference_app model.onnx dog.jpg coco-labels-91.txt --segmentation
```


## Getting Help

- 📖 Full documentation: [docs/backends.md](backends.md)
- 🔧 Refactoring details: [docs/REFACTORING.md](REFACTORING.md)
- 📚 Technical terms: [docs/glossary.md](glossary.md)
- 🐛 Issues: [GitHub Issues](https://github.com/olibartfast/rfdetr_inference/issues)
