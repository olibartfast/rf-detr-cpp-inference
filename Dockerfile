# Unified parametric Dockerfile for RF-DETR C++ Inference.
#
# Builds the full inference-backend x media-backend matrix from a single file:
#
#   INFERENCE_BACKEND   MEDIA_BACKEND     Image
#   ─────────────────   ─────────────     ─────────────────────────────
#   onnx      (default) ffmpeg  (default) ONNX Runtime + FFmpeg/SDL2/stb
#   onnx                opencv            ONNX Runtime + OpenCV
#   tensorrt            ffmpeg            TensorRT     + FFmpeg/SDL2/stb
#   tensorrt            opencv            TensorRT     + OpenCV
#
# Build args:
#   INFERENCE_BACKEND   onnx (default) | tensorrt
#   MEDIA_BACKEND       ffmpeg (default) | opencv
#
# Examples:
#   docker build -t rfdetr-onnx-ffmpeg .
#   docker build -t rfdetr-onnx-opencv  --build-arg MEDIA_BACKEND=opencv .
#   docker build -t rfdetr-trt-ffmpeg   --build-arg INFERENCE_BACKEND=tensorrt .
#   docker build -t rfdetr-trt-opencv   --build-arg INFERENCE_BACKEND=tensorrt --build-arg MEDIA_BACKEND=opencv .
#
# Run (TensorRT images require --gpus all and a .engine/.trt model):
#   docker run --gpus all -v "$PWD/data:/data" -v "$PWD/exports:/exports" \
#     rfdetr-trt-opencv /exports/model.engine /data/dog.jpg /data/coco-labels-91.txt

ARG INFERENCE_BACKEND=onnx
ARG MEDIA_BACKEND=ffmpeg

# --- Base image selected by inference backend (both Ubuntu 24.04) ---
FROM ubuntu:24.04 AS base-onnx
FROM nvcr.io/nvidia/tensorrt:25.12-py3 AS base-tensorrt

# =============================================================================
# Build stage
# =============================================================================
FROM base-${INFERENCE_BACKEND} AS builder

ARG INFERENCE_BACKEND
ARG MEDIA_BACKEND
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        cmake \
        ninja-build \
        clang-18 \
        pkg-config \
        wget \
        ca-certificates \
        git \
    && rm -rf /var/lib/apt/lists/*

# Media/display backend dev packages: OpenCV pulls imgcodecs/imgproc/videoio/
# highgui itself; the FFmpeg backend needs libav* + SDL2.
RUN if [ "$MEDIA_BACKEND" = "opencv" ]; then \
        apt-get update && apt-get install -y --no-install-recommends libopencv-dev \
        && rm -rf /var/lib/apt/lists/*; \
    else \
        apt-get update && apt-get install -y --no-install-recommends \
            libavcodec-dev libavformat-dev libavutil-dev libswscale-dev libsdl2-dev \
        && rm -rf /var/lib/apt/lists/*; \
    fi

WORKDIR /workspace
COPY . .

# For the TensorRT backend, reuse the TensorRT bundled in the NGC base image
# instead of downloading the ~1GB tarball from developer.nvidia.com (which is
# rate-limited and frequently times out). The CMake build looks for it at
# build/_deps/TensorRT-<version>/ (see CMakeLists.txt TRT_VERSION pin);
# symlink the system headers/libs into that layout so the download guard passes.
# Keep this version in sync with the TRT_VERSION pin in CMakeLists.txt.
RUN if [ "$INFERENCE_BACKEND" = "tensorrt" ]; then \
        mkdir -p build/_deps/TensorRT-10.13.3.9 && \
        ln -s /usr/include/x86_64-linux-gnu build/_deps/TensorRT-10.13.3.9/include && \
        ln -s /usr/lib/x86_64-linux-gnu build/_deps/TensorRT-10.13.3.9/lib ; \
    fi

RUN CUDA_INC=""; \
    if [ "$INFERENCE_BACKEND" = "tensorrt" ]; then CUDA_INC="-I/usr/local/cuda/include"; fi; \
    cmake -S . -B build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_COMPILER=/usr/bin/clang-18 \
        -DCMAKE_CXX_COMPILER=/usr/bin/clang++-18 \
        -DCMAKE_C_FLAGS="$CUDA_INC" \
        -DCMAKE_CXX_FLAGS="$CUDA_INC" \
        -DUSE_ONNX_RUNTIME=$([ "$INFERENCE_BACKEND" = onnx ] && echo ON || echo OFF) \
        -DUSE_TENSORRT=$([ "$INFERENCE_BACKEND" = tensorrt ] && echo ON || echo OFF) \
        -DUSE_OPENCV=$([ "$MEDIA_BACKEND" = opencv ] && echo ON || echo OFF) \
    && cmake --build build --parallel

# Stage the ONNX Runtime shared libs (fetched at build time) so the runtime
# stage can copy them. Empty for the TensorRT backend (uses NGC-bundled TRT).
RUN mkdir -p /staging/ort && \
    if [ "$INFERENCE_BACKEND" = "onnx" ]; then \
        cp /workspace/build/_deps/onnxruntime-linux-x64-*/lib/libonnxruntime.so* /staging/ort/ ; \
    fi

# =============================================================================
# Runtime stage
# =============================================================================
FROM base-${INFERENCE_BACKEND} AS runtime

ARG INFERENCE_BACKEND
ARG MEDIA_BACKEND
ENV DEBIAN_FRONTEND=noninteractive

# Runtime media/display libraries for the selected backend. Package names are
# identical across both base images (both Ubuntu 24.04).
RUN if [ "$MEDIA_BACKEND" = "opencv" ]; then \
        apt-get update && apt-get install -y --no-install-recommends libopencv-dev \
        && rm -rf /var/lib/apt/lists/*; \
    else \
        apt-get update && apt-get install -y --no-install-recommends \
            libavcodec60 libavformat60 libavutil58 libswscale7 libx264-164 libsdl2-2.0-0 \
        && rm -rf /var/lib/apt/lists/*; \
    fi

COPY --from=builder /workspace/build/inference_app /usr/local/bin/inference_app
COPY --from=builder /staging/ort/ /usr/local/lib/

RUN ldconfig

ENTRYPOINT ["inference_app"]
CMD ["--help"]
