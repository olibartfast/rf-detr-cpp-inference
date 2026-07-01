# ONNX Runtime Dockerfile for RF-DETR C++ Inference
# Multi-stage build to keep final image small

# --- Build stage ---
FROM ubuntu:24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake \
    ninja-build \
    clang-18 \
    pkg-config \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libswscale-dev \
    libsdl2-dev \
    wget \
    ca-certificates \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY . .

RUN rm -rf build && cmake -S . -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=/usr/bin/clang-18 \
    -DCMAKE_CXX_COMPILER=/usr/bin/clang++-18 \
    -DUSE_ONNX_RUNTIME=ON \
    -DUSE_TENSORRT=OFF

RUN cmake --build build --parallel

# --- Runtime stage ---
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    libavcodec60 \
    libavformat60 \
    libavutil58 \
    libswscale7 \
    libx264-164 \
    libsdl2-2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /workspace/build/inference_app /usr/local/bin/inference_app
COPY --from=builder /workspace/build/_deps/onnxruntime-linux-x64-*/lib/libonnxruntime.so* /usr/local/lib/

RUN ldconfig

ENTRYPOINT ["inference_app"]
CMD ["--help"]

# Usage:
# Build: docker build -t rfdetr-onnx .
# Run:   docker run -v $(pwd)/data:/data rfdetr-onnx /data/model.onnx /data/image.jpg /data/labels.txt
