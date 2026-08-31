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
#   executorch          ffmpeg            ExecuTorch   + FFmpeg/SDL2/stb
#   executorch          opencv            ExecuTorch   + OpenCV
#
# Build args:
#   INFERENCE_BACKEND   onnx (default) | tensorrt | executorch
#   MEDIA_BACKEND       ffmpeg (default) | opencv
#   EXECUTORCH_VERSION  git tag of the ExecuTorch C++ runtime
#   TENSORRT_VERSION    TensorRT bundled in the NGC base image
#   NGC_CONTAINER_TAG   NGC monthly tag for the TensorRT base image
#   DOCKER_BASE_IMAGE   base image for the non-TensorRT stages
#
# The four version args default to the values in versions.env, the single source
# of truth for this project's pins. A Dockerfile cannot read that file, so the
# defaults below restate it and scripts/check_version_sync.sh (run by lint.yml)
# fails when the two drift.
#
# Examples:
#   docker build -t rfdetr-onnx-ffmpeg .
#   docker build -t rfdetr-onnx-opencv  --build-arg MEDIA_BACKEND=opencv .
#   docker build -t rfdetr-trt-ffmpeg   --build-arg INFERENCE_BACKEND=tensorrt .
#   docker build -t rfdetr-trt-opencv   --build-arg INFERENCE_BACKEND=tensorrt --build-arg MEDIA_BACKEND=opencv .
#   docker build -t rfdetr-et-ffmpeg    --build-arg INFERENCE_BACKEND=executorch .
#
# Run (TensorRT images require --gpus all and a .engine/.trt model):
#   docker run --gpus all -v "$PWD/data:/data" -v "$PWD/exports:/exports" \
#     rfdetr-trt-opencv /exports/model.engine /data/dog.jpg /data/coco-labels-91.txt
#
# Run (ExecuTorch images are CPU-only and take a .pte model):
#   docker run -v "$PWD/data:/data" -v "$PWD/exports:/exports" \
#     rfdetr-et-ffmpeg /exports/model.pte /data/dog.jpg /data/coco-labels-91.txt

ARG INFERENCE_BACKEND=onnx
ARG MEDIA_BACKEND=ffmpeg
# Keep in sync with versions.env — see scripts/check_version_sync.sh.
ARG EXECUTORCH_VERSION=v1.4.0
ARG TENSORRT_VERSION=10.13.3.9
ARG NGC_CONTAINER_TAG=25.12
ARG DOCKER_BASE_IMAGE=ubuntu:24.04

# --- Base image selected by inference backend (all Ubuntu 24.04) ---
FROM ${DOCKER_BASE_IMAGE} AS base-onnx
FROM nvcr.io/nvidia/tensorrt:${NGC_CONTAINER_TAG}-py3 AS base-tensorrt
FROM ${DOCKER_BASE_IMAGE} AS base-executorch

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

# NOTE: this runs BEFORE `COPY . .` on purpose. It does not use the repo source, and the
# build takes ~10 minutes, so keeping it above the COPY lets Docker reuse the cached layer
# across source edits instead of rebuilding ExecuTorch every time a .cpp changes.
# For the ExecuTorch backend, build the C++ runtime from source and install it to
# /opt/executorch; the deps facade then resolves it with find_package(executorch
# CONFIG) via EXECUTORCH_ROOTDIR. There is no distro or registry package for it.
#
# Two things this step must get right:
#  1. ExecuTorch's configure runs operator codegen through PYTHON_EXECUTABLE, and
#     tools/cmake/Codegen.cmake does `import torchgen` — so the interpreter needs
#     torch installed. A bare python3 fails. Only torchgen is used, so the CPU
#     wheel is enough, and the whole venv is discarded in the runtime stage.
#  2. Up to v1.3.1, upstream installed extension_evalue_util with
#     DESTINATION ${CMAKE_BINARY_DIR}/lib instead of ${CMAKE_INSTALL_LIBDIR}, so
#     the library never reached the prefix and its exported target kept an
#     absolute build-tree path. find_package then hard-failed once the build tree
#     was gone — even though this project never links that target. Fixed upstream
#     in the default v1.4.0; the sed stays for older EXECUTORCH_VERSION overrides.
#     (The sed body is single-quoted so the shell does not expand ${...}.)
#  4. EXECUTORCH_BUILD_KERNELS_OPTIMIZED is required, not a tuning knob: .pte files
#     from rfdetr >= 1.9.1 call aten::linear.out, which only the optimized kernel
#     set registers. Without it the program fails at load.
#  3. build-essential is required even though the project itself builds with
#     clang+ninja: ExecuTorch pulls in third-party/flatcc as an ExternalProject
#     that configures with the "Unix Makefiles" generator and does not inherit
#     CMAKE_C_COMPILER, so it needs make and cc on PATH or it fails at configure.
#
# Built with the same clang-18 as the project to keep one C++ runtime/ABI.
ARG EXECUTORCH_VERSION
RUN if [ "$INFERENCE_BACKEND" = "executorch" ]; then \
        apt-get update && apt-get install -y --no-install-recommends \
            python3 python3-venv build-essential \
        && rm -rf /var/lib/apt/lists/* && \
        python3 -m venv /tmp/et-venv && \
        /tmp/et-venv/bin/pip install --no-cache-dir --quiet \
            --retries 10 --timeout 120 \
            --index-url https://download.pytorch.org/whl/cpu torch && \
        /tmp/et-venv/bin/pip install --no-cache-dir --quiet \
            --retries 10 --timeout 120 pyyaml setuptools && \
        git clone --depth 1 --branch "${EXECUTORCH_VERSION}" \
            https://github.com/pytorch/executorch.git /tmp/executorch && \
        git -C /tmp/executorch submodule update --init --recursive --depth 1 && \
        sed -i 's|DESTINATION ${CMAKE_BINARY_DIR}/lib|DESTINATION ${CMAKE_INSTALL_LIBDIR}|' \
            /tmp/executorch/extension/evalue_util/CMakeLists.txt && \
        cmake -S /tmp/executorch -B /tmp/executorch/cmake-out -G Ninja \
            -DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_C_COMPILER=/usr/bin/clang-18 \
            -DCMAKE_CXX_COMPILER=/usr/bin/clang++-18 \
            -DCMAKE_INSTALL_PREFIX=/opt/executorch \
            -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
            -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
            -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
            -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
            -DEXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP=ON \
            -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON \
            -DEXECUTORCH_BUILD_XNNPACK=ON \
            -DEXECUTORCH_ENABLE_LOGGING=ON \
            -DEXECUTORCH_BUILD_TESTS=OFF \
            -DEXECUTORCH_BUILD_EXAMPLES=OFF \
            -DPYTHON_EXECUTABLE=/tmp/et-venv/bin/python && \
        cmake --build /tmp/executorch/cmake-out --parallel && \
        cmake --install /tmp/executorch/cmake-out && \
        rm -rf /tmp/executorch /tmp/et-venv ; \
    fi

WORKDIR /workspace
COPY . .

# For the TensorRT backend, reuse the TensorRT bundled in the NGC base image
# instead of downloading the ~1GB tarball from developer.nvidia.com (which is
# rate-limited and frequently times out). The CMake build looks for it at
# build/_deps/TensorRT-<version>/ (the layout cmake/deps/packages/TensorRT.cmake
# builds from TENSORRT_VERSION in versions.env); symlink the system headers/libs
# into that layout so the download guard passes.
ARG TENSORRT_VERSION
RUN if [ "$INFERENCE_BACKEND" = "tensorrt" ]; then \
        mkdir -p "build/_deps/TensorRT-${TENSORRT_VERSION}" && \
        ln -s /usr/include/x86_64-linux-gnu "build/_deps/TensorRT-${TENSORRT_VERSION}/include" && \
        ln -s /usr/lib/x86_64-linux-gnu "build/_deps/TensorRT-${TENSORRT_VERSION}/lib" ; \
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
        -DUSE_EXECUTORCH=$([ "$INFERENCE_BACKEND" = executorch ] && echo ON || echo OFF) \
        -DEXECUTORCH_ROOTDIR=/opt/executorch \
        -DUSE_OPENCV=$([ "$MEDIA_BACKEND" = opencv ] && echo ON || echo OFF) \
    && cmake --build build --parallel

# Stage the ONNX Runtime shared libs (fetched at build time) so the runtime
# stage can copy them. Empty for the TensorRT backend (uses NGC-bundled TRT) and
# for ExecuTorch, which is static-linked (.a) into inference_app — nothing to ship.
RUN mkdir -p /staging/ort && \
    if [ "$INFERENCE_BACKEND" = "onnx" ]; then \
        cp /workspace/build/_deps/onnxruntime-linux-*/lib/libonnxruntime.so* /staging/ort/ ; \
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
