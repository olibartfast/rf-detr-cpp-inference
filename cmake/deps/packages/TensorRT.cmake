# Versions come from versions.env via cmake/versions.cmake: TENSORRT_VERSION is
# the full four-component number, TENSORRT_SHORT_VERSION its major.minor.patch
# truncation (NVIDIA uses that for the URL directory and the Conan recipe), and
# CUDA_VERSION the `cuda-<v>` tag baked into the tarball name.
deps_declare(TensorRT
    REQUIRED              TRUE
    DEFINITIONS           USE_TENSORRT
    APT                   OFF
    CONAN                 "tensorrt/${TENSORRT_SHORT_VERSION}"
    VCPKG                 "tensorrt"
    PROVIDED_ACQUIRE      DOWNLOAD
    PROVIDED_URL          "https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/${TENSORRT_SHORT_VERSION}/tars/TensorRT-${TENSORRT_VERSION}.Linux.x86_64-gnu.cuda-${CUDA_VERSION}.tar.gz"
    PROVIDED_VERSION      "${TENSORRT_VERSION}"
    PROVIDED_SUBDIR       "TensorRT-${TENSORRT_VERSION}"
    PROVIDED_INCLUDE      "include"
    PROVIDED_LIBRARIES    "lib/libnvinfer.so;lib/libnvonnxparser.so"
    PROVIDED_LIBRARY      "lib/libnvinfer.so"
    PROVIDED_HEADER_GUARD "include/NvInfer.h"
    PROVIDED_ROOT_CACHE   "TENSORRT_ROOTDIR"
    PROVIDED_ROOT_VARS    "TENSORRT_ROOTDIR;TensorRT_ROOT"
    PROVIDED_CUDA         ON
)
