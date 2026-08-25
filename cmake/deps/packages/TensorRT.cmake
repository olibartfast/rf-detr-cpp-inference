deps_declare(TensorRT
    REQUIRED              TRUE
    DEFINITIONS           USE_TENSORRT
    APT                   OFF
    CONAN                 "tensorrt/10.13.3"
    VCPKG                 "tensorrt"
    PROVIDED_ACQUIRE      DOWNLOAD
    PROVIDED_URL          "https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.13.3/tars/TensorRT-10.13.3.9.Linux.x86_64-gnu.cuda-13.0.tar.gz"
    PROVIDED_VERSION      "10.13.3.9"
    PROVIDED_SUBDIR       "TensorRT-10.13.3.9"
    PROVIDED_INCLUDE      "include"
    PROVIDED_LIBRARIES    "lib/libnvinfer.so;lib/libnvonnxparser.so"
    PROVIDED_LIBRARY      "lib/libnvinfer.so"
    PROVIDED_HEADER_GUARD "include/NvInfer.h"
    PROVIDED_ROOT_CACHE   "TENSORRT_ROOTDIR"
    PROVIDED_ROOT_VARS    "TENSORRT_ROOTDIR;TensorRT_ROOT"
    PROVIDED_CUDA         ON
)
