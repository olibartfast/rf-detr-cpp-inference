# Versions come from versions.env via cmake/versions.cmake: TENSORRT_VERSION is
# the full four-component number, TENSORRT_SHORT_VERSION its major.minor.patch
# truncation (NVIDIA uses that for the URL directory and the Conan recipe), and
# CUDA_VERSION the `cuda-<v>` tag baked into the tarball name.
#
# NVIDIA renamed the Linux tarball at 11.x, and switched it to zstd:
#   10.x  TensorRT-10.13.3.9.Linux.x86_64-gnu.cuda-13.0.tar.gz
#   11.x  TensorRT-Enterprise-11.2.1.2-Linux-x86_64-cuda-13.3-Release-external.tar.zst
# Both unpack to TensorRT-<version>/, and `cmake -E tar` detects the compression
# on extraction. An 11.x build usually needs CUDA_VERSION overridden as well, since
# each release ships for one CUDA minor (11.2.1.2 is cuda-13.3 only).
if(TENSORRT_VERSION VERSION_LESS 11)
    set(_trt_tarball "TensorRT-${TENSORRT_VERSION}.Linux.x86_64-gnu.cuda-${CUDA_VERSION}.tar.gz")
else()
    set(_trt_tarball "TensorRT-Enterprise-${TENSORRT_VERSION}-Linux-x86_64-cuda-${CUDA_VERSION}-Release-external.tar.zst")
endif()

deps_declare(TensorRT
    REQUIRED              TRUE
    DEFINITIONS           USE_TENSORRT
    APT                   OFF
    CONAN                 "tensorrt/${TENSORRT_SHORT_VERSION}"
    VCPKG                 "tensorrt"
    PROVIDED_ACQUIRE      DOWNLOAD
    PROVIDED_URL          "https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/${TENSORRT_SHORT_VERSION}/tars/${_trt_tarball}"
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
unset(_trt_tarball)
