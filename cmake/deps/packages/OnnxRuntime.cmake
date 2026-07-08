deps_declare(OnnxRuntime
    REQUIRED              TRUE
    DEFINITIONS           USE_ONNX_RUNTIME
    APT                   OFF
    CONAN                 "onnxruntime/1.21.0"
    VCPKG                 "onnxruntime"
    PROVIDED_ACQUIRE      DOWNLOAD
    PROVIDED_URL          "https://github.com/microsoft/onnxruntime/releases/download/v1.21.0/onnxruntime-linux-x64-1.21.0.tgz"
    PROVIDED_VERSION      "1.21.0"
    PROVIDED_SUBDIR       "onnxruntime-linux-x64-1.21.0"
    PROVIDED_INCLUDE      "include"
    PROVIDED_LIBRARY      "lib/libonnxruntime.so.1.21.0"
    PROVIDED_HEADER_GUARD "include/onnxruntime_cxx_api.h"
    PROVIDED_RUNTIME_LIBS "lib/libonnxruntime.so.1.21.0"
    PROVIDED_ROOT_CACHE   "ONNXRUNTIME_ROOTDIR"
    PROVIDED_ROOT_VARS    "ONNXRUNTIME_ROOTDIR;OnnxRuntime_ROOT"
)
