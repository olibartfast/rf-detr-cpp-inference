# Version comes from versions.env via cmake/versions.cmake.
set(_onnxruntime_version "${ONNX_RUNTIME_VERSION}")
string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" _onnxruntime_processor)

if(_onnxruntime_processor MATCHES "^(x86_64|amd64)$")
    set(_onnxruntime_arch "x64")
elseif(_onnxruntime_processor MATCHES "^(aarch64|arm64)$")
    set(_onnxruntime_arch "arm64")
else()
    set(_onnxruntime_arch "")
endif()

set(_onnxruntime_platform "")
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    if(_onnxruntime_arch STREQUAL "arm64")
        set(_onnxruntime_asset_arch "aarch64")
    else()
        set(_onnxruntime_asset_arch "${_onnxruntime_arch}")
    endif()
    set(_onnxruntime_platform "linux")
    set(_onnxruntime_extension "tgz")
    set(_onnxruntime_library "lib/libonnxruntime.so.${_onnxruntime_version}")
    set(_onnxruntime_runtime "lib/libonnxruntime.so.${_onnxruntime_version}")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    set(_onnxruntime_asset_arch "${_onnxruntime_arch}")
    set(_onnxruntime_platform "win")
    set(_onnxruntime_extension "zip")
    set(_onnxruntime_library "lib/onnxruntime.lib")
    set(_onnxruntime_runtime "lib/onnxruntime.dll")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    set(_onnxruntime_library "lib/libonnxruntime.dylib")
    set(_onnxruntime_runtime "lib/libonnxruntime.dylib")
else()
    set(_onnxruntime_library "lib/libonnxruntime${CMAKE_SHARED_LIBRARY_SUFFIX}")
    set(_onnxruntime_runtime "${_onnxruntime_library}")
endif()

# Catalog loading must not require support for a bundled download. ROOT and
# package-manager providers remain available; the resolver reports failure
# only when the dependency is requested and no provider can resolve it.
set(_onnxruntime_acquire ROOT)
set(_onnxruntime_archive "")
set(_onnxruntime_url "")
if(_onnxruntime_platform AND _onnxruntime_arch)
    set(_onnxruntime_acquire DOWNLOAD)
    set(_onnxruntime_archive "onnxruntime-${_onnxruntime_platform}-${_onnxruntime_asset_arch}-${_onnxruntime_version}")
    set(_onnxruntime_url "https://github.com/microsoft/onnxruntime/releases/download/v${_onnxruntime_version}/${_onnxruntime_archive}.${_onnxruntime_extension}")
endif()

deps_declare(OnnxRuntime
    REQUIRED              TRUE
    DEFINITIONS           USE_ONNX_RUNTIME
    APT                   OFF
    CONAN                 "onnxruntime/${_onnxruntime_version}"
    VCPKG                 "onnxruntime"
    PROVIDED_ACQUIRE      "${_onnxruntime_acquire}"
    PROVIDED_URL          "${_onnxruntime_url}"
    PROVIDED_VERSION      "${_onnxruntime_version}"
    PROVIDED_SUBDIR       "${_onnxruntime_archive}"
    PROVIDED_INCLUDE      "include"
    PROVIDED_LIBRARY      "${_onnxruntime_library}"
    PROVIDED_HEADER_GUARD "include/onnxruntime_cxx_api.h"
    PROVIDED_RUNTIME_LIBS "${_onnxruntime_runtime}"
    PROVIDED_ROOT_CACHE   "ONNXRUNTIME_ROOTDIR"
    PROVIDED_ROOT_VARS    "ONNXRUNTIME_ROOTDIR;OnnxRuntime_ROOT"
)
