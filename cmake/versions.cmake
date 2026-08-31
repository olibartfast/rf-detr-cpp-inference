# cmake/versions.cmake — reads versions.env into CMake cache variables.
#
# Must be included before cmake/deps/Deps.cmake: that file globs
# cmake/deps/packages/*.cmake, and those declarations interpolate the versions
# set here.
#
# Every variable becomes a CACHE STRING, so -D on the command line wins:
#   cmake -S . -B build -DONNX_RUNTIME_VERSION=1.22.0
#
# To add a pin: put it in versions.env, then expose it below with
# rfdetr_declare_version(<NAME> "<docstring>"). Names match versions.env exactly
# so `grep -rn ONNX_RUNTIME_VERSION` finds the pin and all of its consumers.

include_guard(GLOBAL)

set(RFDETR_VERSIONS_ENV "${CMAKE_CURRENT_LIST_DIR}/../versions.env" CACHE FILEPATH
    "Path to the versions.env single source of truth")

if(NOT EXISTS "${RFDETR_VERSIONS_ENV}")
    message(FATAL_ERROR "versions.env not found at ${RFDETR_VERSIONS_ENV}")
endif()

# Re-run CMake when the pins change.
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${RFDETR_VERSIONS_ENV}")

# Parse KEY=VALUE lines into _RFDETR_ENV_<KEY> in the current scope. Comments,
# blank lines and anything without a shell-identifier key are skipped.
function(_rfdetr_read_versions_env path)
    file(STRINGS "${path}" _lines)
    foreach(_line IN LISTS _lines)
        string(STRIP "${_line}" _line)
        if(_line STREQUAL "" OR _line MATCHES "^#")
            continue()
        endif()
        if(NOT _line MATCHES "^([A-Za-z_][A-Za-z0-9_]*)=(.*)$")
            continue()
        endif()
        set(_key "${CMAKE_MATCH_1}")
        set(_value "${CMAKE_MATCH_2}")
        # Strip one layer of surrounding quotes, as a shell would.
        string(REGEX REPLACE "^\"(.*)\"$" "\\1" _value "${_value}")
        string(REGEX REPLACE "^'(.*)'$" "\\1" _value "${_value}")
        set("_RFDETR_ENV_${_key}" "${_value}" PARENT_SCOPE)
    endforeach()
endfunction()

# Publish one parsed key as a cache variable. Fails loudly rather than silently
# expanding to the empty string, which would produce a plausible-looking but
# wrong download URL.
macro(rfdetr_declare_version name doc)
    if(NOT DEFINED _RFDETR_ENV_${name})
        message(FATAL_ERROR "${name} is missing from ${RFDETR_VERSIONS_ENV}")
    endif()
    set(${name} "${_RFDETR_ENV_${name}}" CACHE STRING "${doc}")
endmacro()

_rfdetr_read_versions_env("${RFDETR_VERSIONS_ENV}")

rfdetr_declare_version(ONNX_RUNTIME_VERSION     "ONNX Runtime version")
rfdetr_declare_version(TENSORRT_VERSION         "TensorRT version (four components)")
rfdetr_declare_version(EXECUTORCH_VERSION       "ExecuTorch git tag")
rfdetr_declare_version(CUDA_VERSION             "CUDA Toolkit series TensorRT is built against")
rfdetr_declare_version(CUDA_ARCHITECTURES       "Default CMAKE_CUDA_ARCHITECTURES")
rfdetr_declare_version(DALI_VERSION             "NVIDIA DALI version staged for the GPU path")
rfdetr_declare_version(NGC_CONTAINER_TAG        "NGC monthly container tag")
rfdetr_declare_version(FFMPEG_VERSION           "FFmpeg version (Conan)")
rfdetr_declare_version(SDL_VERSION              "SDL2 version (Conan)")
rfdetr_declare_version(OPENCV_VERSION           "OpenCV version (Conan)")
rfdetr_declare_version(GTEST_VERSION            "GoogleTest version")
rfdetr_declare_version(GOOGLE_BENCHMARK_VERSION "Google Benchmark version")
rfdetr_declare_version(RFDETR_VERSION           "rfdetr Python package used for export")
rfdetr_declare_version(ONNX_OPSET_VERSION       "ONNX opset used by deploy/export_*.py")
rfdetr_declare_version(DOCKER_BASE_IMAGE        "Base image for the non-TensorRT Docker stages")

# --- Derived coordinates ------------------------------------------------------
# NVIDIA truncates the TensorRT version differently per artefact. Deriving these
# keeps versions.env down to the one number a bump actually changes.
#   10.13.3.9 -> 10.13.3   download URL directory, Conan recipe revision
string(REGEX MATCH "^[0-9]+\\.[0-9]+\\.[0-9]+" TENSORRT_SHORT_VERSION "${TENSORRT_VERSION}")
if(NOT TENSORRT_SHORT_VERSION)
    message(FATAL_ERROR
        "TENSORRT_VERSION='${TENSORRT_VERSION}' is not major.minor.patch.build; "
        "the download URL and Conan recipe cannot be derived from it.")
endif()

message(STATUS "Dependency pins from ${RFDETR_VERSIONS_ENV}:")
message(STATUS "  ONNX Runtime ${ONNX_RUNTIME_VERSION} | TensorRT ${TENSORRT_VERSION} | ExecuTorch ${EXECUTORCH_VERSION}")
message(STATUS "  CUDA ${CUDA_VERSION} | DALI ${DALI_VERSION} | NGC ${NGC_CONTAINER_TAG}")
