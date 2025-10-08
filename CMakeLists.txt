cmake_minimum_required(VERSION 3.12)
project(rfdetr_inference CXX)

# Set C++ standard to C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Backend Selection Options ---
option(USE_ONNX_RUNTIME "Build with ONNX Runtime backend support" ON)
option(USE_TENSORRT "Build with TensorRT backend support" OFF)

# Validate that at least one backend is enabled
if(NOT USE_ONNX_RUNTIME AND NOT USE_TENSORRT)
    message(FATAL_ERROR "At least one backend must be enabled. Set USE_ONNX_RUNTIME=ON or USE_TENSORRT=ON")
endif()

message(STATUS "Backend configuration:")
message(STATUS "  ONNX Runtime: ${USE_ONNX_RUNTIME}")
message(STATUS "  TensorRT:     ${USE_TENSORRT}")

# --- ONNX Runtime Configuration ---
if(USE_ONNX_RUNTIME)
    message(STATUS "Configuring ONNX Runtime backend...")
    
    set(ORT_VERSION "1.21.0")
    set(ORT_ARCHIVE_BASENAME "onnxruntime-linux-x64-${ORT_VERSION}")
    set(ORT_ARCHIVE_NAME "${ORT_ARCHIVE_BASENAME}.tgz")
    set(ORT_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/${ORT_ARCHIVE_NAME}")

    set(ORT_DEPS_DIR "${CMAKE_BINARY_DIR}/_deps")
    set(ORT_DOWNLOAD_PATH "${ORT_DEPS_DIR}/${ORT_ARCHIVE_NAME}")
    set(ORT_EXTRACT_DIR "${ORT_DEPS_DIR}")
    set(ORT_INSTALL_DIR "${ORT_EXTRACT_DIR}/${ORT_ARCHIVE_BASENAME}")

    if(NOT EXISTS "${ORT_INSTALL_DIR}/include/onnxruntime_cxx_api.h")
        message(STATUS "ONNX Runtime not found in build directory. Downloading and extracting...")
        file(MAKE_DIRECTORY "${ORT_DEPS_DIR}")
        if(NOT EXISTS "${ORT_DOWNLOAD_PATH}")
            message(STATUS "Downloading ONNX Runtime from ${ORT_URL}...")
            file(DOWNLOAD ${ORT_URL} ${ORT_DOWNLOAD_PATH}
                 SHOW_PROGRESS
                 TIMEOUT 600
                 STATUS download_status)
            list(GET download_status 0 download_code)
            list(GET download_status 1 download_message)
            if(NOT download_code EQUAL 0)
                message(FATAL_ERROR "Download failed: ${download_message}")
            endif()
            message(STATUS "Download completed: ${ORT_DOWNLOAD_PATH}")
        else()
            message(STATUS "Using existing ONNX Runtime archive: ${ORT_DOWNLOAD_PATH}")
        endif()
        message(STATUS "Extracting ${ORT_DOWNLOAD_PATH} to ${ORT_EXTRACT_DIR}...")
        execute_process(
            COMMAND ${CMAKE_COMMAND} -E tar xzf "${ORT_DOWNLOAD_PATH}"
            WORKING_DIRECTORY "${ORT_EXTRACT_DIR}"
            RESULT_VARIABLE extract_result
        )
        if(NOT extract_result EQUAL 0)
            message(FATAL_ERROR "Extraction failed with result: ${extract_result}")
        endif()
        message(STATUS "Extraction completed.")
        if(NOT EXISTS "${ORT_INSTALL_DIR}/include/onnxruntime_cxx_api.h")
            message(FATAL_ERROR "Extraction failed: expected directory not found at ${ORT_INSTALL_DIR}")
        endif()
    else()
        message(STATUS "Found existing ONNX Runtime at: ${ORT_INSTALL_DIR}")
    endif()

    set(ONNXRUNTIME_ROOTDIR "${ORT_INSTALL_DIR}" CACHE PATH "Path to ONNX Runtime root directory")
    set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_ROOTDIR}/include")
    set(ONNXRUNTIME_LIBRARY "${ONNXRUNTIME_ROOTDIR}/lib/libonnxruntime.so.${ORT_VERSION}")

    if(NOT EXISTS "${ONNXRUNTIME_INCLUDE_DIR}/onnxruntime_cxx_api.h")
        message(FATAL_ERROR "ONNX Runtime include directory not found at: ${ONNXRUNTIME_INCLUDE_DIR}")
    endif()
    if(NOT EXISTS "${ONNXRUNTIME_LIBRARY}")
        message(FATAL_ERROR "ONNX Runtime library not found at: ${ONNXRUNTIME_LIBRARY}")
    endif()

    message(STATUS "ONNX Runtime include directory: ${ONNXRUNTIME_INCLUDE_DIR}")
    message(STATUS "ONNX Runtime library: ${ONNXRUNTIME_LIBRARY}")
endif()

# --- TensorRT Configuration ---
if(USE_TENSORRT)
    message(STATUS "Configuring TensorRT backend...")
    
    # Try to find TensorRT installation
    find_path(TENSORRT_INCLUDE_DIR NvInfer.h
        HINTS 
            /usr/include/x86_64-linux-gnu
            /usr/local/cuda/include
            /usr/include
            $ENV{TENSORRT_ROOT}/include
        PATH_SUFFIXES include
    )
    
    find_library(TENSORRT_LIBRARY nvinfer
        HINTS
            /usr/lib/x86_64-linux-gnu
            /usr/local/cuda/lib64
            /usr/lib
            $ENV{TENSORRT_ROOT}/lib
        PATH_SUFFIXES lib lib64
    )
    
    find_library(TENSORRT_ONNX_PARSER nvonnxparser
        HINTS
            /usr/lib/x86_64-linux-gnu
            /usr/local/cuda/lib64
            /usr/lib
            $ENV{TENSORRT_ROOT}/lib
        PATH_SUFFIXES lib lib64
    )
    
    # Find CUDA runtime
    find_package(CUDA QUIET)
    if(NOT CUDA_FOUND)
        find_library(CUDA_RUNTIME_LIBRARY cudart
            HINTS
                /usr/local/cuda/lib64
                /usr/lib/x86_64-linux-gnu
            PATH_SUFFIXES lib lib64
        )
    else()
        set(CUDA_RUNTIME_LIBRARY ${CUDA_LIBRARIES})
    endif()
    
    if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_LIBRARY OR NOT TENSORRT_ONNX_PARSER OR NOT CUDA_RUNTIME_LIBRARY)
        message(FATAL_ERROR 
            "TensorRT not found. Please install TensorRT or set TENSORRT_ROOT environment variable.\n"
            "  TENSORRT_INCLUDE_DIR: ${TENSORRT_INCLUDE_DIR}\n"
            "  TENSORRT_LIBRARY: ${TENSORRT_LIBRARY}\n"
            "  TENSORRT_ONNX_PARSER: ${TENSORRT_ONNX_PARSER}\n"
            "  CUDA_RUNTIME_LIBRARY: ${CUDA_RUNTIME_LIBRARY}\n"
            "You can disable TensorRT support with -DUSE_TENSORRT=OFF"
        )
    endif()
    
    message(STATUS "TensorRT include directory: ${TENSORRT_INCLUDE_DIR}")
    message(STATUS "TensorRT library: ${TENSORRT_LIBRARY}")
    message(STATUS "TensorRT ONNX parser: ${TENSORRT_ONNX_PARSER}")
    message(STATUS "CUDA runtime library: ${CUDA_RUNTIME_LIBRARY}")
endif()

# --- Find OpenCV ---
find_package(OpenCV REQUIRED)
if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV not found. Please install it (e.g., via 'sudo apt-get install libopencv-dev')")
endif()
message(STATUS "OpenCV include directories: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")

# --- Fetch Google Test ---
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG release-1.12.1
)
FetchContent_MakeAvailable(googletest)

# --- Source and Include Directories ---
set(SOURCE_DIR "${CMAKE_SOURCE_DIR}/src")
set(TEST_DIR "${CMAKE_SOURCE_DIR}/tests")
set(DATA_DIR "${CMAKE_SOURCE_DIR}/data")

# Create the data directory if it doesn't exist
file(MAKE_DIRECTORY "${DATA_DIR}")

# --- Main Library ---
set(RFDETR_SOURCES
    "${SOURCE_DIR}/rfdetr_inference.cpp"
    "${SOURCE_DIR}/backends/inference_backend.cpp"
)

# Add backend-specific sources
if(USE_ONNX_RUNTIME)
    list(APPEND RFDETR_SOURCES "${SOURCE_DIR}/backends/onnx_runtime_backend.cpp")
endif()

if(USE_TENSORRT)
    list(APPEND RFDETR_SOURCES "${SOURCE_DIR}/backends/tensorrt_backend.cpp")
endif()

add_library(rfdetr_inference_lib STATIC ${RFDETR_SOURCES})

target_include_directories(rfdetr_inference_lib
    PUBLIC
        "${SOURCE_DIR}"
        "${OpenCV_INCLUDE_DIRS}"
)

# Add backend-specific includes and compile definitions
if(USE_ONNX_RUNTIME)
    target_include_directories(rfdetr_inference_lib PUBLIC "${ONNXRUNTIME_INCLUDE_DIR}")
    target_compile_definitions(rfdetr_inference_lib PUBLIC USE_ONNX_RUNTIME)
endif()

if(USE_TENSORRT)
    target_include_directories(rfdetr_inference_lib PUBLIC "${TENSORRT_INCLUDE_DIR}")
    target_compile_definitions(rfdetr_inference_lib PUBLIC USE_TENSORRT)
endif()

# Link libraries
target_link_libraries(rfdetr_inference_lib
    PUBLIC
        "${OpenCV_LIBS}"
)

if(USE_ONNX_RUNTIME)
    target_link_libraries(rfdetr_inference_lib PUBLIC "${ONNXRUNTIME_LIBRARY}")
endif()

if(USE_TENSORRT)
    target_link_libraries(rfdetr_inference_lib PUBLIC 
        "${TENSORRT_LIBRARY}"
        "${TENSORRT_ONNX_PARSER}"
        "${CUDA_RUNTIME_LIBRARY}"
    )
endif()

# --- Main Application ---
add_executable(inference_app "${SOURCE_DIR}/main.cpp")
target_link_libraries(inference_app PRIVATE rfdetr_inference_lib)

# Copy backend shared libraries to the executable directory
if(USE_ONNX_RUNTIME)
    add_custom_command(TARGET inference_app POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${ONNXRUNTIME_LIBRARY}"
                $<TARGET_FILE_DIR:inference_app>
        COMMENT "Copying ONNX Runtime library to output directory"
    )
endif()

# --- Unit Tests ---
add_executable(unit_tests "${TEST_DIR}/unit/test_rfdetr_inference.cpp")
target_link_libraries(unit_tests PRIVATE rfdetr_inference_lib gtest gtest_main gmock)
target_include_directories(unit_tests PRIVATE "${SOURCE_DIR}" "${ONNXRUNTIME_INCLUDE_DIR}" "${OpenCV_INCLUDE_DIRS}")

# --- Integration Tests ---
add_executable(integration_tests "${TEST_DIR}/integration/integration_test_rfdetr_inference.cpp")
target_link_libraries(integration_tests PRIVATE rfdetr_inference_lib gtest gtest_main)
target_include_directories(integration_tests PRIVATE "${SOURCE_DIR}" "${ONNXRUNTIME_INCLUDE_DIR}" "${OpenCV_INCLUDE_DIRS}")

# Ensure the data directory exists before running integration tests
add_custom_command(TARGET integration_tests PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "${DATA_DIR}"
    COMMENT "Creating data directory for integration tests"
)

# Enable testing
enable_testing()
add_test(NAME UnitTests COMMAND unit_tests)
add_test(NAME IntegrationTests COMMAND integration_tests
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"  # Ensure tests run from the project root
)

# --- Custom Target to Run All Tests ---
add_custom_target(run_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
    DEPENDS unit_tests integration_tests
    COMMENT "Running all tests..."
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
)