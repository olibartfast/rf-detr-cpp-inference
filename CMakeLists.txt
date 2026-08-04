cmake_minimum_required(VERSION 3.12)
project(rfdetr_inference CXX C)

# Set C++ standard to C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Compiler Warnings (applied to project targets only, not third-party) ---
set(PROJECT_WARNING_FLAGS
    -Wall
    -Wextra
    -Wpedantic
    -Wshadow
    -Wnon-virtual-dtor
    -Wold-style-cast
    -Wcast-align
    -Wunused
    -Woverloaded-virtual
    -Wconversion
    -Wsign-conversion
    -Wnull-dereference
    -Wdouble-promotion
    -Wformat=2
    -Wimplicit-fallthrough
)

# Treat warnings as errors in CI (opt-in via -DWERROR=ON)
option(WERROR "Treat compiler warnings as errors" OFF)
if(WERROR)
    list(APPEND PROJECT_WARNING_FLAGS -Werror)
endif()

# --- Sanitizers (debug builds, opt-in) ---
# ASan+UBSan, strict UBSan, and TSan are separate modes. ASan and TSan are
# mutually exclusive: clang/gcc forbid combining them (e.g.
# -fsanitize=address,thread is a hard error). Keep strict UBSan separate so the
# noisier checks stay opt-in.
option(SANITIZERS "Enable AddressSanitizer + UndefinedBehaviorSanitizer" OFF)
option(STRICT_UBSAN "Enable stricter UndefinedBehaviorSanitizer checks" OFF)
option(THREAD_SANITIZER "Enable ThreadSanitizer (data-race detection)" OFF)

if((SANITIZERS AND STRICT_UBSAN) OR (SANITIZERS AND THREAD_SANITIZER) OR (STRICT_UBSAN AND THREAD_SANITIZER))
    message(FATAL_ERROR "SANITIZERS, STRICT_UBSAN, and THREAD_SANITIZER are mutually exclusive. Enable only one.")
endif()

if(SANITIZERS)
    add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address,undefined)
    message(STATUS "Sanitizers: AddressSanitizer + UndefinedBehaviorSanitizer enabled")
elseif(STRICT_UBSAN)
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        set(STRICT_UBSAN_CXX_FLAGS -fsanitize=undefined,local-bounds,vptr,implicit-conversion)
        set(STRICT_UBSAN_C_FLAGS -fsanitize=undefined,local-bounds,implicit-conversion)
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        set(STRICT_UBSAN_CXX_FLAGS -fsanitize=undefined,bounds-strict,vptr)
        set(STRICT_UBSAN_C_FLAGS -fsanitize=undefined,bounds-strict)
    else()
        message(FATAL_ERROR "STRICT_UBSAN is supported only with Clang or GCC")
    endif()
    foreach(flag IN LISTS STRICT_UBSAN_CXX_FLAGS)
        add_compile_options($<$<COMPILE_LANGUAGE:CXX>:${flag}>)
    endforeach()
    foreach(flag IN LISTS STRICT_UBSAN_C_FLAGS)
        add_compile_options($<$<COMPILE_LANGUAGE:C>:${flag}>)
    endforeach()
    add_compile_options(-fno-omit-frame-pointer)
    add_link_options(${STRICT_UBSAN_CXX_FLAGS})
    message(STATUS "Sanitizer: strict UndefinedBehaviorSanitizer checks enabled (CXX: ${STRICT_UBSAN_CXX_FLAGS}; C: ${STRICT_UBSAN_C_FLAGS})")
elseif(THREAD_SANITIZER)
    add_compile_options(-fsanitize=thread -fno-omit-frame-pointer)
    add_link_options(-fsanitize=thread)
    message(STATUS "Sanitizer: ThreadSanitizer enabled")
endif()

# --- Backend Selection Options ---
option(USE_ONNX_RUNTIME "Build with ONNX Runtime backend support" ON)
option(USE_TENSORRT "Build with TensorRT backend support" OFF)
option(USE_EXECUTORCH "Build with ExecuTorch backend support (.pte models)" OFF)
option(USE_OPENCV "Use OpenCV for image/video/display I/O instead of FFmpeg+SDL2+stb" OFF)

# ExecuTorch delegates are chosen at .pte export time and consumed by the runtime; the C++ side's
# job is to link the matching backend library so the delegate self-registers. 'xnnpack' is the
# default and matches rfdetr's export(format="executorch", backend="xnnpack"). 'portable' links
# only the CPU portable kernels. SDK-backed delegates (coreml, qnn) each pull an external SDK and
# are intentionally not wired here.
set(EXECUTORCH_DELEGATE "xnnpack" CACHE STRING "ExecuTorch delegate to link: xnnpack or portable")
set_property(CACHE EXECUTORCH_DELEGATE PROPERTY STRINGS xnnpack portable)

# Backend selection is compile-time and exclusive: exactly one backend is compiled in, so the
# count must be exactly 1. Without this check a second enabled backend would be silently dropped
# by the #ifdef/#elif chain in src/backends/inference_backend.cpp, and the binary would use a
# backend the user did not ask for.
set(_ENABLED_BACKENDS "")
foreach(_be USE_ONNX_RUNTIME USE_TENSORRT USE_EXECUTORCH)
    if(${_be})
        list(APPEND _ENABLED_BACKENDS ${_be})
    endif()
endforeach()
list(LENGTH _ENABLED_BACKENDS _NUM_BACKENDS)
if(_NUM_BACKENDS EQUAL 0)
    message(FATAL_ERROR
        "At least one backend must be enabled. Set USE_ONNX_RUNTIME=ON, USE_TENSORRT=ON, or USE_EXECUTORCH=ON")
elseif(_NUM_BACKENDS GREATER 1)
    string(REPLACE ";" ", " _BACKEND_LIST "${_ENABLED_BACKENDS}")
    message(FATAL_ERROR
        "Only one inference backend can be enabled at a time, but ${_NUM_BACKENDS} are on: ${_BACKEND_LIST}. "
        "Turn the others OFF, e.g. -DUSE_ONNX_RUNTIME=OFF -DUSE_EXECUTORCH=ON")
endif()

if(USE_EXECUTORCH AND NOT EXECUTORCH_DELEGATE MATCHES "^(xnnpack|portable)$")
    message(FATAL_ERROR "Unsupported EXECUTORCH_DELEGATE '${EXECUTORCH_DELEGATE}'. Valid values: xnnpack, portable")
endif()

message(STATUS "Backend configuration:")
message(STATUS "  ONNX Runtime: ${USE_ONNX_RUNTIME}")
message(STATUS "  TensorRT:     ${USE_TENSORRT}")
message(STATUS "  ExecuTorch:   ${USE_EXECUTORCH}")
if(USE_EXECUTORCH)
    message(STATUS "  ET delegate:  ${EXECUTORCH_DELEGATE}")
endif()

# --- Dependency resolution (unified package-manager facade) ------------------
# Encapsulates three resolution policies (apt+provided, conan+provided,
# vcpkg+provided) behind find_dependency_unified(). Each dependency is declared
# once in cmake/deps/packages/<Name>.cmake; the resolver picks the acquisition
# strategy per -DDEPS_MODE=apt|conan|vcpkg|auto. See docs/package-manager-architecture.md.
include("${CMAKE_SOURCE_DIR}/cmake/deps/Deps.cmake")

include(FetchContent)

if(USE_ONNX_RUNTIME)
    find_dependency_unified(OnnxRuntime REQUIRED)
endif()

if(USE_TENSORRT)
    find_dependency_unified(TensorRT REQUIRED)
endif()

if(USE_EXECUTORCH)
    # Let the catalog's find_package(executorch CONFIG) probe see a prebuilt install prefix
    # before the chain falls through to building ExecuTorch from source.
    if(EXECUTORCH_ROOTDIR)
        list(APPEND CMAKE_PREFIX_PATH "${EXECUTORCH_ROOTDIR}")
    endif()
    find_dependency_unified(ExecuTorch REQUIRED)
endif()

if(USE_OPENCV)
    find_dependency_unified(OpenCV REQUIRED)
else()
    find_dependency_unified(FFmpeg REQUIRED)
    find_dependency_unified(SDL2 REQUIRED)
endif()
find_dependency_unified(Threads REQUIRED)
find_dependency_unified(GTest REQUIRED)
find_dependency_unified(stb REQUIRED)
find_dependency_unified(font8x8 REQUIRED)

# --- Source and Include Directories ---
set(SOURCE_DIR "${CMAKE_SOURCE_DIR}/src")
set(TEST_DIR "${CMAKE_SOURCE_DIR}/tests")
set(DATA_DIR "${CMAKE_SOURCE_DIR}/data")
set(THIRD_PARTY_DIR "${CMAKE_SOURCE_DIR}/third_party")

# Create the data directory if it doesn't exist
file(MAKE_DIRECTORY "${DATA_DIR}")

# --- Main Library ---
set(RFDETR_SOURCES
    "${SOURCE_DIR}/rfdetr_inference.cpp"
    "${SOURCE_DIR}/processing_utils.cpp"
    "${SOURCE_DIR}/media.cpp"
    "${SOURCE_DIR}/video_reader.cpp"
    "${SOURCE_DIR}/video_writer.cpp"
    "${SOURCE_DIR}/display.cpp"
    "${SOURCE_DIR}/video_pipeline.cpp"
    "${SOURCE_DIR}/backends/inference_backend.cpp"
    "${THIRD_PARTY_DIR}/font8x8/font8x8_basic.c"
)

# Add backend-specific sources
if(USE_ONNX_RUNTIME)
    list(APPEND RFDETR_SOURCES "${SOURCE_DIR}/backends/onnx_runtime_backend.cpp")
endif()

if(USE_TENSORRT)
    list(APPEND RFDETR_SOURCES "${SOURCE_DIR}/backends/tensorrt_backend.cpp")
endif()

if(USE_EXECUTORCH)
    list(APPEND RFDETR_SOURCES "${SOURCE_DIR}/backends/executorch_backend.cpp")
endif()

add_library(rfdetr_inference_lib STATIC ${RFDETR_SOURCES})
target_compile_options(rfdetr_inference_lib PRIVATE ${PROJECT_WARNING_FLAGS})

target_include_directories(rfdetr_inference_lib
    PUBLIC
        "${SOURCE_DIR}"
)

# Link libraries (includes, compile definitions, and link lines for every
# resolved dependency are carried by the Deps::<Name> targets produced by the
# unified facade).
target_link_libraries(rfdetr_inference_lib
    PUBLIC
        Deps::Threads
        Deps::stb
        Deps::font8x8
)

if(USE_OPENCV)
    target_link_libraries(rfdetr_inference_lib PUBLIC Deps::OpenCV)
else()
    target_link_libraries(rfdetr_inference_lib PUBLIC Deps::FFmpeg Deps::SDL2)
endif()

if(USE_ONNX_RUNTIME)
    target_link_libraries(rfdetr_inference_lib PUBLIC Deps::OnnxRuntime)
endif()

if(USE_EXECUTORCH)
    target_link_libraries(rfdetr_inference_lib PUBLIC Deps::ExecuTorch)

    # The delegate library must be linked for its backend to self-register with the ExecuTorch
    # runtime; without it a .pte exported for that delegate fails at run time, not link time.
    if(EXECUTORCH_DELEGATE STREQUAL "xnnpack")
        target_link_libraries(rfdetr_inference_lib PUBLIC xnnpack_backend)
    endif()
endif()

if(USE_TENSORRT)
    target_link_libraries(rfdetr_inference_lib PUBLIC Deps::TensorRT)

    # RPATH so the executable finds TensorRT/CUDA libs without LD_LIBRARY_PATH.
    deps_get_rec(TensorRT RPATH_DIRS _TRT_RPATH_DIRS)
    set_target_properties(rfdetr_inference_lib PROPERTIES
        BUILD_RPATH "${_TRT_RPATH_DIRS}"
        INSTALL_RPATH "${_TRT_RPATH_DIRS}"
        BUILD_RPATH_USE_ORIGIN TRUE
        INSTALL_RPATH_USE_LINK_PATH TRUE
    )
endif()

# --- Main Application ---
add_executable(inference_app "${SOURCE_DIR}/main.cpp")
target_compile_options(inference_app PRIVATE ${PROJECT_WARNING_FLAGS})
target_link_libraries(inference_app PRIVATE rfdetr_inference_lib)

# Set RPATH for the executable
if(USE_TENSORRT)
    deps_get_rec(TensorRT RPATH_DIRS _TRT_RPATH_DIRS)
    set_target_properties(inference_app PROPERTIES
        BUILD_RPATH "${_TRT_RPATH_DIRS}"
        INSTALL_RPATH "${_TRT_RPATH_DIRS}"
        BUILD_RPATH_USE_ORIGIN TRUE
        INSTALL_RPATH_USE_LINK_PATH TRUE
    )
endif()

# Copy backend shared libraries next to the executable
if(USE_ONNX_RUNTIME)
    deps_get_rec(OnnxRuntime RUNTIME_LIBS _ORT_RUNTIME_LIBS)
    add_custom_command(TARGET inference_app POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${_ORT_RUNTIME_LIBS}
                $<TARGET_FILE_DIR:inference_app>
        COMMENT "Copying ONNX Runtime library to output directory"
    )
endif()

# --- Unit Tests ---
add_executable(unit_tests "${TEST_DIR}/unit/test_rfdetr_inference.cpp")
target_compile_options(unit_tests PRIVATE ${PROJECT_WARNING_FLAGS})
target_link_libraries(unit_tests PRIVATE rfdetr_inference_lib Deps::GTest)
target_include_directories(unit_tests PRIVATE "${TEST_DIR}/unit")

# --- Integration Tests ---
add_executable(integration_tests "${TEST_DIR}/integration/integration_test_rfdetr_inference.cpp")
target_compile_options(integration_tests PRIVATE ${PROJECT_WARNING_FLAGS})
target_link_libraries(integration_tests PRIVATE rfdetr_inference_lib Deps::GTest)

# Ensure the data directory exists before running integration tests
add_custom_command(TARGET integration_tests PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "${DATA_DIR}"
    COMMENT "Creating data directory for integration tests"
)

# --- Google Benchmark (opt-in via -DBENCHMARKS=ON) ---
option(BENCHMARKS "Build benchmark targets" OFF)
if(BENCHMARKS)
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    find_dependency_unified(GoogleBenchmark REQUIRED)

    add_executable(benchmarks "${TEST_DIR}/benchmark/bench_preprocessing.cpp")
    target_compile_options(benchmarks PRIVATE ${PROJECT_WARNING_FLAGS})
    target_link_libraries(benchmarks PRIVATE rfdetr_inference_lib Deps::GoogleBenchmark)
endif()

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

# --- Write dependency lockfile (records which handler + version resolved each dep) ---
deps_lock_write()

# --- Valgrind: memcheck (correctness) + profilers (callgrind/massif) ---
# memcheck runs valgrind directly on the unit_tests binary (more predictable
# than `ctest -T MemCheck`, which requires full CTest/Dart dashboard config).
# The profilers run on the `benchmarks` target (preferred) or `inference_app`,
# producing callgrind.out / massif.out.<pid>. Read with `callgrind_annotate` /
# `ms_print`; `perf record/report` is a lower-overhead alternative.
find_program(VALGRIND_EXECUTABLE NAMES valgrind)
mark_as_advanced(VALGRIND_EXECUTABLE)
if(VALGRIND_EXECUTABLE)
    set(VALGRIND_MEMCHECK_OPTS
        "--error-exitcode=1 --leak-check=full --show-leak-kinds=definite,indirect --errors-for-leak-kinds=definite,indirect"
        CACHE STRING "Flags passed to valgrind by the 'memcheck' target")
    if(EXISTS "${CMAKE_SOURCE_DIR}/valgrind.supp")
        string(APPEND VALGRIND_MEMCHECK_OPTS " --suppressions=${CMAKE_SOURCE_DIR}/valgrind.supp")
    endif()
    # Split into a CMake list so COMMAND passes each flag as a separate argument.
    separate_arguments(_VALGRIND_MEMCHECK_ARGS UNIX_COMMAND "${VALGRIND_MEMCHECK_OPTS}")

    add_custom_target(memcheck
        COMMAND ${VALGRIND_EXECUTABLE} ${_VALGRIND_MEMCHECK_ARGS} "$<TARGET_FILE:unit_tests>"
        DEPENDS unit_tests
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
        COMMENT "Running unit_tests under Valgrind memcheck"
        VERBATIM
    )

    if(TARGET benchmarks)
        set(_VALGRIND_PROFILE_TARGET benchmarks)
    else()
        set(_VALGRIND_PROFILE_TARGET inference_app)
    endif()
    set(VALGRIND_PROFILE_ARGS "" CACHE STRING "Extra args passed to the profiled binary by the callgrind/massif targets")
    separate_arguments(_VALGRIND_PROFILE_ARGS UNIX_COMMAND "${VALGRIND_PROFILE_ARGS}")

    add_custom_target(callgrind
        COMMAND ${VALGRIND_EXECUTABLE} --tool=callgrind --collect-jumps=yes --cache-sim=yes
                "$<TARGET_FILE:${_VALGRIND_PROFILE_TARGET}>" ${_VALGRIND_PROFILE_ARGS}
        DEPENDS ${_VALGRIND_PROFILE_TARGET}
        WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
        COMMENT "CPU/cache profile of ${_VALGRIND_PROFILE_TARGET} with callgrind -> callgrind.out.<pid>"
        VERBATIM
    )

    add_custom_target(massif
        COMMAND ${VALGRIND_EXECUTABLE} --tool=massif --stacks=yes
                "$<TARGET_FILE:${_VALGRIND_PROFILE_TARGET}>" ${_VALGRIND_PROFILE_ARGS}
        DEPENDS ${_VALGRIND_PROFILE_TARGET}
        WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
        COMMENT "Heap profile of ${_VALGRIND_PROFILE_TARGET} with massif -> massif.out.<pid>"
        VERBATIM
    )

    message(STATUS "Valgrind found: targets 'memcheck', 'callgrind', 'massif' available")
else()
    message(STATUS "Valgrind not found; memcheck/profiler targets disabled (install valgrind to enable)")
endif()
