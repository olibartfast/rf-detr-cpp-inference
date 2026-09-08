# ExecuTorch — runtime for .pte models exported by rfdetr >= 1.9.0
# (`model.export(format="executorch", backend="xnnpack")`).
#
# The op library (portable_ops_lib or optimized_native_cpu_ops_lib) is linked by CMakeLists.txt,
# not listed here: only one may be linked, and which one is available depends on whether the
# prefix was built with EXECUTORCH_BUILD_KERNELS_OPTIMIZED. rfdetr >= 1.9.1 exports need the
# optimized set for aten::linear.out.
#
# Resolution order follows the standard apt;provided chain:
#   1. apt      — find_package(executorch CONFIG) against a prebuilt install prefix. Point at it
#                 with -DEXECUTORCH_ROOTDIR=<prefix> (CMakeLists.txt puts that on CMAKE_PREFIX_PATH)
#                 or with -DCMAKE_PREFIX_PATH directly. This is the fast, recommended path.
#   2. provided — FETCHCONTENT: clone and build ExecuTorch from source. Correct but slow, and it
#                 needs a Python interpreter with ExecuTorch's build-time deps available, because
#                 ExecuTorch runs flatbuffers codegen during its own configure step.
#
# The delegate library (e.g. xnnpack_backend) is appended by CMakeLists.txt from
# EXECUTORCH_DELEGATE rather than declared here, since it is a build-time choice, not a coordinate.
deps_declare(ExecuTorch
    REQUIRED                 TRUE
    DEFINITIONS              USE_EXECUTORCH
    APT                      ON
    APT_METHOD               FIND_PACKAGE
    APT_FIND_NAME            executorch
    APT_IMPORTED_TARGETS     "executorch;extension_module_static;extension_tensor;portable_kernels"
    CONAN                    OFF
    VCPKG                    OFF
    PROVIDED_ACQUIRE         FETCHCONTENT
    PROVIDED_VERSION         "${EXECUTORCH_VERSION}"
    PROVIDED_FC_REPO         "https://github.com/pytorch/executorch.git"
    PROVIDED_FC_TAG          "${EXECUTORCH_VERSION}"
    PROVIDED_FC_SUBMODULES   RECURSE
    PROVIDED_FC_TARGETS      "executorch;extension_module_static;extension_tensor;optimized_native_cpu_ops_lib;portable_kernels"
    PROVIDED_FC_OPTIONS      "EXECUTORCH_BUILD_EXTENSION_MODULE=ON;EXECUTORCH_BUILD_EXTENSION_TENSOR=ON;EXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON;EXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP=ON;EXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON;EXECUTORCH_BUILD_XNNPACK=ON;EXECUTORCH_ENABLE_LOGGING=ON;EXECUTORCH_BUILD_TESTS=OFF;EXECUTORCH_BUILD_EXAMPLES=OFF"
)
