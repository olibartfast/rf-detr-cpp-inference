# NVIDIA DALI — GPU preprocessing (nvJPEG decode, resize, normalise) hosted
# in-process through the DALI C API (`dali/c_api.h`).
#
# ROOT-only on purpose. NVIDIA publishes no standalone C++ DALI distribution:
# the headers and shared libraries ship inside a pip wheel whose filename carries
# an opaque build number (nvidia_dali_cuda120-1.51.2-<build>-py3-none-...whl), so
# a pinned PROVIDED_URL cannot be kept working. Acquire with
# `scripts/fetch_dali.sh` (extracts the libraries from a pinned Triton container)
# and point the build at the result with -DDALI_ROOT=<dir>.
#
# libdali.so and libdali_operators.so must both be linked. libdali.so pulls in
# libdali_core.so and libdali_kernels.so through DT_NEEDED, but *not* the operator
# library — DALI's Python bindings dlopen that one. Without it every pipeline
# fails at run time with `No schema found for operator "decoders__Image"`, and
# the C++ side must additionally call daliInitOperators() to register the schemas
# (see dali_preprocessor.cpp).
deps_declare(DALI
    REQUIRED              TRUE
    DEFINITIONS           USE_DALI
    APT                   OFF
    PROVIDED_ACQUIRE      ROOT
    PROVIDED_INCLUDE      "include"
    PROVIDED_LIBRARIES    "libdali.so;libdali_operators.so"
    PROVIDED_HEADER_GUARD "include/dali/c_api.h"
    PROVIDED_ROOT_CACHE   "DALI_ROOTDIR"
    PROVIDED_ROOT_VARS    "DALI_ROOTDIR;DALI_ROOT"
)
