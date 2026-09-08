# CUDA Toolkit — provides cudart (device buffers, streams, memcpy) and the
# header-only CUB used by the postprocessing kernels. CMake ships a first-class
# FindCUDAToolkit module, so the apt handler resolves this everywhere; there is
# no download fallback because the toolkit must match the installed driver.
deps_declare(CUDAToolkit
    REQUIRED             TRUE
    APT                  ON
    APT_METHOD           FIND_PACKAGE
    APT_FIND_NAME        CUDAToolkit
    APT_IMPORTED_TARGETS "CUDA::cudart"
    APT_INCLUDE_VAR      "CUDAToolkit_INCLUDE_DIRS"
    APT_VERSION_VAR      "CUDAToolkit_VERSION"
)
