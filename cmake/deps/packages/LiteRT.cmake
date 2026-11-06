# Stage the official SDK headers and matching native runtime with scripts/fetch_litert.sh.
# ROOT-only: loading the dependency catalog must work on targets without LiteRT.
deps_declare(LiteRT
    REQUIRED TRUE
    DEFINITIONS USE_LITERT
    APT OFF
    PROVIDED_ACQUIRE ROOT
    PROVIDED_VERSION "${LITERT_VERSION}"
    PROVIDED_INCLUDE "include"
    PROVIDED_LIBRARY "lib/${CMAKE_SHARED_LIBRARY_PREFIX}LiteRt${CMAKE_SHARED_LIBRARY_SUFFIX}"
    PROVIDED_HEADER_GUARD "include/litert/c/litert_compiled_model.h"
    PROVIDED_ROOT_CACHE "LITERT_ROOTDIR"
    PROVIDED_ROOT_VARS "LITERT_ROOTDIR;LiteRT_ROOT"
)
