# cmake/deps/strategies/ConanPackageManager.cmake — Strategy: Conan 2.
#
# Two activation modes:
#   1. CMakeToolchain (default): -DCMAKE_TOOLCHAIN_FILE=<dir>/conan_toolchain.cmake
#      The toolchain adds <dir> to CMAKE_PREFIX_PATH automatically.
#   2. CMakeDeps-only: -DDEPS_CONAN_DIR=<dir> (no conan_toolchain.cmake). Use this
#      to consume ConanCenter prebuilt binaries built for an ABI-compatible but
#      different compiler (e.g. gcc11/gnu17 binaries on a gcc13 system). The
#      strategy adds <dir> to CMAKE_PREFIX_PATH itself so find_package finds the
#      generated configs without the toolchain overriding the system compiler.
#
# Note: the Conan recipe name (CONAN_RECIPE) and the CMake package name
# (CONAN_FIND, e.g. recipe "gtest" -> package "GTest") can differ.

include_guard(GLOBAL)

function(deps_conan_can_resolve NAME OUT_VAR)
    deps_decl_get(${NAME} CONAN_FIND _find)
    if(NOT _find)
        set(${OUT_VAR} FALSE PARENT_SCOPE)
        return()
    endif()

    set(_conan_dir "")
    if(CMAKE_TOOLCHAIN_FILE)
        get_filename_component(_tc_name "${CMAKE_TOOLCHAIN_FILE}" NAME)
        if(_tc_name STREQUAL "conan_toolchain.cmake")
            get_filename_component(_conan_dir "${CMAKE_TOOLCHAIN_FILE}" DIRECTORY)
        endif()
    endif()
    if(DEPS_CONAN_DIR AND NOT _conan_dir)
        set(_conan_dir "${DEPS_CONAN_DIR}")
    endif()
    if(NOT _conan_dir)
        set(${OUT_VAR} FALSE PARENT_SCOPE)
        return()
    endif()

    set(_cfg "${_conan_dir}/${_find}Config.cmake")
    set(_findmod "${_conan_dir}/Find${_find}.cmake")
    if(EXISTS "${_cfg}" OR EXISTS "${_findmod}")
        set_property(GLOBAL PROPERTY deps.conan.${NAME}.dir "${_conan_dir}")
        list(APPEND CMAKE_PREFIX_PATH "${_conan_dir}")
        set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} PARENT_SCOPE)
        set(${OUT_VAR} TRUE PARENT_SCOPE)
        return()
    endif()
    set(${OUT_VAR} FALSE PARENT_SCOPE)
endfunction()

function(deps_conan_resolve NAME)
    deps_decl_get(${NAME} CONAN_FIND _find)
    find_package(${_find} CONFIG REQUIRED)
    deps_decl_get(${NAME} CONAN_TARGETS _targets)
    deps_decl_get(${NAME} CONAN_VERSION _ver)
    deps_decl_get(${NAME} DEFINITIONS _defs)
    deps_rec_set(${NAME} FOUND TRUE)
    deps_rec_set(${NAME} RESOLVED_BY "conan")
    deps_rec_set(${NAME} LIBRARIES "${_targets}")
    deps_rec_set(${NAME} DEFINITIONS "${_defs}")
    deps_rec_set(${NAME} VERSION "${_ver}")
endfunction()
