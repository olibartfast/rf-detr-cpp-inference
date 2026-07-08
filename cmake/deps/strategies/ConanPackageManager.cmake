# cmake/deps/strategies/ConanPackageManager.cmake — Strategy: Conan 2.
# Active when CMAKE_TOOLCHAIN_FILE points at a conan_toolchain.cmake (produced by
# `conan install ... -of=<dir>`). The toolchain adds <dir> to CMAKE_PREFIX_PATH so
# find_package(<cmake-pkg> CONFIG) resolves the CMakeDeps-generated targets.
# Note: the Conan recipe name (CONAN_RECIPE) and the CMake package name
# (CONAN_FIND, e.g. recipe "gtest" -> package "GTest") can differ.

include_guard(GLOBAL)

function(deps_conan_can_resolve NAME OUT_VAR)
    deps_decl_get(${NAME} CONAN_FIND _find)
    if(NOT _find)
        set(${OUT_VAR} FALSE PARENT_SCOPE)
        return()
    endif()
    if(NOT CMAKE_TOOLCHAIN_FILE)
        set(${OUT_VAR} FALSE PARENT_SCOPE)
        return()
    endif()
    get_filename_component(_tc_name "${CMAKE_TOOLCHAIN_FILE}" NAME)
    if(NOT _tc_name STREQUAL "conan_toolchain.cmake")
        set(${OUT_VAR} FALSE PARENT_SCOPE)
        return()
    endif()
    get_filename_component(_conan_dir "${CMAKE_TOOLCHAIN_FILE}" DIRECTORY)
    set(_cfg "${_conan_dir}/${_find}Config.cmake")
    set(_findmod "${_conan_dir}/Find${_find}.cmake")
    if(EXISTS "${_cfg}" OR EXISTS "${_findmod}")
        set_property(GLOBAL PROPERTY deps.conan.${NAME}.dir "${_conan_dir}")
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
