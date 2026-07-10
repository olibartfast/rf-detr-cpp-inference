# cmake/deps/strategies/VcpkgPackageManager.cmake — Strategy: vcpkg manifest mode.
# Active when CMAKE_TOOLCHAIN_FILE points at vcpkg.cmake. In manifest mode vcpkg
# builds/installs every port listed in vcpkg.json at the start of configure.
# Some ports provide CONFIG files with IMPORTED targets (SDL2, GTest); others
# provide only Find modules with variables (FFmpeg). The strategy tries CONFIG
# first, falls back to MODULE, and uses targets when available or variables
# (<FIND>_LIBRARIES / <FIND>_INCLUDE_DIRS) otherwise.

include_guard(GLOBAL)

function(deps_vcpkg_can_resolve NAME OUT_VAR)
    set(_ok FALSE)
    deps_decl_get(${NAME} VCPKG_FIND _find)
    if(NOT _find)
        set(${OUT_VAR} FALSE PARENT_SCOPE)
        return()
    endif()
    if(NOT CMAKE_TOOLCHAIN_FILE)
        set(${OUT_VAR} FALSE PARENT_SCOPE)
        return()
    endif()
    get_filename_component(_tc_name "${CMAKE_TOOLCHAIN_FILE}" NAME)
    if(NOT _tc_name STREQUAL "vcpkg.cmake")
        set(${OUT_VAR} FALSE PARENT_SCOPE)
        return()
    endif()
    find_package(${_find} CONFIG QUIET)
    if(NOT ${_find}_FOUND)
        find_package(${_find} MODULE QUIET)
    endif()
    if(${_find}_FOUND)
        set(_ok TRUE)
    endif()
    set(${OUT_VAR} ${_ok} PARENT_SCOPE)
endfunction()

function(deps_vcpkg_resolve NAME)
    deps_decl_get(${NAME} VCPKG_FIND _find)
    find_package(${_find} CONFIG QUIET)
    if(NOT ${_find}_FOUND)
        find_package(${_find} MODULE REQUIRED)
    endif()
    deps_decl_get(${NAME} VCPKG_TARGETS _targets)
    deps_decl_get(${NAME} DEFINITIONS _defs)
    deps_rec_set(${NAME} FOUND TRUE)
    deps_rec_set(${NAME} RESOLVED_BY "vcpkg")

    # Prefer IMPORTED targets when the catalog declares them and they exist.
    if(_targets)
        list(GET _targets 0 _first)
        if(TARGET ${_first})
            deps_rec_set(${NAME} LIBRARIES "${_targets}")
            return()
        endif()
    endif()

    # Fall back to <FIND>_LIBRARIES / <FIND>_INCLUDE_DIRS variables.
    deps_rec_set(${NAME} LIBRARIES "${${_find}_LIBRARIES}")
    if(${_find}_INCLUDE_DIRS)
        deps_rec_set(${NAME} INCLUDE_DIRS "${${_find}_INCLUDE_DIRS}")
    endif()
    deps_rec_set(${NAME} DEFINITIONS "${_defs}")
endfunction()
