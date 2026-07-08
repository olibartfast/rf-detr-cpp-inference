# cmake/deps/PackageManager.cmake — Registry (deps_declare) + Chain of Responsibility
# driver (deps_resolve) + IMPORTED-target Builder + Facade (find_dependency_unified).

include_guard(GLOBAL)

# Vocabulary of declaration keys. Values that are themselves lists
# (e.g. "libavcodec;libavformat") flatten when expanded through ARGN, so the
# parser below accumulates tokens into the current value until it sees the next
# known key.
set_property(GLOBAL PROPERTY deps.keyset
    "REQUIRED;DEFINITIONS;APT;CONAN;VCPKG;\
CONAN_RECIPE;CONAN_FIND;CONAN_VERSION;CONAN_TARGETS;\
VCPKG_FIND;VCPKG_TARGETS;\
PROVIDED_ACQUIRE;PROVIDED_URL;PROVIDED_VERSION;PROVIDED_SUBDIR;PROVIDED_INCLUDE;\
PROVIDED_LIBRARY;PROVIDED_LIBRARIES;PROVIDED_HEADER_GUARD;PROVIDED_RUNTIME_LIBS;\
PROVIDED_ROOT_CACHE;PROVIDED_ROOT_VARS;PROVIDED_CUDA;PROVIDED_VENDORED_DIR;\
PROVIDED_FC_REPO;PROVIDED_FC_TAG;PROVIDED_FC_TARGETS;\
APT_METHOD;APT_FIND_NAME;APT_COMPONENTS;APT_LINK_VARS;APT_INCLUDE_VAR;\
APT_VERSION_VAR;APT_PKG_PREFIX;APT_PKG_MODULES;APT_IMPORTED_TARGETS")

function(deps_declare NAME)
    get_property(_keys GLOBAL PROPERTY deps.keyset)
    set(_cur_key "")
    set(_cur_val "")
    foreach(_t ${ARGN})
        if(_t IN_LIST _keys)
            if(_cur_key)
                set_property(GLOBAL PROPERTY "deps.decl.${NAME}.${_cur_key}" "${_cur_val}")
            endif()
            set(_cur_key "${_t}")
            set(_cur_val "")
        else()
            if(_cur_val STREQUAL "")
                set(_cur_val "${_t}")
            else()
                set(_cur_val "${_cur_val};${_t}")
            endif()
        endif()
    endforeach()
    if(_cur_key)
        set_property(GLOBAL PROPERTY "deps.decl.${NAME}.${_cur_key}" "${_cur_val}")
    endif()
    set_property(GLOBAL PROPERTY "deps.decl.${NAME}.declared" TRUE)
endfunction()

function(deps_decl_get NAME KEY OUT_VAR)
    get_property(_v GLOBAL PROPERTY "deps.decl.${NAME}.${KEY}")
    set(${OUT_VAR} "${_v}" PARENT_SCOPE)
endfunction()

function(deps_rec_set NAME FIELD VALUE)
    set_property(GLOBAL PROPERTY "deps.rec.${NAME}.${FIELD}" "${VALUE}")
endfunction()

function(deps_get_rec NAME FIELD OUT_VAR)
    get_property(_v GLOBAL PROPERTY "deps.rec.${NAME}.${FIELD}")
    set(${OUT_VAR} "${_v}" PARENT_SCOPE)
endfunction()

function(deps_make_resolver MODE)
    set(_offline "")
    if(DEPS_OFFLINE)
        set(_chain "provided")
        set(_offline ",offline")
    elseif(MODE STREQUAL "apt")
        set(_chain "apt;provided")
    elseif(MODE STREQUAL "conan")
        set(_chain "conan;provided")
    elseif(MODE STREQUAL "vcpkg")
        set(_chain "vcpkg;provided")
    elseif(MODE STREQUAL "auto")
        set(_chain "apt;conan;vcpkg;provided")
    else()
        message(FATAL_ERROR "Unknown DEPS_MODE='${MODE}' (expected apt|conan|vcpkg|auto)")
    endif()
    set_property(GLOBAL PROPERTY deps.chain "${_chain}")
    if(DEPS_DEBUG)
        message(STATUS "[deps] resolver chain (${MODE}${_offline}): ${_chain}")
    endif()
endfunction()

function(deps_resolve NAME OUT_HANDLER)
    get_property(_chain GLOBAL PROPERTY deps.chain)
    if(_chain STREQUAL "")
        message(FATAL_ERROR "[deps] resolver not built; call deps_make_resolver() first")
    endif()
    foreach(_tok ${_chain})
        set(_ok FALSE)
        if(_tok STREQUAL "apt")
            deps_apt_can_resolve(${NAME} _ok)
        elseif(_tok STREQUAL "conan")
            deps_conan_can_resolve(${NAME} _ok)
        elseif(_tok STREQUAL "vcpkg")
            deps_vcpkg_can_resolve(${NAME} _ok)
        elseif(_tok STREQUAL "provided")
            deps_provided_can_resolve(${NAME} _ok)
        else()
            message(FATAL_ERROR "[deps] unknown strategy token '${_tok}' in chain")
        endif()
        if(_ok)
            if(_tok STREQUAL "apt")
                deps_apt_resolve(${NAME})
            elseif(_tok STREQUAL "conan")
                deps_conan_resolve(${NAME})
            elseif(_tok STREQUAL "vcpkg")
                deps_vcpkg_resolve(${NAME})
            elseif(_tok STREQUAL "provided")
                deps_provided_resolve(${NAME})
            endif()
            set(${OUT_HANDLER} "${_tok}" PARENT_SCOPE)
            return()
        endif()
    endforeach()
    set(${OUT_HANDLER} "" PARENT_SCOPE)
endfunction()

# Builds the Deps::<Name> consumer-facing target. Uses a non-IMPORTED INTERFACE
# library (deps_<name>_internal) with an ALIAS (Deps::<name>), not an IMPORTED
# target: IMPORTED targets are strictly validated at generation time and reject
# usage-requirement paths that don't yet exist (e.g. FetchContent source dirs),
# while these wrappers are build-internal and never installed/exported.
function(deps_build_target NAME)
    if(TARGET Deps::${NAME})
        return()
    endif()
    get_property(_found GLOBAL PROPERTY "deps.rec.${NAME}.FOUND")
    get_property(_inc   GLOBAL PROPERTY "deps.rec.${NAME}.INCLUDE_DIRS")
    get_property(_lib   GLOBAL PROPERTY "deps.rec.${NAME}.LIBRARIES")
    get_property(_def   GLOBAL PROPERTY "deps.rec.${NAME}.DEFINITIONS")
    get_property(_by    GLOBAL PROPERTY "deps.rec.${NAME}.RESOLVED_BY")
    get_property(_ver   GLOBAL PROPERTY "deps.rec.${NAME}.VERSION")

    add_library(deps_${NAME}_internal INTERFACE)
    if(_inc)
        target_include_directories(deps_${NAME}_internal SYSTEM INTERFACE ${_inc})
    endif()
    if(_lib)
        target_link_libraries(deps_${NAME}_internal INTERFACE ${_lib})
    endif()
    if(_def)
        target_compile_definitions(deps_${NAME}_internal INTERFACE ${_def})
    endif()
    add_library(Deps::${NAME} ALIAS deps_${NAME}_internal)

    if(DEPS_DEBUG AND _found)
        message(STATUS "[deps] ${NAME} resolved by ${_by} (version ${_ver}) -> Deps::${NAME}")
    endif()
endfunction()

function(find_dependency_unified NAME)
    set(options REQUIRED)
    cmake_parse_arguments(ARG "${options}" "" "" ${ARGN})
    deps_resolve(${NAME} _handler)
    deps_build_target(${NAME})
    if(NOT _handler)
        if(ARG_REQUIRED)
            get_property(_chain GLOBAL PROPERTY deps.chain)
            message(FATAL_ERROR
                "[deps] REQUIRED dependency '${NAME}' could not be resolved by any handler "
                "(tried: ${_chain}). Declare it under cmake/deps/packages/ or supply it "
                "via apt/conan/vcpkg, or set -D${NAME}_ROOT=/path/to/it.")
        elseif(DEPS_DEBUG)
            message(STATUS "[deps] optional '${NAME}' not resolved (skipped)")
        endif()
    endif()
endfunction()
