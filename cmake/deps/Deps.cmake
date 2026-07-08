# cmake/deps/Deps.cmake — Facade. Public options + module includes + registry
# load + resolver construction. Consumers call find_dependency_unified(<Name>).

include_guard(GLOBAL)

set(DEPS_MODE         "apt" CACHE STRING "Dependency ecosystem: apt|conan|vcpkg|auto")
set(DEPS_OFFLINE      OFF CACHE BOOL "Disable network lookups; ROOT provided only")
set(DEPS_PROVIDED_DIR "${CMAKE_BINARY_DIR}/_deps" CACHE PATH "Where provided downloads extract")
set(DEPS_DEBUG        OFF CACHE BOOL "Log dependency resolution decisions")

set_property(CACHE DEPS_MODE PROPERTY STRINGS apt conan vcpkg auto)

include("${CMAKE_CURRENT_LIST_DIR}/PackageManager.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/strategies/AptPackageManager.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/strategies/ProvidedPackageManager.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/strategies/ConanPackageManager.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/strategies/VcpkgPackageManager.cmake")

file(GLOB _dep_catalog "${CMAKE_CURRENT_LIST_DIR}/packages/*.cmake")
foreach(_f ${_dep_catalog})
    include("${_f}")
endforeach()

deps_make_resolver(${DEPS_MODE})
