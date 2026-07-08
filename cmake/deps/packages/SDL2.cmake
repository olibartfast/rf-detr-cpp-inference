deps_declare(SDL2
    REQUIRED             OFF
    APT                  ON
    APT_METHOD           PKG_CONFIG
    APT_PKG_PREFIX       SDL2
    APT_PKG_MODULES      "sdl2"
    APT_IMPORTED_TARGETS "PkgConfig::SDL2"
    CONAN_RECIPE         sdl
    CONAN_FIND           sdl
    CONAN_TARGETS        "sdl::sdl"
    VCPKG_FIND           SDL2
    VCPKG_TARGETS        "SDL2::SDL2"
)
