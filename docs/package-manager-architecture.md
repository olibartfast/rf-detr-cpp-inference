# Dependency Resolution Architecture

Unified facade over package-manager strategies. Each dependency is declared once
in `cmake/deps/packages/<Name>.cmake`; the resolver picks the acquisition
strategy per `-DDEPS_MODE`.

```
find_dependency_unified(OnnxRuntime REQUIRED)
   └→ chain: apt → conan → vcpkg → provided  (first handler that can_resolve wins)
      └→ Deps::OnnxRuntime  (INTERFACE target; consumer just links it)
```

## Strategies

| strategy | activation | acquires via |
|---|---|---|
| **apt** (default) | always | `find_package` / `pkg_check_modules` (system packages) |
| **conan** | `CMAKE_TOOLCHAIN_FILE=conan_toolchain.cmake` **or** `-DDEPS_CONAN_DIR=<dir>` | Conan 2 CMakeDeps `find_package` |
| **vcpkg** | `CMAKE_TOOLCHAIN_FILE=vcpkg.cmake` | vcpkg manifest mode `find_package` (CONFIG + MODULE) |
| **provided** | always (fallback) | `DOWNLOAD` (pinned URL), `VENDORED` (third_party/), `FETCHCONTENT` (git clone), `ROOT` (user path) |

`DEPS_MODE=auto` chains all four. Each strategy auto-detects whether it's active
(toolchain file or output dir), so default builds never trigger conan/vcpkg.

## Module layout

```
cmake/deps/
├── Deps.cmake                    facade + public options
├── PackageManager.cmake          registry, chain driver, builder, lockfile
├── strategies/
│   ├── AptPackageManager.cmake       find_package / pkg_check_modules
│   ├── ConanPackageManager.cmake     CMakeDeps find_package (+ CMakeDeps-only mode)
│   ├── VcpkgPackageManager.cmake     manifest mode find_package
│   └── ProvidedPackageManager.cmake  download / vendored / fetchcontent / root
└── packages/                    one file per dependency (the catalog)
    ├── OnnxRuntime.cmake  TensorRT.cmake  ExecuTorch.cmake  OpenCV.cmake
    ├── FFmpeg.cmake  SDL2.cmake  Threads.cmake  GTest.cmake
    └── GoogleBenchmark.cmake  stb.cmake  font8x8.cmake
```

Patterns: Facade (`find_dependency_unified`), Strategy (per-ecosystem), Chain of
Responsibility (`deps_resolve`), Registry (`deps_declare`), Builder
(`deps_build_target` → `Deps::<Name>`).

## Catalog declaration

Flat key/value — one `deps_declare` per dependency:

```cmake
deps_declare(OnnxRuntime
    REQUIRED            TRUE
    DEFINITIONS         USE_ONNX_RUNTIME
    APT                 OFF
    PROVIDED_ACQUIRE    DOWNLOAD
    PROVIDED_URL        "https://github.com/microsoft/onnxruntime/releases/download/v1.21.0/onnxruntime-linux-x64-1.21.0.tgz"
    PROVIDED_VERSION    "1.21.0"
    PROVIDED_SUBDIR     "onnxruntime-linux-x64-1.21.0"
    PROVIDED_INCLUDE    "include"
    PROVIDED_LIBRARY    "lib/libonnxruntime.so.1.21.0"
    PROVIDED_HEADER_GUARD "include/onnxruntime_cxx_api.h"
    PROVIDED_RUNTIME_LIBS "lib/libonnxruntime.so.1.21.0"
    PROVIDED_ROOT_CACHE "ONNXRUNTIME_ROOTDIR"
    PROVIDED_ROOT_VARS  "ONNXRUNTIME_ROOTDIR;OnnxRuntime_ROOT"
)
```

To add a conan/vcpkg coordinate: add `CONAN_FIND`/`CONAN_TARGETS` or
`VCPKG_FIND`/`VCPKG_TARGETS` to the catalog entry, and add the package to
`conanfile.txt` or `vcpkg.json`.

### Install-prefix first, source build as fallback

A dependency can prefer an existing install prefix and fall back to building from
source without any new precedence logic, by mapping onto the standard `apt;provided`
chain — the `apt` link runs `find_package(<name> CONFIG)`, and `provided` only runs
if that misses. ExecuTorch uses this:

```cmake
deps_declare(ExecuTorch
    DEFINITIONS              USE_EXECUTORCH
    APT                      ON
    APT_METHOD               FIND_PACKAGE      # 1st: find_package(executorch CONFIG)
    APT_FIND_NAME            executorch
    APT_IMPORTED_TARGETS     "executorch;extension_module_static;..."
    PROVIDED_ACQUIRE         FETCHCONTENT      # 2nd: clone + build from source
    PROVIDED_FC_REPO         "https://github.com/pytorch/executorch.git"
    PROVIDED_FC_TAG          "v1.3.1"
    PROVIDED_FC_SUBMODULES   RECURSE
    PROVIDED_FC_OPTIONS      "EXECUTORCH_BUILD_XNNPACK=ON;..."
)
```

`CMakeLists.txt` puts `EXECUTORCH_ROOTDIR` on `CMAKE_PREFIX_PATH` so the `find_package`
link can see a user-supplied prefix.

### FetchContent build configuration

Two keys let a `FETCHCONTENT` dependency configure its own build — both are additive and
unused by the existing `GTest` / `GoogleBenchmark` entries:

| key | effect |
|---|---|
| `PROVIDED_FC_OPTIONS` | `NAME=VALUE` list seeded into the CMake cache before `FetchContent_MakeAvailable`. Set **without** `FORCE`, so an explicit `-DNAME=...` on the command line still wins. A malformed entry is a `FATAL_ERROR`, not a silent skip. |
| `PROVIDED_FC_SUBMODULES` | `RECURSE` passes `GIT_SUBMODULES_RECURSE TRUE` to `FetchContent_Declare` (requires CMake 3.17+). |

## Dependency mapping

| dependency | apt mode | conan mode | vcpkg mode |
|---|---|---|---|
| ONNX Runtime 1.21.0 | provided-download | provided-download | provided-download |
| TensorRT 10.13.3.9 | provided-download | provided-download | provided-download |
| ExecuTorch v1.3.1 | apt-find_package, else provided-FetchContent | same | same |
| OpenCV | apt | **conan** | vcpkg (slow) |
| FFmpeg | apt | **conan** | **vcpkg** |
| SDL2 | apt | **conan** | **vcpkg** |
| Threads | apt | apt (fallback) | apt (fallback) |
| GTest | provided-FetchContent | **conan** | **vcpkg** |
| Google Benchmark | provided-FetchContent | conan-ready | vcpkg-ready |
| stb, font8x8 | provided-vendored | provided-vendored | provided-vendored |

ONNX Runtime/TensorRT stay provided-download (registries have wrong versions or
are absent). ExecuTorch is not in conan or vcpkg either, so both modes fall through
to the same `find_package`-then-FetchContent path as `apt` mode. OpenCV and FFmpeg
are mutually exclusive (`USE_OPENCV` option);
each conan graph resolves independently. In `conan` and `vcpkg` modes, `apt` is
chained as a fallback for system packages (Threads).

## Options

| option | default | effect |
|---|---|---|
| `DEPS_MODE` | `apt` | `apt` `conan` `vcpkg` `auto` |
| `DEPS_CONAN_DIR` | (empty) | CMakeDeps-only mode (no toolchain, keeps system compiler) |
| `DEPS_OFFLINE` | `OFF` | ROOT lookups only (no network) |
| `DEPS_PROVIDED_DIR` | `<build>/_deps` | download extraction dir |
| `DEPS_DEBUG` | `OFF` | log resolution decisions |
| `<NAME>_ROOT` | (user) | per-dep override |

## Lockfile

`deps.lock.json` is written to `${CMAKE_BINARY_DIR}` at configure end, recording
which handler + version resolved each dependency:

```json
{"dependencies":{"OnnxRuntime":{"handler":"provided","version":"1.21.0"},...}}
```

## Conan CMakeDeps-only mode

Consumes ConanCenter prebuilt binaries (e.g. gcc11/cpp17) with the system
compiler (e.g. gcc13), without the conan toolchain overriding CC/CXX:

```bash
conan install . -pr gcc11-bin -of=build/conan-deps -g=CMakeDeps
cmake -S . -B build -DDEPS_MODE=auto -DDEPS_CONAN_DIR=build/conan-deps
```
