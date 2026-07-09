# Package Manager Abstraction Architecture

A pattern-based design that wraps and encapsulates three dependency-resolution
policies behind one uniform facade, so the rest of the build describes
dependencies once and lets the resolver pick the acquisition strategy.

```
  three policies (Strategy), each a Chain: [primary source  ->  PROVIDED fallback]
  ┌───────────────┐   ┌───────────────┐   ┌───────────────┐
  │  apt+provided │   │ conan+provided│   │ vcpkg+provided│
  │ [APT,PROVIDED]│   │[CONAN,PROVIDED│   │[VCPKG,PROVIDED│
  └───────┬───────┘   └───────┬───────┘   └───────┬───────┘
          └───────────────────┴───────────────────┘
                              │  selected by -DDEPS_MODE=apt|conan|vcpkg|auto
                              ▼
                 ┌─────────────────────────┐
                 │ find_dependency_unified │  ← Facade (one entry point)
                 └─────────────────────────┘
```

The `PROVIDED` block is a **shared collaborator**, reused as the last link of
every chain. "Provided" means any of: a pinned download URL + checksum + extract
layout (today's ONNX Runtime / TensorRT flow), a vendored header/source under
`third_party/` (stb, font8x8), a local Conan recipe / vcpkg port overlay, or a
user-supplied `-D<NAME>_ROOT=...` path.

## 1. Why this design

Today `CMakeLists.txt` mixes four ad-hoc mechanisms with no abstraction:

| mechanism | used for | fragility |
|---|---|---|
| manual `file(DOWNLOAD)` + `tar` (~80 lines each) | ONNX Runtime, TensorRT | duplicated; NVIDIA URL rate-limits; version bump edits 5+ vars |
| `find_package` / `pkg_check_modules` (system apt) | OpenCV, FFmpeg, SDL2, Threads | no fallback if apt package missing |
| `FetchContent` (git clone) | GoogleTest, Google Benchmark | unconditional clone even in release |
| vendored headers | stb, font8x8 | fine, but unmanaged |

This architecture replaces the first two with one call site, makes the
acquisition strategy swappable, and gives Conan/vcpkg a defined integration
shape (stubs now, fill in later) so the project is not locked to apt + manual
downloads.

## 2. Design patterns catalog

| pattern | role in this design | where |
|---|---|---|
| **Facade** | single entry point `find_dependency_unified(NAME ...)` hides the whole machinery | `Deps.cmake` |
| **Strategy** | one policy object per ecosystem (`apt` / `conan` / `vcpkg`); `DEPS_MODE` selects which | `strategies/*.cmake` |
| **Chain of Responsibility** | each policy is a chain `[primary, PROVIDED]`; first handler that *can* resolve wins; `auto` mode chains all primaries then PROVIDED | `PackageManager.cmake` driver |
| **Specification** | every handler answers a `can_resolve(dep)` predicate before attempting `resolve(dep)` | each strategy module |
| **Template Method** | `PackageManagerBase` defines `resolve()` = `if can_resolve then acquire else skip`; concrete handlers override `acquire()` only | `PackageManager.cmake` |
| **Adapter** | each handler normalizes its native representation (dpkg list / Conan CMake config / vcpkg triplet / manual paths) into one `ResolvedDependency` record | all strategies |
| **Registry / Catalog** | declarative per-dependency, per-strategy coordinates — single source of truth | `packages/*.cmake` |
| **Factory** | `deps_make_resolver()` builds the chain from options/env at configure time | `PackageManager.cmake` |
| **Builder** | `deps_build_imported_target()` constructs a CMake `IMPORTED` / `INTERFACE` target from a `ResolvedDependency` | `PackageManager.cmake` |
| **Cache / Lockfile** | `deps.lock.json` records which handler + version satisfied each dep for reproducible reconfigures | future (hooks present) |

## 3. Layered architecture

```
┌──────────────────────────────────────────────────────────────┐
│  Consumer: CMakeLists.txt                                     │
│     find_dependency_unified(OnnxRuntime REQUIRED)             │
│     find_dependency_unified(OpenCV COMPONENTS core imgcodecs) │
└──────────────────────────────┬───────────────────────────────┘
                               │ Facade
┌──────────────────────────────▼───────────────────────────────┐
│  Resolution layer                                             │
│   DependencyResolver  (Chain of Responsibility driver)        │
│   deps_make_resolver  (Factory)                               │
│   ResolutionCache     (deps.lock.json — future)              │
└──────────────────────────────┬───────────────────────────────┘
                               │
┌──────────────────────────────▼───────────────────────────────┐
│  Strategy layer   — IPackageManager                          │
│   PackageManagerBase (Template Method: can_resolve/acquire)  │
│  ┌────────────┬────────────┬────────────┬────────────────┐  │
│  │ Apt        │ Conan      │ Vcpkg      │ Provided       │  │
│  │ (FULL)     │ (STUB)     │ (STUB)     │ (FULL, shared) │  │
│  └────────────┴────────────┴────────────┴────────────────┘  │
└──────────────────────────────┬───────────────────────────────┘
                               │ Adapter
┌──────────────────────────────▼───────────────────────────────┐
│  Normalization layer                                          │
│   ResolvedDependency  →  deps_build_imported_target (Builder)│
│   → IMPORTED / INTERFACE CMake target                         │
└──────────────────────────────┬───────────────────────────────┘
                               │ reads coordinates
┌──────────────────────────────▼───────────────────────────────┐
│  Catalog layer  (Registry)                                    │
│   cmake/deps/packages/<Name>.cmake  — per-dep, per-strategy   │
│   third_party/   vendored sources (stb, font8x8)              │
│   recipes/       local Conan recipe overlays (future)         │
│   ports/         local vcpkg port overlays (future)           │
└──────────────────────────────────────────────────────────────┘
```

## 4. Module layout

```
cmake/deps/
├── Deps.cmake                    Facade + public options (DEPS_MODE, DEPS_OFFLINE, ...)
├── PackageManager.cmake          core types, chain driver (Factory + CoR), IMPORTED-target Builder
├── strategies/
│   ├── AptPackageManager.cmake       FULL: find_package / pkg_check_modules
│   ├── ProvidedPackageManager.cmake  FULL: download+extract / vendored / user-ROOT
│   ├── ConanPackageManager.cmake       FULL: conan toolchain + CMakeDeps find_package
│   └── VcpkgPackageManager.cmake      FULL: vcpkg manifest mode + toolchain find_package
└── packages/
    ├── OnnxRuntime.cmake         provided: pinned URL
    ├── TensorRT.cmake            provided: pinned URL + CUDA
    ├── OpenCV.cmake              apt: find_package COMPONENTS
    ├── FFmpeg.cmake              apt: pkg_check_modules
    ├── SDL2.cmake                apt: pkg_check_modules
    ├── Threads.cmake             apt: find_package(Threads)
    ├── GoogleTest.cmake          catalog entry (FetchContent today; provided-ready)
    └── GoogleBenchmark.cmake     catalog entry (FetchContent today; provided-ready)
```

## 5. Interfaces (CMake emulation of OOP)

CMake has no classes; patterns are emulated with a documented calling
convention. A strategy is any module that defines these two functions:

```cmake
# Specification predicate
deps_<strategy>_can_resolve(<dep_name> <out_var>)
#   sets <out_var>=TRUE when this strategy can satisfy <dep_name>

# Template Method "acquire" hook — fills a ResolvedDependency record
deps_<strategy>_resolve(<dep_name> <out_record>)
```

### `ResolvedDependency` record (the Adapter's output)

A flat variable namespace so it is trivially inspectable:

| field | meaning |
|---|---|
| `${REC}.FOUND` | `TRUE`/`FALSE` |
| `${REC}.INCLUDE_DIRS` | `;`-list of include roots |
| `${REC}.LIBRARIES` | `;`-list of lib files / imported targets |
| `${REC}.DEFINITIONS` | `;`-list of compile defs (e.g. `USE_ONNX_RUNTIME`) |
| `${REC}.VERSION` | resolved version string |
| `${REC}.RESOLVED_BY` | `apt`/`conan`/`vcpkg`/`provided` |
| `${REC}.RPATH_DIRS` | dirs to add to BUILD/INSTALL_RPATH (TensorRT) |
| `${REC}.RUNTIME_LIBS` | shared libs to copy next to the exe (ONNX Runtime `.so`) |

The Builder turns this into a `IMPORTED` / `INTERFACE` target named
`Deps::<Name>` that consumers link against, so strategy details never leak past
the facade.

### Catalog declaration (Registry)

Each `packages/<Name>.cmake` calls one declarative macro:

```cmake
deps_declare(OnnxRuntime
    REQUIRED      TRUE
    DEFINITIONS   USE_ONNX_RUNTIME
    APT           OFF                                  # not available via apt
    CONAN         "onnxruntime/1.21.0"                 # ConanCenter coordinate (future)
    VCPKG         "onnxruntime"                        # vcpkg port (future)
    PROVIDED
        ACQUIRE     DOWNLOAD                           # DOWNLOAD | VENDORED | ROOT
        URL         "https://github.com/microsoft/onnxruntime/releases/download/v1.21.0/onnxruntime-linux-x64-1.21.0.tgz"
        VERSION     "1.21.0"
        SUBDIR      "onnxruntime-linux-x64-1.21.0"
        INCLUDE     "include"
        LIBRARY     "lib/libonnxruntime.so.1.21.0"
        HEADER_GUARD "include/onnxruntime_cxx_api.h"
        RUNTIME_LIB "lib/libonnxruntime.so.1.21.0"    # copy next to exe
)
```

## 6. Resolution algorithm

For `find_dependency_unified(NAME ...)`:

1. Load `packages/<NAME>.cmake` → registry entry.
2. Build chain from `DEPS_MODE`:
   - `apt`   → `[APT, PROVIDED]`
   - `conan` → `[CONAN, PROVIDED]`
   - `vcpkg` → `[VCPKG, PROVIDED]`
   - `auto`  → `[APT, CONAN, VCPKG, PROVIDED]`
3. For each handler `H` in chain:
   a. `deps_<H>_can_resolve(NAME, ok)` (Specification).
   b. If `ok`: `deps_<H>_resolve(NAME, REC)` (Template Method acquire).
   c. `deps_build_imported_target(NAME, REC)` (Builder/Adapter). Done.
4. If no handler resolves and `REQUIRED` → `FATAL_ERROR` naming the tried
   handlers and the suggested install; else create an empty `Deps::<NAME>`
   INTERFACE target and continue.

Offline mode (`-DDEPS_OFFLINE=ON`) disables every handler except `ROOT`-style
provided lookups (no network, no apt queries) for air-gapped / hermetic builds.

## 7. Dependency mapping (current → new)

| dependency | today | mode | new source | notes |
|---|---|---|---|---|
| ONNX Runtime 1.21.0 | manual download (80 lines) | all | PROVIDED `DOWNLOAD` | registries have 1.23+/1.24+ — wrong header API; stays provided |
| TensorRT 10.13.3.9 + CUDA 13.0 | manual download + `find_package(CUDA)` | all | PROVIDED `DOWNLOAD` | absent from both registries |
| OpenCV (optional) | `find_package` COMPONENTS | apt | APT | `USE_OPENCV` gates inclusion; vcpkg works but 20-40 min source build |
| FFmpeg libs | `pkg_check_modules` | apt | APT | alt media backend; vcpkg works but 15-20 min source build |
| SDL2 | `pkg_check_modules` | apt | APT | conan needs 20+ X11 -dev packages; kept on apt |
| Threads | `find_package(Threads)` | apt | APT | always available |
| GoogleTest 1.12.1 | `FetchContent` | apt | PROVIDED `FETCHCONTENT` | routed through facade; conan + vcpkg both verified e2e |
| Google Benchmark 1.9.1 | `FetchContent` | apt | PROVIDED `FETCHCONTENT` | routed through facade; conan + vcpkg coordinates declared |
| stb, font8x8 | vendored SYSTEM include | all | direct `target_include_directories` | unchanged |

## 8. Public options

| option | default | effect |
|---|---|---|
| `DEPS_MODE` | `apt` | `apt` / `conan` / `vcpkg` / `auto` — which ecosystem drives resolution |
| `DEPS_OFFLINE` | `OFF` | disable network/apt lookups; only `ROOT` provided lookups |
| `DEPS_PROVIDED_DIR` | `${CMAKE_BINARY_DIR}/_deps` | where provided downloads extract |
| `DEPS_DEBUG` | `OFF` | log the chosen handler + version per dependency |
| `<NAME>_ROOT` | (user) | per-dep override bypassing the chain (provided `ROOT`) |

## 9. Conan / vcpkg integration (implemented)

Both strategies are real and verified end-to-end (see §11). They are **activated
by the toolchain**, not by a flag: each detects `CMAKE_TOOLCHAIN_FILE` and only
ever answers `can_resolve=TRUE` when its own toolchain is in effect, so the
default `apt`/`auto` builds never accidentally trigger conan or vcpkg.

- **Conan** (`deps_conan_*`): active when `CMAKE_TOOLCHAIN_FILE` is a
  `conan_toolchain.cmake`. Flow: `conan install . -of=<dir> -b=missing` produces
  `<dir>/conan_toolchain.cmake` + CMakeDeps configs; configure with that
  toolchain; `deps_conan_can_resolve` checks `<dir>/<CONAN_FIND>Config.cmake`
  exists; `deps_conan_resolve` does `find_package(<CONAN_FIND> CONFIG)` and links
  `CONAN_TARGETS`. Note the Conan recipe name (`CONAN_RECIPE`, e.g. `gtest`) and
  the CMake package name (`CONAN_FIND`, e.g. `GTest`) can differ.
- **vcpkg** (`deps_vcpkg_*`): active when `CMAKE_TOOLCHAIN_FILE` is `vcpkg.cmake`.
  In manifest mode vcpkg builds/installs every port in `vcpkg.json` at the start
  of configure, so `deps_vcpkg_can_resolve` simply runs
  `find_package(<VCPKG_FIND> CONFIG QUIET)`; `deps_vcpkg_resolve` re-runs it
  REQUIRED and links `VCPKG_TARGETS`.

Packages absent from a registry fall through to the next chain link and finally
to `PROVIDED` (download / vendored / `FETCHCONTENT`). In practice on Linux:

| package | conan | vcpkg | default (apt) |
|---|---|---|---|
| GTest | works (binary or source) | works (source) | FetchContent |
| OpenCV | **version conflict** (opencv/4.8.1 → ffmpeg/4.4 not on conancenter) | works (20-40 min source) | apt |
| FFmpeg | **version conflict** (same opencv chain) | works (15-20 min source) | apt |
| SDL2 | needs 20+ X11 `-dev` packages | upstream vcpkg regression | apt |
| ONNX Runtime | wrong version (1.23+, project pins 1.21) | wrong version | provided-download |
| TensorRT | absent | absent | provided-download |

ConanCenter serves prebuilt Windows binaries for the heavy frameworks; Linux gets
source builds. vcpkg does not serve prebuilt binaries. The catalogs carry
`VCPKG_FIND`/`VCPKG_TARGETS` for OpenCV and FFmpeg (vcpkg handles their
dependency trees correctly) but the default builds keep them on apt for speed.

## 10. Backwards compatibility

The refactor preserves byte-for-byte runtime behaviour:

- identical ORT 1.21.0 URL, `.so.1.21.0` filename, and post-build copy next to
  `inference_app` (now expressed via `RUNTIME_LIBS`);
- identical TensorRT 10.13.3.9 URL, CUDA fallback probing, and `BUILD/INSTALL
  _RPATH` (now via `RPATH_DIRS`);
- identical `find_package(OpenCV COMPONENTS ...)` / `pkg_check_modules(...)` for
  the media backend, gated by the same `USE_OPENCV` switch;
- existing cache vars (`ONNXRUNTIME_ROOTDIR`, `TENSORRT_ROOTDIR`, ...) keep their
  names so external tooling and the Dockerfile symlink workaround keep working.

The default `DEPS_MODE=apt` reproduces today's build exactly; switching modes is
opt-in and additive.

## 11. End-to-end verification

Three full project builds with GTest as the cross-manager test dependency
(media + inference deps stay on apt + provided-download):

| build | source | GTest resolved by | result |
|---|---|---|---|
| default (regression) | local + CI | `PROVIDED` FetchContent | build + all tests pass |
| conan e2e | local | conan cache `~/.conan2/p/b/gtest*/lib/libgtest.a` | build + all tests pass |
| vcpkg e2e | CI (PR #1) | `vcpkg_installed/.../lib/libgtest.a` | build + all tests pass |

CI coverage: `ci.yml` (default apt: build, unit tests, benchmarks, ASan/TSan/valgrind)
+ `lint.yml` (format, clang-tidy, cppcheck, -Werror). The `deps-modes.yml`
workflow (conan/vcpkg matrix) is `workflow_dispatch`-only by design — the default
path exhaustively tests the facade on every PR; conan/vcpkg are for local or
opt-in CI runs.

### Reproducing

```bash
# Conan: pre-install, then configure with the conan toolchain
conan install . -pr <profile> -of=build-conan/conan -b=missing
cmake -S . -B build-conan -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_TOOLCHAIN_FILE=build-conan/conan/conan_toolchain.cmake \
      -DDEPS_MODE=auto
cmake --build build-conan && ctest --test-dir build-conan --output-on-failure

# vcpkg: manifest mode installs ports during configure
cmake -S . -B build-vcpkg -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_TOOLCHAIN_FILE=<vcpkg>/scripts/buildsystems/vcpkg.cmake \
      -DDEPS_MODE=auto
cmake --build build-vcpkg && ctest --test-dir build-vcpkg --output-on-failure
```

`conanfile.txt` and `vcpkg.json` (repo root) carry each manager's dependency
list; add a `<pkg>` line there plus a catalog `CONAN_FIND`/`VCPKG_FIND`+
`CONAN_TARGETS`/`VCPKG_TARGETS` pair to route additional dependencies through the
manager once prebuilt binaries are available for the toolchain.
