# AGENTS.md

## Backend Selection
- ONNX Runtime (default): 
  `cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build build --parallel`
- TensorRT:
  `cmake -S . -B build -G Ninja -DUSE_ONNX_RUNTIME=OFF -DUSE_TENSORRT=ON -DCMAKE_BUILD_TYPE=Release && cmake --build build --parallel`

## Dependency Resolution
- Default (`-DDEPS_MODE=apt`): system packages + pinned downloads — no extra tooling
- Conan/vcpkg: auto-activate via toolchain; see [docs/package-manager-architecture.md](docs/package-manager-architecture.md)
- `-DDEPS_DEBUG=ON` logs which handler resolved each dependency

## Code Quality
- Format check: `find src tests -name '*.cpp' -o -name '*.hpp' | xargs clang-format-18 --dry-run --Werror`
- Format apply: `find src tests -name '*.cpp' -o -name '*.hpp' | xargs clang-format-18 -i`
- Clang-tidy: 
  `cmake -S . -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON`
  `find src -name '*.cpp' | xargs clang-tidy-18 -p build`
- Cppcheck: `cppcheck --enable=all --std=c++20 --suppress=missingIncludeSystem --suppress=unmatchedSuppression --suppress=unusedFunction --error-exitcode=1 -I src src/`
- Strict warnings (CI): `-DWERROR=ON` at configure time

## Release Documentation Sync
- Mandatory for every release or dependency-facing patch: update `README.md` in the same change when code, build options, backend versions, Docker images, or Python export packages change.
- Verify README dependency/version statements against `CMakeLists.txt`, `CMakePresets.json`, `deploy/requirements.txt`, `Dockerfile*`, and `docs/export.md`.
- README must list current C++ library/runtime versions, CMake options, backend constraints, and pip packages used for export tooling.
- If a release intentionally needs no README change, say why in `CHANGELOG.md` or the PR/release notes.

## Testing
- Unit tests: `ctest --test-dir build --output-on-failure -R UnitTests`
- Integration tests: `ctest --test-dir build --output-on-failure -R IntegrationTests`
- All tests: `cmake --build build --target run_tests`
- Benchmarks (if enabled): `./build/benchmarks`

## Sanitizers
ASan+UBSan, strict UBSan, and TSan are mutually exclusive (pick one).

### AddressSanitizer + UndefinedBehaviorSanitizer
- Configure: `cmake -S . -B build-san -DCMAKE_BUILD_TYPE=Debug -DSANITIZERS=ON`
- Build: `cmake --build build-san --parallel`
- Run unit tests: `./build-san/unit_tests`
- Run integration tests: `./build-san/integration_tests`

### Strict UndefinedBehaviorSanitizer
- Configure: `cmake -S . -B build-strict-ubsan -DCMAKE_BUILD_TYPE=Debug -DSTRICT_UBSAN=ON`
- Build: `cmake --build build-strict-ubsan --parallel`
- Run unit tests: `./build-strict-ubsan/unit_tests`
- Run integration tests: `./build-strict-ubsan/integration_tests`

### ThreadSanitizer (data races)
- Configure: `cmake -S . -B build-tsan -DCMAKE_BUILD_TYPE=Debug -DTHREAD_SANITIZER=ON`
- Build: `cmake --build build-tsan --parallel`
- Run: `TSAN_OPTIONS="halt_on_error=1" ./build-tsan/unit_tests`

## Valgrind / Profiling
Requires a plain Debug build (no sanitizers — ASan/TSan conflict with Valgrind). The `memcheck`, `callgrind`, and `massif` CMake targets are auto-generated when Valgrind is found.
- Configure: `cmake -S . -B build-valg -DCMAKE_BUILD_TYPE=Debug`
- Memcheck (correctness — run by CI): `cmake --build build-valg --target memcheck`
- CPU/cache profile: `cmake --build build-valg --target callgrind` → read with `callgrind_annotate build-valg/callgrind.out.<pid>`
- Heap profile: `cmake --build build-valg --target massif` → read with `ms_print build-valg/massif.out.<pid>`
- Profilers run on `benchmarks` if built (`-DBENCHMARKS=ON`), else `inference_app` (pass args via `-DVALGRIND_PROFILE_ARGS="..."`).
- Lower-overhead alternative: `perf record ./build/benchmarks && perf report`.
- Optional suppressions file: `valgrind.supp` at repo root is picked up automatically if present.

## Pre-commit
- Install: `pip install pre-commit && pre-commit install`
- Run all: `pre-commit run --all-files`

## Usage
- Detection: `./build/inference_app model.onnx image.jpg coco-labels-91.txt`
- Segmentation: add `--segmentation`
- Video: replace image with video file (e.g., video.mp4)
- Display: add `--display`
- TensorRT engine: use .engine or .trt model file

## Notes
- Only one backend (ONNX Runtime or TensorRT) can be enabled at compile time.
- TensorRT requires manually installed CUDA toolkit.
- Data directory is auto-created by CMake.
- CI does not test TensorRT backend; test manually.