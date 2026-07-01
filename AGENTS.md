# AGENTS.md

## Backend Selection
- ONNX Runtime (default): 
  `cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build build --parallel`
- TensorRT:
  `cmake -S . -B build -G Ninja -DUSE_ONNX_RUNTIME=OFF -DUSE_TENSORRT=ON -DCMAKE_BUILD_TYPE=Release && cmake --build build --parallel`

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
- Configure: `cmake -S . -B build-san -DCMAKE_BUILD_TYPE=Debug -DSANITIZERS=ON`
- Build: `cmake --build build-san --parallel`
- Run unit tests: `./build-san/unit_tests`
- Run integration tests: `./build-san/integration_tests`

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