# Development

Formatting, static analysis, sanitizers, Valgrind, tests, and benchmarks — everything for working on the code rather than just building it.

> Part of the [RF-DETR C++ Inference](../README.md) documentation.

---

## Format Code (Optional)
If you have `clang-format-18` installed, you can check and auto-format all source files:

```bash
# Check for formatting issues (no changes made):
find src tests -name '*.cpp' -o -name '*.hpp' | xargs clang-format-18 --dry-run --Werror

# Auto-format in place:
find src tests -name '*.cpp' -o -name '*.hpp' | xargs clang-format-18 -i
```

## Static Analysis (Optional)
If you have `clang-tidy-18` installed, you can run static analysis using the compile commands database:

```bash
# Generate compile_commands.json first:
cmake -S . -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

# Run clang-tidy on project sources:
find src -name '*.cpp' | xargs clang-tidy-18 -p build
```

## Cppcheck (Optional)
If you have `cppcheck` installed, you can run additional static analysis:

```bash
# Install cppcheck:
sudo apt-get install -y cppcheck

# Run manually:
cppcheck --enable=all --std=c++20 \
  --suppress=missingIncludeSystem \
  --suppress=unmatchedSuppression \
  --error-exitcode=1 \
  -I src src/
```

This is also run automatically on every commit via pre-commit (see [Pre-commit](#pre-commit-optional)).

## Sanitizers (Optional)
Build with **AddressSanitizer + UndefinedBehaviorSanitizer** via `-DSANITIZERS=ON`, with stricter **UndefinedBehaviorSanitizer** checks via `-DSTRICT_UBSAN=ON`, or with **ThreadSanitizer** (data-race detection) via `-DTHREAD_SANITIZER=ON`. These modes are mutually exclusive — enable only one per build directory. Use separate build directories to keep them independent:

These options use the sanitizer instrumentation and runtime libraries provided
by the active C++ compiler toolchain (Clang or GCC). The project does not fetch,
vendor, or depend on the archived `google/sanitizers` repository.

```bash
# ASan + UBSan
cmake -S . -B build-san -DCMAKE_BUILD_TYPE=Debug -DSANITIZERS=ON
cmake --build build-san --parallel
./build-san/unit_tests

# Strict UBSan (extra bounds and vptr checks; Clang also enables implicit-conversion)
cmake -S . -B build-strict-ubsan -DCMAKE_BUILD_TYPE=Debug -DSTRICT_UBSAN=ON
cmake --build build-strict-ubsan --parallel
./build-strict-ubsan/unit_tests

# ThreadSanitizer (data races)
cmake -S . -B build-tsan -DCMAKE_BUILD_TYPE=Debug -DTHREAD_SANITIZER=ON
cmake --build build-tsan --parallel
TSAN_OPTIONS="halt_on_error=1" ./build-tsan/unit_tests
```

Sanitizers catch memory errors, use-after-free, undefined behaviour, integer overflow (ASan+UBSan), stricter bounds/vptr issues, plus implicit-conversion issues on Clang (strict UBSan), and data races (TSan) at runtime. ASan+UBSan and TSan run in CI; strict UBSan is opt-in for local diagnosis because it can be noisier.

## Valgrind / Profiling (Optional)
A plain **Debug build without sanitizers** is required (ASan/TSan conflict with Valgrind). When Valgrind is detected, CMake generates `memcheck`, `callgrind`, and `massif` targets:

```bash
cmake -S . -B build-valg -DCMAKE_BUILD_TYPE=Debug

cmake --build build-valg --target memcheck      # memory errors + leaks (run by CI)
cmake --build build-valg --target callgrind     # CPU/cache profile -> callgrind.out.<pid>
cmake --build build-valg --target massif        # heap profile       -> massif.out.<pid>

# Read the profiles:
callgrind_annotate build-valg/callgrind.out.<pid>
ms_print build-valg/massif.out.<pid>
```

The profilers target `benchmarks` (if built with `-DBENCHMARKS=ON`, recommended for self-contained workloads) or `inference_app`; pass extra args with `-DVALGRIND_PROFILE_ARGS="model.onnx image.jpg labels.txt"`. A lower-overhead alternative is `perf record ./build/benchmarks && perf report`. An optional `valgrind.supp` at the repo root is picked up automatically if present.

## Pre-commit (Optional)
[pre-commit](https://pre-commit.com/) runs `clang-format` and `cppcheck` automatically on every commit:

```bash
pip install pre-commit
pre-commit install           # install the git hook
pre-commit run --all-files   # run manually on all files
```

## Strict Compilation (Optional)
To treat all compiler warnings as errors (as CI does), pass `-DWERROR=ON`:

```bash
cmake -S . -B build -DWERROR=ON
cmake --build build
```

## Testing


## Unit Tests
Unit tests use Google Test and run without any model files:

```bash
cmake -S . -B build
cmake --build build --parallel
ctest --test-dir build --output-on-failure -R UnitTests
```

Integration tests need a real model **in the format the compiled-in backend accepts**. They probe
`~/Downloads/`, `exports/`, and `output/` for `rfdetr-medium` (or legacy `inference_model`) with the
extensions below, or you can point `RFDETR_TEST_MODEL` at one directly:

| Build | Extensions probed | Example |
|-------|-------------------|---------|
| ONNX Runtime (default) | `.onnx` | `export RFDETR_TEST_MODEL=/path/to/rfdetr-medium.onnx` |
| TensorRT | `.engine`, `.trt`, then `.onnx` | `export RFDETR_TEST_MODEL=/path/to/rfdetr-medium.engine` |
| ExecuTorch | `.pte` | `export RFDETR_TEST_MODEL=/path/to/rfdetr-medium.pte` |

```bash
export RFDETR_TEST_MODEL=/path/to/rfdetr-medium.onnx
ctest --test-dir build --output-on-failure -R IntegrationTests
```

TensorRT keeps `.onnx` as a lower-priority candidate so the ONNX-to-engine conversion path stays
covered; a prebuilt engine is preferred because it loads directly. Keypoint tests use the same
scheme with `rfdetr-keypoint` / `rfdetr-keypoint-preview` and `RFDETR_KEYPOINT_MODEL`.

## Manual Cross-Backend Checks
CI compiles the TensorRT backend but has no GPU to run it on, and does not build ExecuTorch at
all, so backend agreement is verified by hand.
[backend-parity-segmentation-video.md](backend-parity-segmentation-video.md) records a
segmentation video run across all three backends — commands, results, and the parity comparison.

Without a matching model, integration tests that need inference are skipped, and the skip message
names the format the build expects.

In a `-DUSE_CUDA_POSTPROCESS=ON` build, the unit tests additionally include a
CPU-versus-GPU parity gate for the segmentation postprocessor
(`tests/unit/test_gpu_postprocess.cpp`). It needs no model and no DALI —
synthetic tensors are served from both host and device memory through a mock
backend — and every case skips (rather than fails) when no CUDA device is
present, so CI can compile the GPU targets on runners without a GPU.

## Benchmarks
Benchmarks use [Google Benchmark](https://github.com/google/benchmark) to measure preprocessing performance. Enable with `-DBENCHMARKS=ON`:

```bash
cmake -S . -B build -DBENCHMARKS=ON
cmake --build build --target benchmarks --parallel
./build/benchmarks
```

## Code Quality Tools
| Tool | Purpose | How to run |
|------|---------|------------|
| `clang-format-18` | Code formatting | `find src tests -name '*.cpp' -o -name '*.hpp' \| xargs clang-format-18 -i` |
| `clang-tidy-18` | Static analysis (AST-based) | `find src -name '*.cpp' \| xargs clang-tidy-18 -p build` |
| `cppcheck` | Static analysis (flow-based) | `cppcheck --enable=all --std=c++20 -I src src/` |
| AddressSanitizer(ASan) + UndefinedBehaviorSanitizer(UBSan) | Runtime memory/UB detection | `-DSANITIZERS=ON` at configure time |
| Strict UndefinedBehaviorSanitizer (UBSan) | Extra bounds and vptr checks; Clang also enables implicit-conversion | `-DSTRICT_UBSAN=ON` at configure time |
| ThreadSanitizer (TSan) | Runtime data-race detection | `-DTHREAD_SANITIZER=ON` at configure time |
| Valgrind (memcheck) | Memory errors + leak detection | `cmake --build build-valg --target memcheck` |
| Valgrind (callgrind/massif) | CPU/cache + heap profiling | `cmake --build build-valg --target callgrind` / `massif` |
| pre-commit | Automates format + cppcheck on commit | `pre-commit install` |
