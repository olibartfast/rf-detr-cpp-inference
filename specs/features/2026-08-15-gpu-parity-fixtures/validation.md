# Validation: GPU Parity Fixtures

Validated on 2026-09-08, branch feature/phase1-known-issues (extended), on an RTX 3060 laptop GPU.

## Automated — no GPU

- [x] Default build untouched: `cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build build --parallel`
- [x] `ctest --test-dir build --output-on-failure -R UnitTests` passes (CPU determinism test is
      compiled into the GPU build only, per the registration decision below)
- [x] `-DWERROR=ON` clean — new test and benchmark sources included (verified via `build-gpu` /
      `build-gpu-bench`)
- [x] Format check clean
- [x] Cppcheck exits 0
- [x] CPU determinism: `GpuParityCpuDeterminism.FixturesReproduceBitIdentically` — each fixture
      re-derived twice in one process, bit-identical to the stored bytes
- [x] `-DBENCHMARKS=ON` builds with `USE_CUDA_POSTPROCESS=OFF` (GPU benchmark guarded)

## Automated — with a CUDA device

- [x] Preprocess parity: max `|Δ|` = **0.0088** (resize + normalise, frame path) on `small`,
      `wide` and `tall` at res 432, printed per fixture — within `2e-2`.
  - The *encoded* path (nvJPEG decode + resize) is 0.050–0.069; the delta is entirely the
    JPEG decode (nvJPEG vs stb), which the frame path isolates out. Recorded in the CHANGELOG
    and the fixture README as the known "DALI preprocessing parity remains open" item.
- [x] No-letterbox assertion passes on `wide` and `tall`
- [x] Detection-set regression passes on all four fixtures (set, not order)
- [x] `dense` fixture yields 200 (> 100) above-threshold detections and both paths agree

## Automated — compile-without-device

- [x] With GPU options on and no device, every GPU-dependent case `SKIP`s (the `SKIP_WITHOUT_GPU`
      macro is shared from `tests/unit/gpu_test_utils.hpp`, not copied). Verified by the
      `GTEST_SKIP` path (CI compiles the same sources).

## Manual

- [x] `tests/data/gpu_parity/README.md` records geometry, resolution, and seeds for every fixture
- [x] Total fixture size (~8.6 MB, tensors at res 432 only) justified in that README
- [x] CPU baseline numbers recorded (preprocess, transfer, seg postprocess 1080p); "infer" is
      engine-bound and deferred to the Phase 4 exit gate
- [x] No production file under `src/` modified; the preprocess decode gap is written up, not fixed
- [x] `test_gpu_parity.cpp` does not duplicate `test_gpu_postprocess.cpp`; `SKIP_WITHOUT_GPU` and
      `MockDeviceBackend` are lifted into `tests/unit/gpu_test_utils.hpp`

## Deviations

- `test_gpu_parity.cpp` is registered under `if(USE_DALI OR USE_CUDA_POSTPROCESS)` (not the
  spec's `if(USE_CUDA_POSTPROCESS)`), because the preprocess-parity cases need DALI while the
  end-to-end regression needs CUDA postprocess; each group is `#ifdef`-guarded internally.
- The generator is a dedicated `gpu_parity_gen` executable (the spec offered "a `--generate` flag
  or a script — pick one"); fixtures are committed and the generator is committed alongside.
- The expected-results file is `name.expected.txt` (line-based), not `.json` — no JSON library
  is a project dependency and the test reads it back with plain stream extraction.
