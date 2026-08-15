# Validation: GPU Parity Fixtures

The merge gate for roadmap [Phase 2](../../roadmap.md#phase-2--gpu-parity-fixtures). Everything
below must hold before the phase is ticked.

## Automated — no GPU (must pass on CI and on any dev machine)

- [ ] Default build is untouched:
      `cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build build --parallel`
- [ ] `ctest --test-dir build --output-on-failure -R UnitTests` passes
- [ ] `-DWERROR=ON` build is clean — new test and benchmark sources included
- [ ] Format check clean:
      `find src tests -name '*.cpp' -o -name '*.hpp' | xargs clang-format-18 --dry-run --Werror`
- [ ] Cppcheck exits 0 (see [AGENTS.md](../../../AGENTS.md) for the exact invocation)
- [ ] **CPU determinism**: the fixture-reproduction test passes — each fixture re-derived from its
      source is bit-identical to the stored bytes, twice consecutively in one process
- [ ] `-DBENCHMARKS=ON` builds with `USE_CUDA_POSTPROCESS=OFF` (the GPU benchmark must be guarded,
      not unconditionally compiled)

## Automated — with a CUDA device

Built with `-DUSE_ONNX_RUNTIME=OFF -DUSE_TENSORRT=ON -DUSE_GPU_PIPELINE=ON -DDALI_ROOT=$HOME/dependencies/dali`.

- [ ] Preprocess parity: `max |Δ| ≤ 2e-2` on `small`, `wide` and `tall` at res 432 — and the actual
      max delta is printed for each, pass or fail
- [ ] No-letterbox assertion passes on `wide` and `tall`
- [ ] Detection-set regression passes on all four fixtures: same class set and count, scores within
      `1e-3`, box centres within 1 px, asserted on the **set** and not the order
- [ ] `dense` fixture yields **> 100** above-threshold detections and both paths agree on the count
      — a fixture that yields ≤ 100 has not met the requirement, whatever else passes

## Automated — compile-without-device

- [ ] With `USE_CUDA_POSTPROCESS=ON` on a machine with **no** GPU, `unit_tests` builds and every
      GPU-dependent case reports `SKIPPED`, not `FAILED`, and the skip line is **visible** in
      `--output-on-failure` output

## Manual

- [ ] `tests/data/gpu_parity/README.md` states, for every fixture, how it was produced, at which
      resolution, and with which RNG seed — a fixture nobody can regenerate is a liability
- [ ] Total fixture size is justified in that README; tensors are stored for res 432 only
- [ ] CPU baseline benchmark numbers recorded in that README: 560×560 detection, and segmentation
      with a 1080p source, four stages each
- [ ] No production file under `src/` was modified. If a parity gap was found, it is written up in
      CHANGELOG "Known Issues" or as a roadmap item — not silently fixed here
- [ ] `test_gpu_parity.cpp` does not duplicate coverage already in `test_gpu_postprocess.cpp`; the
      `SKIP_WITHOUT_GPU()` macro is shared, not copied

## Definition of done

All boxes above ticked; `CHANGELOG.md` updated under `[Unreleased]` in the house style with a
per-file table; the three Phase 2 items in [roadmap.md](../../roadmap.md) ticked `[x]` and the
phase heading marked `(Complete)`; branch merged into `develop` and deleted.

Phase 3 may not start before this is done — its CI job runs these tests, and Phase 4's tolerance
gate is measured against these fixtures and these baseline numbers.
