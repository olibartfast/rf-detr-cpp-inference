# Plan: GPU Parity Fixtures

Four task groups. Groups 1 and 2 are independent of a GPU — they run on any machine. Groups 3 and 4
need a CUDA device to *execute*, but must **compile** without one.

---

## Group 1: Fixture data and generator

1. Create `tests/data/gpu_parity/` with a `README.md` stating how each fixture was produced and at
   which resolution, so a stale fixture can be regenerated rather than guessed at.

2. Add the three natural images (`small`, `wide`, `tall`) — small enough to commit, each with the
   geometry described in [requirements.md](requirements.md#fixture-set).

3. Add a generator under `scripts/` (or a `--generate` flag on the test binary — pick one and say
   which in the fixture README) that writes, per fixture:
   - `<name>.preprocessed.bin` — the CPU `float[1,3,432,432]` tensor from `preprocess_bgr_image`
   - `<name>.outputs.bin` — the synthetic model-output tensors fed through `MockBackend`
   - `<name>.expected.json` — CPU detections (class, score, xyxy) and, for segmentation fixtures,
     the mask digest plus the raw mask bytes
   - The generator must be deterministic: fixed RNG seed for the synthetic tensors.

4. Add the `dense` fixture: a synthetic output tensor engineered to yield **> 100** above-threshold
   detections after the class-index offset. Follow the construction already used by
   `GpuSegPostprocessParity.MatchesCpuOnDenseDetections` in `tests/unit/test_gpu_postprocess.cpp`.

---

## Group 2: CPU determinism check

5. Add a test (in `test_gpu_parity.cpp`, no device required, **not** skipped) that loads each
   fixture, re-runs the CPU path, and asserts the result is **bit-identical** to the stored
   fixture — twice in a row within the same process.
   - This is the only equality assertion in the phase. If the CPU path is not bit-reproducible,
     every tolerance downstream is meaningless, so this test gates the rest.

---

## Group 3: `tests/unit/test_gpu_parity.cpp`

6. Create the file with the header comment convention used by `test_gpu_postprocess.cpp`, and reuse
   its `SKIP_WITHOUT_GPU()` macro (`tests/unit/test_gpu_postprocess.cpp:28`) — lift it into a small
   shared header under `tests/unit/` rather than copying it, since two files now need it.

7. Preprocess parity cases, one per natural fixture: run `src/gpu/dali_preprocessor` at res 432,
   compare against the stored CPU tensor with `max |Δ| ≤ 2e-2`. Report the actual max delta in the
   failure message — the number is wanted even when it passes.

8. A no-letterbox assertion: with the `wide` and `tall` fixtures, confirm the GPU tensor has no
   constant-valued border rows/columns that the CPU tensor lacks. This is the cheap direct test for
   [gpu-pipeline.md](../../gpu-pipeline.md) model-contract rule 1.

9. End-to-end fixture regression: feed `<name>.outputs.bin` through `MockDeviceBackend` and assert
   the detections match `<name>.expected.json` on the *set* (class, count, score within `1e-3`, box
   centres within 1 px) — never on order, per the score-sort-ties rule.

10. Register the file at `CMakeLists.txt:387`, inside the existing `if(USE_CUDA_POSTPROCESS)` guard
    that already appends `test_gpu_postprocess.cpp`.

---

## Group 4: `tests/benchmark/bench_gpu_pipeline.cpp`

11. Create the benchmark timing **four stages separately** — preprocess, H2D + infer, D2H,
    postprocess — for a still image and for a video run, following the idiom in
    `tests/benchmark/bench_preprocessing.cpp`.

12. Add it to the `benchmarks` target at `CMakeLists.txt:424` (currently a single-source
    `add_executable`; make it a source list). Guard the GPU sources so `-DBENCHMARKS=ON` still
    builds without `USE_CUDA_POSTPROCESS`.

13. Record the **CPU baseline** numbers — 560×560 detection, and segmentation with a 1080p source —
    into the fixture README. Every Phase 4 number is measured against these, so they must be
    captured before any GPU comparison is run.
