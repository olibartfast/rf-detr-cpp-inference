# Requirements: GPU Parity Fixtures

Roadmap [Phase 2](../../roadmap.md#phase-2--gpu-parity-fixtures). Builds the measurement apparatus
that Phases 3 and 4 are gated on. No production code changes — this phase adds test data, one unit
test file, and one benchmark file.

## Scope

### In

| Deliverable | Path |
|-------------|------|
| Golden CPU fixtures | `tests/data/gpu_parity/` (does not exist yet) |
| Preprocess parity unit test | `tests/unit/test_gpu_parity.cpp` |
| Per-stage GPU benchmark | `tests/benchmark/bench_gpu_pipeline.cpp` |
| Build registration for both | `CMakeLists.txt:386-397` (unit tests), `:418-426` (benchmarks) |

### Fixture set

Four cases, chosen so that each can fail differently:

| Fixture | Shape | What it catches |
|---------|-------|-----------------|
| `small` | smaller than `res` on both axes | upscale path; DALI's default antialias behaviour does not apply here |
| `wide` | wide aspect, larger than `res` | independent `scale_x` — a letterbox regression shows up as a horizontal shift |
| `tall` | tall aspect, larger than `res` | independent `scale_y`; the downscale path where `antialias=False` matters most |
| `dense` | synthetic, engineered to yield **> 100** above-threshold detections | a truncating or capped postprocessor. Stock photos yield 10–50 detections, below any cap, so the natural images cannot distinguish one |

Each fixture stores: the source image, the CPU-produced preprocessed tensor (`float[1,3,res,res]`),
and the CPU-produced final detections (and masks, where segmentation applies).

### Out

- No `integration_test_gpu_parity.cpp` — that is Phase 4, and it consumes these fixtures.
- No CI job, no `CMakePresets.json` preset, no `Dockerfile` change — that is Phase 3.
- No production code under `src/`. If a parity gap is found while building the fixtures, record it
  and raise it; do not fix it inside this phase.
- No new dependency. GoogleTest and Google Benchmark are already resolved
  ([tech-stack.md](../../tech-stack.md)).

## Decisions

- **Fixtures are file-backed, the existing GPU test stays synthetic.**
  `tests/unit/test_gpu_postprocess.cpp` already covers CPU-vs-GPU *segmentation postprocess* parity
  from synthetic in-memory tensors, dense case included
  (`GpuSegPostprocessParity.MatchesCpuOnDenseDetections`). `test_gpu_parity.cpp` must not re-do
  that. Its job is the half that has no coverage: **DALI preprocess vs. CPU preprocess** on real
  image geometry, plus regression against a stored golden tensor rather than against a
  simultaneously-computed one.

- **No model file, in either test.** [mission.md](../../mission.md) makes "unit tests need no model
  file" an architectural commitment. Preprocessed-tensor fixtures need only an image; detection and
  mask fixtures are produced from **synthetic model-output tensors** served through
  `MockBackend`/`MockDeviceBackend` (`tests/unit/mock_backend.hpp`), exactly as
  `test_gpu_postprocess.cpp` already does. The synthetic outputs are stored alongside the expected
  results so a fixture is reproducible without a `.onnx` or `.engine`.

- **Tolerance gate, never an equality gate, for anything that resamples.** DALI's resize will not
  bit-match the CPU 4-tap bilinear — [gpu-pipeline.md](../../gpu-pipeline.md) correctness rules.
  Equality is asserted only for the CPU path against its own stored fixture.

- **`GTEST_SKIP()`, never fail, without a device.** Reuse the `SKIP_WITHOUT_GPU()` macro pattern
  from `tests/unit/test_gpu_postprocess.cpp:28`; the skip must be visible in test output.

- **Registration mirrors the existing conditional.** `test_gpu_parity.cpp` is appended to
  `UNIT_TEST_SOURCES` under the same `if(USE_CUDA_POSTPROCESS)` guard at `CMakeLists.txt:387`.
  `bench_gpu_pipeline.cpp` joins the `benchmarks` target, which today compiles
  `bench_preprocessing.cpp` alone.

- **Fixture size is bounded.** Images stay small enough to commit; the stored `float` tensors are
  the bulk (`3 × res × res × 4` bytes ≈ 2.8 MB at res 432). Store tensors for **one resolution
  only** — 432, which has a checked-in `.dali` pipeline (`data/dali/`) — and derive the rest at
  runtime.

## Context

### Existing patterns to follow

- `tests/unit/test_gpu_postprocess.cpp` — device-available skip macro, `MockDeviceBackend`, how a
  test feeds identical bytes to both postprocessors.
- `tests/unit/mock_backend.hpp` — the injected-backend constructor of `RFDETRInference`.
- `tests/benchmark/bench_preprocessing.cpp` — current benchmark idiom; it covers only `sigmoid`,
  `cxcywh_to_xyxy`, and `normalize_image`, i.e. nothing GPU and nothing per-stage.
- `src/gpu/dali_preprocessor.*` and `src/media.cpp:212-213` (the CPU stretch) — the two
  implementations being compared.

### Constraints carried in from the constitution

- [gpu-pipeline.md](../../gpu-pipeline.md) model contract rules 1, 2, 6 and 7 are what the fixtures
  exist to detect violations of: no letterbox, ImageNet normalisation, global top-k for
  segmentation, class-index offset.
- Only resolutions **432** and **576** have checked-in `.dali` pipelines
  ([tech-stack.md](../../tech-stack.md)); anything else needs
  `./scripts/generate_dali_pipelines.sh <res>` and a GPU-enabled Docker.
- CI has no GPU. Nothing added here may fail on a CI runner.

### Open question to resolve during implementation

The dense fixture needs > 100 above-threshold detections. Synthesising the model-output tensor
directly is the reliable route (the existing dense postprocess test already does this); a natural
image that happens to produce 100+ detections is not reproducible across model versions. Confirm
the synthetic route also exercises the DALI preprocess path meaningfully — if it cannot (a
synthetic tensor bypasses preprocessing entirely), the dense fixture covers postprocess only and
the requirement is met by pairing it with the three natural images for preprocess.
