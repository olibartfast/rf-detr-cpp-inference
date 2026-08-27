# Roadmap

Work this project has committed to, as a phased queue. Phases are ordered so each one leaves the tree building and the CPU path untouched. Release history lives in [CHANGELOG.md](../CHANGELOG.md) and is not repeated here; GPU design constraints live in [gpu-pipeline.md](gpu-pipeline.md).

**The next phase is the first one below whose items are all `[ ]`.** Tick an item only when it is merged and verified. A phase that is fully ticked gets `(Complete)` in its heading.

## Status

- Last tag **v0.4.0** (2026-08-04). Flow is git-flow: `develop` → `release/vX.Y.Z` → `master`, merged back.
- **v0.5.0 is staged on `develop`** — the rfdetr 1.9.1 and 1.9.2 alignments and the whole GPU pipeline are unreleased. Phase 5 cuts it, gated on Phases 1–4.
- GPU pipeline: preprocessing and segmentation postprocessing **work end to end**; what remains is the test, build, and CI scaffolding around them (Phases 2–4).

Before starting any phase, run the [`feature-spec`](../.claude/skills/feature-spec/SKILL.md) workflow — every phase here qualifies as multi-session work, so each one gets a spec directory under [`features/`](features/) before code is written.

---

## Phase 1 — Close the known issues

From the Known Issues table in [CHANGELOG.md](../CHANGELOG.md). Independent of each other and of everything below.

- [ ] Make `src/backends/tensorrt_backend.cpp` compile under `-DWERROR=ON`
  - The gpu-verify gate builds with `-DWERROR=ON`, so this fails gate step 1 and skips every later step
  - Mechanical: `-Wsign-conversion` at 148, 161, 257, 267, 271; `-Wunused-parameter` on `input_shape` at 171
  - A decision, not mechanical: `kEXPLICIT_BATCH` (180) is a no-op in TensorRT 10, and `platformHasFastFp16()` (223) / `BuilderFlag::kFP16` (224) are superseded by strongly-typed networks
  - Touches a CI-unexecutable path, so it needs a spec directory per [AGENTS.md](../AGENTS.md)
- [ ] Support the active-first keypoint schema (`rfdetr` 1.8.2+)
  - Upstream 1.8.2 changed the default `num_keypoints_per_class` from background-first `[0, 17]` to active-first `[17]` ([#1160](https://github.com/roboflow/rf-detr/pull/1160)); `Config::keypoint_counts` still defaults to `{0, 17}` and has no CLI override
  - `deploy/requirements.txt` pins 1.9.4, so the documented export path produces a schema the default build cannot decode — expected to throw `Keypoint tensor channels (17) not divisible by number of keypoint classes (2)`
  - **Verify against a real 1.8.2+ keypoint export first.** The failure is derived from the release notes and the code, not observed; the `labels` column count under the new schema is unconfirmed and decides whether `background_class_id` also needs to change
  - Decide between a `--keypoint-counts` flag and auto-detecting the schema from the tensor shape. Either way, pre-1.8.2 exports must keep working
  - Also re-export keypoint models with 1.8.1+: [#1135](https://github.com/roboflow/rf-detr/pull/1135) fixed eval-mode query routing, which export traces
- [ ] Add segmentation export to `deploy/export_executorch.py`
  - `--model_type` offers only detection classes; the script instantiates `RFDETRNano`…`RFDETR2XLarge`, never `RFDETRSeg*`
  - Not an upstream or runtime limitation: `rfdetr` exports `RFDETRSegMedium` to `.pte` without error, `ExecuTorchBackend::validate_output_order()` inspects only outputs 0 and 1 so a third `masks` output passes, and `postprocess_segmentation_outputs()` addresses outputs positionally
  - Segmentation `.pte` files must be hand-exported today — see [docs/backend-parity-segmentation-video.md](../docs/backend-parity-segmentation-video.md)
- [ ] Add an output-path flag to `src/main.cpp`
  - Video output is hardcoded to `output_video.mp4` in the working directory, so comparing backends requires running each from its own directory
  - Image output is likewise hardcoded to `output_image.jpg`
- [ ] Verify and close the `.gitignore` item
  - **Partly stale**: `*.pte` is now ignored (`.gitignore:7`), and `output_video.mp4` (`:13`) has no leading slash so it already matches at any depth
  - What remains: once the output flag above lands, arbitrary output filenames are no longer covered. Decide the ignore pattern with that flag, then strike the item from the CHANGELOG

---

## Phase 2 — GPU parity fixtures

The parity gate the GPU work was supposed to be measured against was never built. Everything in Phases 3 and 4 depends on it.

Spec: [`features/2026-08-15-gpu-parity-fixtures/`](features/2026-08-15-gpu-parity-fixtures/) — acceptance criteria and tolerances live in its `validation.md`.

**Evidence from the first real gate run (2026-08-26).** The four combinations split by *preprocessing*, not postprocessing: `cpu-cpu` and `cpupre-gpupost` agree to the last digit, as do `gpupre-cpupost` and `gpu-gpu` — CUDA postprocessing is bit-identical to CPU. DALI preprocessing is not: max score delta 0.0163 on one image, 16× the 1e-3 score tolerance. One image and final scores rather than the preprocessed tensor, so not conclusive — but write these fixtures expecting to find a discrepancy, not to confirm its absence. Details in [CHANGELOG.md](../CHANGELOG.md).

- [ ] Add golden CPU fixtures under `tests/data/gpu_parity/` (directory does not exist)
  - A small, a wide, and a tall image; save the CPU-produced preprocessed tensor and the final detections/masks with explicit tolerances
  - Add a **dense synthetic fixture** engineered to produce more than 100 above-threshold detections. Stock photos yield 10–50 detections, below any cap, so they cannot distinguish a truncating postprocessor from a correct one
- [ ] Add `tests/unit/test_gpu_parity.cpp`
  - Follow `tests/unit/test_gpu_postprocess.cpp` for the `GTEST_SKIP()`-without-a-device pattern
- [ ] Add `tests/benchmark/bench_gpu_pipeline.cpp` and register it in `CMakeLists.txt`
  - `tests/benchmark/bench_preprocessing.cpp` currently covers only `sigmoid`, `cxcywh_to_xyxy`, and `normalize_image`
  - Time four stages separately — preprocess, H2D+infer, D2H, postprocess — for a still image and for a video run

---

## Phase 3 — GPU build and CI integration

The one incomplete part of the GPU pipeline's build work; dependency declarations and CMake options are already done.

- [ ] Add a `gpu-pipeline` configure preset to `CMakePresets.json` (none of the five existing presets covers TensorRT, ExecuTorch, OpenCV, or GPU)
- [ ] Add DALI staging and the GPU options to `Dockerfile` — it contains **zero** DALI references today
  - Base the stage on `nvcr.io/nvidia/tensorrt:<tag>` with the DALI libraries staged in
- [ ] Add a compile-only GPU job to CI
  - Compile the GPU targets and skip execution, matching the posture already taken for TensorRT. `nvcc` is available on runners; a GPU is not
  - **Verify:** CI green with GPU targets compiled and GPU tests skipped, and the skip **visible** in the test output rather than silent

---

## Phase 4 — GPU parity gate and benchmarks

- [ ] Add `tests/integration/integration_test_gpu_parity.cpp`
  - Run every fixture through all four combinations: CPU/CPU, GPU-pre/CPU-post, CPU-pre/GPU-post, GPU/GPU
  - Assert: preprocessed tensor `max |Δ| ≤ 2e-2`; detection sets match on class and count with scores within `1e-3`; box centres within 1 px; mask IoU ≥ 0.999
  - **Verify:** all four pass on the dense fixture as well as the natural images
- [ ] Extend the benchmark to the same four combinations, per-stage, still image and video
  - Expect a large improvement in segmentation postprocess — the mask resize is the whole point
  - Expect **little or no end-to-end gain from preprocessing on single still images**: at 560×560 the CPU preprocess is ~1–2 ms and DALI adds its own launch overhead. The wins are the eliminated 3.7 MB H2D, the freed CPU in the video pipeline's preprocess stage, and headroom at higher resolutions. Record what the numbers actually say, including where they are flat
- [ ] Run the exit gate on real hardware — the [`gpu-verify`](../.claude/skills/gpu-verify/SKILL.md) workflow:
  1. All three tasks run with `--gpu-preprocess` inside the tolerances above
  2. Segmentation runs with `--gpu-postprocess` at mask IoU ≥ 0.999, including on the dense fixture
  3. A 1000-frame video run completes with no leak and no `compute-sanitizer` findings
  4. The default (ONNX Runtime, CPU) build and its results are bit-identical to today
  5. Benchmarks recorded, including the flat ones
  6. README and CHANGELOG updated per [AGENTS.md](../AGENTS.md)

  Items 4 and 6 are already satisfied; 1, 2, 3, and 5 are not.

---

## Phase 5 — Release v0.5.0

The [`release`](../.claude/skills/release/SKILL.md) workflow. Gated on Phases 1–4.

- [ ] Read `AGENTS.md`, `README.md`, and `CHANGELOG.md`, then verify the rfdetr release against upstream — the mandatory "Spec Sync" rule
- [ ] Move `[Unreleased]` to `[v0.5.0]`, sync `README.md` version statements against `CMakeLists.txt`, `CMakePresets.json`, `deploy/requirements.txt`, `Dockerfile`, and `docs/export.md`
- [ ] Resolve the version disagreement noted in [tech-stack.md](tech-stack.md#known-pin-duplications): `project()` declares none, `vcpkg.json` says `0.1.0`, the README badge says `0.4.0`
- [ ] Cut `release/v0.5.0`, merge to `master`, tag, merge back to `develop`

---

## Deferred

Not started, each for a recorded reason. Reopening one is a decision, not a task.

| Item | Why deferred |
|------|--------------|
| Batch size > 1 | Every tensor contract fixes batch 1, as the current code does |
| GPU postprocessing for detection | 300×91 sigmoids and a threshold — not a bottleneck. Moving it costs a kernel launch plus a D2H round trip for no gain |
| GPU postprocessing for keypoint | Cholesky-to-covariance maths and per-class keypoint mapping are branch-heavy; better on the CPU until profiling says otherwise |
| GPU rendering | Drawing stays on the CPU (`src/media.cpp`) |
| ONNX Runtime CUDA execution provider | The backend registers none; the GPU pipeline requires `USE_TENSORRT=ON` |
| Box-cropped masks | Masks are full-frame to match the CPU path. Cropping to the box and carrying the origin changes `rfdetr::media::Mask` and the drawing code |

**Standing obligation:** every upstream `rfdetr` release triggers an alignment pass — the [`rfdetr-alignment`](../.claude/skills/rfdetr-alignment/SKILL.md) workflow. It is event-driven, not a phase, and preempts the queue above. Read `AGENTS.md` first and verify against the upstream release notes before touching anything.

---

## Reference

- [gpu-pipeline.md](gpu-pipeline.md) — GPU architecture, the 8-rule model contract, packed output contract, correctness rules, risks. Read before modifying `src/gpu/`.
- [mission.md](mission.md) — architectural commitments and what is out of scope.
- [tech-stack.md](tech-stack.md) — pinned versions, CMake options, CI coverage.
