# Roadmap

Work this project has committed to, as a phased queue. Phases are ordered so each one leaves the tree building and the CPU path untouched. Release history lives in [CHANGELOG.md](../CHANGELOG.md) and is not repeated here; GPU design constraints live in [gpu-pipeline.md](gpu-pipeline.md).

**The next phase is the first one below whose items are all `[ ]`.** Tick an item only when it is merged and verified. A phase that is fully ticked gets `(Complete)` in its heading.

## Status

- Last tag **v0.4.0** (2026-08-04). Flow is git-flow: `develop` → `release/vX.Y.Z` → `master`, merged back.
- **v0.5.0 is staged on `develop`** — the rfdetr 1.9.1 and 1.9.2 alignments and the whole GPU pipeline are unreleased. Phase 5 cuts it, gated on Phases 1–4.
- GPU pipeline: preprocessing and segmentation postprocessing **work end to end**; what remains is the test, build, and CI scaffolding around them (Phases 2–4).

Before starting any phase, run the [`feature-spec`](../.claude/skills/feature-spec/SKILL.md) workflow — every phase here qualifies as multi-session work, so each one gets a spec directory under [`features/`](features/) before code is written.

---

## Phase 1 — Close the known issues (Complete)

From the Known Issues table in [CHANGELOG.md](../CHANGELOG.md). Independent of each other and of everything below.

- [x] Make `src/backends/tensorrt_backend.cpp` compile under `-DWERROR=ON`
  - Landed with the GPU pipeline work: `gpu-compile.yml` compiles the TensorRT backend under
    `-DWERROR=ON` in CI, and the file's sign-conversions, unused parameter, and TensorRT 10
    deprecations (`kEXPLICIT_BATCH`, `platformHasFastFp16`, `BuilderFlag::kFP16`) are resolved.
    No further change was needed; verification is CI, per the project's TensorRT posture.
- [x] Support the active-first keypoint schema (`rfdetr` 1.8.2+)
  - Added `--keypoint-counts`; verified against a real 1.10.1 export that the official
    `RFDETRKeypointPreview` checkpoint is **background-first** (`labels [1,100,2]`,
    `keypoints [1,100,34,8]`) and decodes with the default `{0, 17}` — the earlier "default
    export cannot decode" premise was wrong. The flag supports genuinely active-first `[17]`
    finetunes (`--keypoint-counts 17 --background-class-id none`).
  - Spec: [`features/2026-09-08-keypoint-schema/`](features/2026-09-08-keypoint-schema/)
- [x] Add segmentation export to `deploy/export_executorch.py`
  - `--segmentation` now instantiates `RFDETRSeg*` and documents the third `masks` output.
- [x] Add an output-path flag to `src/main.cpp`
  - `--output <path>` overrides `output_image.jpg` / `output_video.mp4` for either input kind.
- [x] Verify and close the `.gitignore` item
  - `output_image.jpg` joined `output_video.mp4`; arbitrary `--output` paths are the user's own,
    so they are intentionally not ignored.

---

## Phase 2 — GPU parity fixtures (Complete)

The parity gate the GPU work was supposed to be measured against was never built. Everything in Phases 3 and 4 depends on it.

Spec: [`features/2026-08-15-gpu-parity-fixtures/`](features/2026-08-15-gpu-parity-fixtures/) — acceptance criteria and tolerances live in its `validation.md`.

**Evidence from the first real gate run (2026-08-26).** The four combinations split by *preprocessing*, not postprocessing: `cpu-cpu` and `cpupre-gpupost` agree to the last digit, as do `gpupre-cpupost` and `gpu-gpu` — CUDA postprocessing is bit-identical to CPU. DALI preprocessing is not: max score delta 0.0163 on one image, 16× the 1e-3 score tolerance. One image and final scores rather than the preprocessed tensor, so not conclusive — but write these fixtures expecting to find a discrepancy, not to confirm its absence. Details in [CHANGELOG.md](../CHANGELOG.md).

The fixtures confirmed it and split the cause: resize + normalise matches within `8.8e-3`, while the
encoded path diverges up to `0.069` — entirely the JPEG decode (nvJPEG vs stb). See
`tests/data/gpu_parity/README.md`.

- [x] Add golden CPU fixtures under `tests/data/gpu_parity/`
  - A small, a wide, and a tall image; the CPU-produced preprocessed tensor and final detections
    are stored with explicit tolerances, plus a `dense` synthetic fixture (200 detections) that
    saturates the cap
- [x] Add `tests/unit/test_gpu_parity.cpp`
  - Follows `tests/unit/test_gpu_postprocess.cpp` for the `GTEST_SKIP()`-without-a-device pattern;
    `SKIP_WITHOUT_GPU` and `MockDeviceBackend` lifted into `tests/unit/gpu_test_utils.hpp`
- [x] Add `tests/benchmark/bench_gpu_pipeline.cpp` and register it in `CMakeLists.txt`
  - Times preprocess (CPU and DALI), the H2D+D2H transfer, and segmentation postprocess (CPU and
    CUDA) separately; the H2D+infer+D2H stage is engine-bound and measured on the Phase 4 gate

---

## Phase 3 — GPU build and CI integration (Complete)

- [x] Add a `gpu-pipeline` configure preset to `CMakePresets.json`
  - New `gpu-pipeline` preset: TensorRT + `USE_GPU_PIPELINE=ON`, DALI_ROOT defaulting to
    `~/dependencies/dali`; TensorRT resolves from `TENSORRT_ROOTDIR` or the download resolver
- [x] Add DALI staging and the GPU options to `dockerfile.trt`
  - Already landed with the backend-split Dockerfiles: `dockerfile.trt` carries the `GPU_PIPELINE`
    build arg and the `dali-fetch`/`dali-selected` staging stages (this roadmap item predated it)
- [x] Add a compile-only GPU job to CI
  - `gpu-compile.yml` compiles TensorRT/DALI/CUDA under `-DWERROR=ON` across all four option
    combinations (this roadmap item predated it). It compiles the library only — the GPU test
    targets link real DALI/TensorRT symbols and are compile-verified on the manual GPU build

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
- [ ] Move `[Unreleased]` to `[v0.5.0]`, sync `README.md` version statements against `CMakeLists.txt`, `CMakePresets.json`, `deploy/requirements.txt`, `dockerfile.*`, and `docs/export.md`
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
