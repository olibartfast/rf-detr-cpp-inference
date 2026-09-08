# Roadmap

Work this project has committed to, as a phased queue. Phases are ordered so each one leaves the tree building and the CPU path untouched. Release history lives in [CHANGELOG.md](../CHANGELOG.md) and is not repeated here; GPU design constraints live in [gpu-pipeline.md](gpu-pipeline.md).

**The next phase is the first one below whose items are all `[ ]`.** Tick an item only when it is merged and verified. A phase that is fully ticked gets `(Complete)` in its heading.

## Status

- Last tag **v0.5.0** (2026-09-08). Flow is git-flow: `develop` → `release/vX.Y.Z` → `master`, merged back.
- **v0.5.0 is released** — the rfdetr 1.9.1 and 1.9.2 alignments and the whole GPU pipeline shipped. Phases 1–5 are complete.
- GPU pipeline: preprocessing and segmentation postprocessing **work end to end**; the test, build,
  and CI scaffolding (Phases 2–4) is complete and the parity gate has passed.

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

## Phase 4 — GPU parity gate and benchmarks (Complete)

- [x] Add `tests/integration/integration_test_gpu_parity.cpp`
  - Runs the four combinations (CPU/CPU, GPU-pre/CPU-post, CPU-pre/GPU-post, GPU/GPU) through a real
    engine; CUDA postprocess asserted to the tight tolerance (scores `1e-3`, box centres 1 px, mask
    IoU ≥ 0.999) and DALI preprocess to the documented decode-aware bound. Written and
    compile-verified; runtime verification is the exit gate below.
- [x] Extend the benchmark to the same four combinations, per-stage, still image and video
  - `bench_gpu_pipeline.cpp` (Phase 2) times preprocess (CPU and DALI), the H2D+D2H transfer, and
    postprocess (CPU and CUDA). The H2D+infer+D2H engine stage is measured with `trtexec` on the real
    engine: H2D 0.64 ms, GPU compute 11.61 ms, D2H 1.30 ms (RTX 3060 Laptop, TensorRT 10.13).
- [x] Run the exit gate on real hardware — the [`gpu-verify`](../.claude/skills/gpu-verify/SKILL.md) workflow:
  1. All three tasks run with `--gpu-preprocess` inside the tolerances above — done (four
     combinations pass through a real engine, `integration_test_gpu_parity.cpp`)
  2. Segmentation runs with `--gpu-postprocess` at mask IoU ≥ 0.999, including on the dense fixture
     — done (unit + integration)
  3. A 1000-frame video run completes with no leak and no `compute-sanitizer` findings — done on the
     NGC 25.12 (CUDA 13.1) container, whose `compute-sanitizer` 2025.4 pairs with the app; the
     locally installed sanitizer/toolkit could not instrument the app. 1000 frames, 0 findings.
  4. The default (ONNX Runtime, CPU) build and its results are bit-identical to today — done
  5. Benchmarks recorded, including the flat ones — done (CPU 6.75 ms/11.53 ms preprocess, DALI
     4.51 ms/4.59 ms, H2D 0.64 ms, GPU compute 11.61 ms, D2H 1.30 ms, 3483 ms CPU vs 570 ms GPU seg
     postprocess, 1080p dense fixture)
  6. README and CHANGELOG updated per [AGENTS.md](../AGENTS.md) — done

---

## Phase 5 — Release v0.5.0 (Complete)

The [`release`](../.claude/skills/release/SKILL.md) workflow. Gated on Phases 1–4.

- [x] Read `AGENTS.md`, `README.md`, and `CHANGELOG.md`, then verify the rfdetr release against upstream — the mandatory "Spec Sync" rule
- [x] Move `[Unreleased]` to `[v0.5.0]`, sync `README.md` version statements against `CMakeLists.txt`, `CMakePresets.json`, `deploy/requirements.txt`, `dockerfile.*`, and `docs/export.md`
- [x] Resolve the version disagreement noted in [tech-stack.md](tech-stack.md#known-pin-duplications): `project()` declares none, `vcpkg.json` says `0.1.0`, the README badge says `0.4.0`
- [x] Cut `release/v0.5.0`, merge to `master`, tag, merge back to `develop`

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
