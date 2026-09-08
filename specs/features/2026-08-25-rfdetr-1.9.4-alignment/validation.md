# Validation: RF-DETR 1.9.3 / 1.9.4 Alignment

Status recorded as of 2026-08-25. Anything not ticked is stated as such in `CHANGELOG.md` rather
than implied to work, per the skill's Step 5.

## Automated — no GPU

Commands quoted from `AGENTS.md`.

- [x] Default build: `cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build build --parallel`
- [x] Unit tests: `ctest --test-dir build --output-on-failure -R UnitTests` — **52/52 pass**
      (41 pre-existing + 11 added)
- [x] Integration tests: `ctest --test-dir build --output-on-failure -R IntegrationTests` — passes;
      the 4 model-backed cases report `SKIPPED` with no `.onnx` present, which is the documented
      behaviour, not a failure
- [x] Format: `find src tests -name '*.cpp' -o -name '*.hpp' | xargs clang-format-18 --dry-run --Werror` — clean
- [x] Cppcheck: `cppcheck --enable=all --std=c++20 --suppress=missingIncludeSystem --suppress=unmatchedSuppression --suppress=unusedFunction --error-exitcode=1 -I src src/` — clean
- [ ] Clang-tidy: `find src -name '*.cpp' | xargs clang-tidy-18 -p build` — **not run**

### Selection rules pinned by test

Each is an exact assertion, not a tolerance:

| Rule | Test |
|------|------|
| A query above threshold on 2 classes yields 2 detections sharing one box | `PostprocessTest.MultiLabelQueryYieldsEveryClassAboveThreshold` |
| Output is ordered by strictly descending score | `PostprocessTest.ResultsAreRankedByDescendingScore` |
| Exact ties order by ascending flattened index — asserts `class_ids == {2, 3, 0, 1}` | `PostprocessTest.TiesResolveByAscendingFlattenedIndex` |
| `max_detections` caps candidates *before* thresholding — 4 above threshold, cap 2, expect 2 | `PostprocessTest.MaxDetectionsCapsCandidatesBeforeThresholding` |
| A NaN score is dropped, not emitted with NaN confidence | `PostprocessTest.NaNScoreIsDroppedNotKept` |
| `background_class_id = none` keeps slot 0 as a real class | `PostprocessTest.BackgroundClassIdNoneKeepsEverySlot` |
| `background_class_id = -1` excludes the final slot | `PostprocessTest.BackgroundClassIdCountsFromTheEnd` |
| An out-of-range slot throws `std::invalid_argument` | `PostprocessTest.BackgroundClassIdOutOfRangeRejected` |
| Negative `max_detections` throws at construction | `PostprocessTest.NegativeMaxDetectionsRejectedAtConstruction` |
| Segmentation shares the identical selection | `PostprocessTest.SegmentationRanksFlattenedQueryClassPairs` |

## Automated — with a device

Run per [gpu-verify](../../../.claude/skills/gpu-verify/SKILL.md). Hardware used: RTX 3060 Laptop
(sm_86), TensorRT 10.13.3.9, nvcc 12.0.

- [x] Configure and build:
      `cmake -S . -B build-gpu -G Ninja -DUSE_ONNX_RUNTIME=OFF -DUSE_TENSORRT=ON -DUSE_CUDA_POSTPROCESS=ON -DTENSORRT_ROOTDIR=$HOME/dependencies/TensorRT-10.13.3.9 -DCMAKE_BUILD_TYPE=Release`
- [x] `build-gpu/unit_tests` — **62/62 pass** (52 CPU + 10 GPU)
- [x] CPU-vs-GPU segmentation parity, all 6 cases pass: `MatchesCpuOnSparseDetections`,
      `MatchesCpuOnDenseDetections` (saturates the cap at 50), `MatchesCpuOnNonSquareLargeFrame`,
      `MatchesCpuWithNonZeroMaskThreshold`, `HandlesNoDetections`, `ReusesOneInstanceAcrossFrames`.
      **Tolerance: exact.** These compare counts, class ids, scores, boxes, and mask bytes for
      equality — the gate on Group 3, since the kernel now ranks a compacted candidate set and any
      divergence from `select_topk_multiclass()` shows up as a mismatch.

> First attempt failed all 10 with `cudaErrorMemoryAllocation` at `cudaSetDevice` — an unrelated
> process held 5.2 GB of the 6 GB card. Re-run once free. Not a code failure; recorded so the
> signature is recognisable.

## Compile-without-device

- [x] The GPU targets build on a machine with nvcc; `test_gpu_postprocess.cpp` reports `SKIPPED`
      rather than `FAILED` when no device is present (`SKIP_WITHOUT_GPU()` guard, unchanged)
- [x] The default ONNX Runtime build compiles with no CUDA toolchain at all

## Manual

- [ ] **Re-export at least one model with the new package version and run it end to end** — skill
      Step 5, **not done**. `rfdetr_venv` currently holds rfdetr 1.8.0 and no `.onnx` is checked in;
      this needs `pip install rfdetr[onnx]==1.9.4`, a re-export of `rf-detr-nano.pth`, and a run
      through `inference_app`. Until then, the multi-label change is verified only against synthetic
      tensors, never against real model output. Stated in `CHANGELOG.md`.
- [ ] Visual check of `--background-class-id none` against a fine-tuned (0-based) checkpoint — no
      such checkpoint available here.

## Definition of done

- [x] `CHANGELOG.md` updated under `[Unreleased]`, house style, per-file table
- [x] `README.md` updated in the same change (Spec Sync rule)
- [x] `specs/mission.md` and `specs/tech-stack.md` version statements propagated in the same commit
- [ ] Roadmap items ticked — n/a, this is the standing obligation rather than a numbered phase
- [ ] Branch merged into `develop` and deleted
