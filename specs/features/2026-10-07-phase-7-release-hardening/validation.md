# Validation — Phase 7, release hardening

Written before any code change. The baseline was captured on `cc0d4f9`, as described below.

## Baseline (captured 2026-10-07, before implementation)

- **Build.** `cmake -S . -B build-baseline -G Ninja -DCMAKE_BUILD_TYPE=Release` at `cc0d4f9`
  (ONNX Runtime, FFmpeg media).
- **Workload.** [`golden_cli.sh`](golden_cli.sh) `<app> <out>`. The baseline hashes are at
  `/tmp/rfdetr-phase7/baseline/SHA256SUMS`, and the clip is `/tmp/rfdetr-phase7-clip.mp4`.
  - Detection, segmentation, keypoint, and keypoint with `--keypoint-counts 0,17`.
  - Detection with `--threshold 0.3 --max-detections 5 --background-class-id none`.
  - Segmentation with `--mask-threshold -0.5 --resolution 576`.
  - Usage (`argc < 4`).
  - `--threshold abc`, `--max-detections 1.5`, `--keypoint-counts 0,x`, a missing model, and
    `--gpu-preprocess` on a CPU build.
  - A 30-frame 640-wide H.264 clip made from `data/dog.jpg`.
- **Inputs.** Models `data/models/rfdetr-nano-1101.onnx`, `rfdetr-seg-nano-576.onnx` and
  `rfdetr-keypoint-preview.onnx`; image `data/dog.jpg`; labels `data/coco-labels-91.txt`.
- **Captured per run.** stdout, stderr and exit code in separate files, plus the output image:
  45 SHA-256 hashes in all.
- **Normalisation.** The output directory becomes `<OUT>`, and libx264's `@ 0x…` pointer
  addresses become `<PTR>`.
- **Determinism.** Two baseline runs give identical `SHA256SUMS`.

## Automated — no GPU

| ID | Req | Check | Pass when | Result |
|----|-----|-------|-----------|--------|
| V-1 | all | `./scripts/scoreboard.sh`, once per packet, by the implementer | `SCOREBOARD: PASS` | |
| V-2 | R-3, R-5 | `cmake -S . -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON && find src -name '*.cpp' ! -name 'tensorrt_backend.cpp' \| xargs clang-tidy-18 -p build` | 0 warnings, exit 0 with `WarningsAsErrors: '*'` | |
| V-3 | R-6 | `golden_cli.sh` on the phase tip, then `diff baseline/SHA256SUMS candidate/SHA256SUMS` | No difference, all 45 entries | |
| V-4 | R-1 | `ctest --test-dir build --output-on-failure -R UnitTests`, including the new `CliOptions*` cases | All pass; every flag, error message, range check, `none`, unknown-flag and trailing-flag case is asserted | |
| V-5 | R-1 | Numeric-parse edge cases (`" 5"`, `"+5"`, `"0x10"`, `"1e-1"`, `"inf"`, `"nan"`) are pinned in tests to the outcome observed on the **baseline** binary | Tests encode the measured baseline outcome, not an assumed one | |
| V-6 | R-4 | V-3's `kp`/`kpcnt` entries | Byte-identical (keypoint output printed to 6 significant digits, and the drawn image) | |
| V-7 | all | Format: `find src tests -name '*.cpp' -o -name '*.hpp' \| xargs clang-format-18 --dry-run --Werror` | Exit 0 | |
| V-8 | all | `cppcheck --enable=all --std=c++20 --suppress=missingIncludeSystem --suppress=unmatchedSuppression --suppress=unusedFunction --error-exitcode=1 -I src src/` | Exit 0 | |
| V-9 | all | ASan+UBSan: `cmake -S . -B build-san -DCMAKE_BUILD_TYPE=Debug -DSANITIZERS=ON`, then `./build-san/unit_tests` | 0 sanitizer reports | |
| V-10 | R-3 | `-DUSE_OPENCV=ON -DWERROR=ON` builds `inference_app` and `unit_tests` | Builds; unit tests pass | |
| V-11 | R-9 | `./scripts/check_version_sync.sh` | Passes | |
| V-12 | R-8 | The investigation's V-14…V-22 (its `validation.md`) | Per that file; its own tolerances govern | |

## Automated — with a device (RTX 3060 Laptop, `rfdetr-p6:*` images)

| ID | Req | Check | Pass when | Result |
|----|-----|-------|-----------|--------|
| V-13 | R-1, R-2 | `inference_app` builds with `-DUSE_ONNX_RUNTIME=OFF -DUSE_TENSORRT=ON -DUSE_GPU_PIPELINE=ON -DWERROR=ON`, and again with `-DUSE_DALI=ON` added | Both link; `unit_tests` passes or reports `SKIPPED` | |
| V-14 | R-6 (GPU) | `data/dog.jpg` through the four combinations (CPU/CPU, `--gpu-preprocess`, `--gpu-postprocess --segmentation`, both), with the baseline `cc0d4f9` build and the phase-tip build in the same image and with the same engine | stdout, stderr, exit code and output image byte-identical per combination | |

## Compile without a device

| ID | Check | Pass when | Result |
|----|-------|-----------|--------|
| V-15 | The ExecuTorch build of `inference_app` against the v1.4.0 optimized-kernels prefix (memory: ExecuTorch install prefix) | Builds under `-DWERROR=ON`; `kExampleModel` usage text shows `.pte` | |

## Manual

| ID | Req | Check | Pass when | Result |
|----|-----|-------|-----------|--------|
| V-16 | R-7 | CUDA-pipeline build: video with `--display --gpu-preprocess --gpu-postprocess --segmentation`, with X forwarded into the container | The user confirms the window opens, plays the annotated frames with masks, closes cleanly at the end of the stream and on window close, and the app exits 0 | |
| V-17 | R-7 | The same as V-16 in the DALI build | As V-16 | |
| V-18 | R-7 | CPU/CPU `--display` in the GPU build (control) | As V-16. A failure here makes V-16/V-17 failures not GPU-specific | |

A window that opens but is not watched is recorded as "opened, not inspected" (Q-2), never as a pass.

## Definition of done

- [ ] V-1…V-18 have results. Anything not run is `UNRUN`, with its reason.
- [ ] Each packet has a `reviewer` verdict recorded in the commit message that merges it.
- [ ] `AGENTS.md`, `specs/tech-stack.md`, `README.md` and `docs/advanced-usage.md` are updated in
      one commit (Spec Sync).
- [ ] `CHANGELOG.md` `[Unreleased]` has the entry, with prose plus a per-file table.
- [ ] Phase 7 is ticked in `specs/roadmap.md` and its heading marked `(Complete)`.
- [ ] The branch is merged to `develop` and deleted. `feature/performance-memory-investigation`
      and `stash@{0}` are dropped only after the user confirms. Their content is on this branch.
