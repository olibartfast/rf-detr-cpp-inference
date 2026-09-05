# CHANGELOG

All notable changes to this C++ inference project: its own features and fixes, and the
upstream `rfdetr` releases it is kept in step with.

---

## [Unreleased]

### RF-DETR 1.10.0 alignment

**Upstream release:** [1.10.0](https://github.com/roboflow/rf-detr/releases/tag/1.10.0)

1.10.0 is primarily a training and Python-inference performance release. Its optimizer,
matcher, validation, DataLoader, dataset, metric, and segmentation-loss changes do not execute in
this native runtime. Python `predict()` moves fewer bytes and performs less host work, while the
release explicitly preserves detections. The exported input/output names, order, shapes, dtypes,
and decode semantics are unchanged, so no C++, CUDA, backend validation, or DALI pipeline changed.
The default ONNX opset remains 17 (`opset_version: int = 17` in `detr.py` and the ONNX exporter,
and the export CLI forwards it), so the pinned `ONNX_OPSET_VERSION=17` is unchanged.

The deploy-facing change is artifact naming. `RFDETR.export()` now accepts `output_name`; without
it, ExecuTorch names include the delegate (`*_xnnpack.pte`) and TensorRT names include resolved
precision (`*_fp16.trt` or `*_fp32.trt`). The repository export scripts pass stable explicit stems,
validate the returned path exists, and print that authoritative path instead of guessing it.

| File | Change |
|------|--------|
| `versions.env`, `deploy/requirements.txt` | Export tooling pin 1.9.4 → 1.10.0; no runtime pin moved. |
| `deploy/export_*.py`, `deploy/export_common.py` | Forward stable `output_name` values and validate/report the returned artifact. |
| `tests/python/test_export_scripts.py` | Hermetic coverage for default/custom names, missing returns, and the keypoint compatibility copy. |
| `specs/features/2026-09-04-rfdetr-1.10.0-alignment/` | Required classification, implementation plan, and validation record. |
| `README.md`, `specs/{mission,tech-stack,roadmap}.md`, `docs/` | Live package references and export naming guidance synchronized. |

Validation status is recorded in the alignment spec. ExecuTorch re-export/runtime validation and
native Python TensorRT export remain `UNRUN` unless their matching optional environments are
available; no backend parity is inferred from the unchanged contract.

### Build: split the parametric `Dockerfile` into three backend Dockerfiles

The single parametric `Dockerfile` (driven by `INFERENCE_BACKEND=onnx|tensorrt|executorch`
plus `MEDIA_BACKEND` and the TensorRT-only `GPU_PIPELINE`) is split into three files, one per
backend, chosen by which file you pass to `-f`. There is no bare `Dockerfile`, so a build never
silently defaults to a backend. This reverses the v0.3.0 consolidation that merged
`Dockerfile.onnx` + `Dockerfile.tensorrt` into one file, and folds in the ExecuTorch backend and
the GPU pipeline that arrived after it.

Docker has no usable include mechanism at this project's toolchain floor, so the blocks that
must stay identical across the three files are triplicated under `# === shared:<name> ===`
markers and guarded by a new `scripts/check_dockerfile_parity.sh` (the `Dockerfile shared
blocks` step in `lint.yml`). `scripts/check_version_sync.sh` now checks each pin against the
specific file that restates it.

| File | Change |
|------|--------|
| `dockerfile.onnxrt` | New. ONNX Runtime (CPU) + `MEDIA_BACKEND`; the `INFERENCE_BACKEND` conditionals collapse to literal `-DUSE_ONNX_RUNTIME=ON` etc. |
| `dockerfile.executorch` | New. ExecuTorch (CPU, `.pte`) + `MEDIA_BACKEND`; the from-source build is now unconditional and stays above `COPY . .` (the ~11-min cache rule). |
| `dockerfile.trt` | New. TensorRT (GPU) + `MEDIA_BACKEND` + `GPU_PIPELINE`; carries the DALI staging stages and the `dali-selected` forwarding (still required — `GPU_PIPELINE` remains an ARG). |
| `Dockerfile` | Deleted. |
| `scripts/check_version_sync.sh` | Rewritten: an `expect_all` wrapper checks each pin against the file(s) that restate it, with a missing-file guard. |
| `scripts/check_dockerfile_parity.sh` | New. Extracts and diffs the `shared:` blocks across the three Dockerfiles. |
| `.github/workflows/lint.yml` | `version-sync` job gained the `Dockerfile shared blocks` step. |
| `docs/docker.md`, `README.md`, `AGENTS.md`, `specs/tech-stack.md`, `specs/roadmap.md`, `docs/building.md`, `.claude/skills/{release,rfdetr-alignment}/SKILL.md` | Updated to the three-file layout and lowercase names. |
| `.gitignore` | Ignores `third_party/googletest/`, the offline gtest pre-seed. |

The builds also gained an offline gtest path. `find_dependency_unified(GTest)` resolves via
`FetchContent` unconditionally, so every image's configure `git clone`s googletest from
github.com — the *only* GitHub dependency `dockerfile.trt` has left (TensorRT is shimmed from
the NGC image, DALI staged from the Triton image, stb/font8x8 vendored). When the gitignored
`third_party/googletest/` (the pinned `release-1.12.1` source, e.g. copied from an existing
`build/_deps/gtest-src`) is present in the build context, each Dockerfile passes
`FETCHCONTENT_SOURCE_DIR_GTEST` and configure skips the clone; found when a sandboxed GPU
build could not reach github.com and died at configure.

The first build of each image is a full rebuild (new file content has no layer cache to reuse);
`docker build -f dockerfile.<backend> .` is the new invocation, and both base images must stay
Ubuntu 24.04 (the FFmpeg runtime library names are the 24.04 set). GPU/DALI/ExecuTorch image
behaviour is unchanged and still needs manual verification — CI builds no Docker images.

The TensorRT builder invokes Ubuntu `/usr/bin/cmake` explicitly because the pinned NGC 25.12
image puts CMake 3.24 first on `PATH`; that older copy cannot enable CUDA C++20 for the image's CUDA
13.1 compiler and otherwise fails during CMake generation.

### Build: `versions.env` as the single source of truth for dependency pins

Third-party versions were spread across the tree, several of them written down more than
once. `specs/tech-stack.md` had a standing "Known pin duplications" section listing the
worst of it: TensorRT `10.13.3.9` in both `TensorRT.cmake` and a hardcoded path in the
`Dockerfile`; the NGC tag `25.12` in three scripts; `scripts/ci/stage_gpu_headers.sh`
restating TensorRT and DALI a third time, in the form the CI compile gate fetches. Each
was a comment asking the next person to remember.

All of them now live in [`versions.env`](versions.env), and two loaders feed the consumers:

| Loader | Consumers |
|--------|-----------|
| `cmake/versions.cmake` | Included by `CMakeLists.txt` before `cmake/deps/Deps.cmake`, so every `cmake/deps/packages/*.cmake` interpolates the values. Each pin is a `CACHE STRING` — `-DTENSORRT_VERSION=…` still overrides it |
| `scripts/versions.sh` | `source`d by `fetch_dali.sh`, `generate_dali_pipelines.sh`, `ci/stage_gpu_headers.sh`, `run_gate.sh`, `export_trt.sh` and `gpu-compile.yml`. Never clobbers a value already in the environment, so the documented `TRITON_IMAGE=… ./scripts/fetch_dali.sh` overrides are unchanged |

Coordinates that are truncations of another pin are *derived*, not stored, so a TensorRT
bump stays one line. `TENSORRT_SHORT_VERSION` (`10.13.3`, the download-URL directory and
the Conan recipe) is derived by both loaders; `TENSORRT_DEB_VERSION`
(`10.13.3.9-1+cuda13.0`, the apt packages CI stages headers from) and the two NGC image
names are derived in `scripts/versions.sh` only, having no CMake consumer.

Four formats cannot read a file: the Dockerfile's `ARG` defaults, `conanfile.txt`,
`deploy/requirements.txt` and the argparse defaults in `deploy/export_*.py`. They restate
the values, and `scripts/check_version_sync.sh` — the new `Version Sync` job in
`lint.yml` — fails with the offending file, the expected line and the line found. The
Dockerfile additionally gained `TENSORRT_VERSION`, `NGC_CONTAINER_TAG` and
`DOCKER_BASE_IMAGE` args, replacing three hardcoded TensorRT paths and two literal
`FROM` images.

`gpu-compile.yml` keyed its staged-header cache on `hashFiles('scripts/ci/stage_gpu_headers.sh')`.
With the versions moved out of that script, a bump would have silently reused stale
headers, so the key now covers `versions.env` and `scripts/versions.sh` too.

Four defects found in review, each a way the "single source of truth" could have been true
on paper and false in practice:

- **Stale CMake cache.** A plain `set(… CACHE STRING …)` never overwrites an existing entry,
  so editing `versions.env` in an existing build tree reconfigured (`CMAKE_CONFIGURE_DEPENDS`
  saw the change) and still built the *old* version. Every pin now also records the
  `versions.env` value it was configured from in an INTERNAL stamp: cache equal to stamp means
  the value is still the file default and the new one is adopted; cache different means someone
  passed `-D` and it is left alone.
- **Dockerfile build arg only half-wired.** `--build-arg TENSORRT_VERSION` moved the shim
  directory but was never passed to CMake, which kept reading the `versions.env` default, missed
  the shim, and would have downloaded a different TensorRT than the NGC image provides. The
  `cmake` invocation now receives `-DTENSORRT_VERSION`.
- **CI CUDA toolkit hardcoded.** `gpu-compile.yml` installed `cuda-nvcc-13-0` /
  `cuda-cudart-dev-13-0` literally while `stage_gpu_headers.sh` derived the TensorRT package
  coordinates from `CUDA_VERSION`, so a bump would have compiled the new TensorRT headers
  against the old toolkit. The workflow now derives the apt suffix and the `PATH` entry from
  the pin.
- **GPU gate provenance.** `scripts/run_gate.sh` was a consumer that the first pass missed. Its `probe_dali` scraped
`fetch_dali.sh` with `sed` for the `${TRITON_IMAGE:-…}` assignment this change removed, so the
GPU gate's `environment.txt` would have recorded an empty `extracted from:` field and lost the
DALI provenance its verification record depends on. It now sources `versions.sh` like the
staging scripts and reports `${TRITON_IMAGE}` directly — the same value `fetch_dali.sh`
resolves, which is what the original `sed` was reaching for.

No version changed. Both backends resolve to byte-identical download URLs and extraction
directories — verified by configuring ONNX Runtime and TensorRT against the existing
`build/_deps` cache, where each was found in place rather than re-downloaded.

### Docs: split the 1012-line README into five subdocuments

`README.md` had grown to 1012 lines, a third of it (`## Building`, 369 lines) step-by-step
procedure that a reader evaluating the project never needs. The reference material and the
procedure now live apart: the README keeps what has to be verifiable at a glance — supported
versions, the pip packages for export tooling, backend constraints, and the full CMake option
list — and everything procedural moved under `docs/`.

That division is not cosmetic. `AGENTS.md` requires the README itself to carry the version,
CMake-option, backend-constraint and pip-package statements, and `Spec Sync` requires them
reconciled against `CMakeLists.txt`, `deploy/requirements.txt` and the Dockerfiles on every
dependency-facing change. Keeping exactly those in the README means the rule still holds as
written, with no amendment.

| File | Change |
|------|--------|
| `README.md` | 1012 → 283 lines. Keeps the intro, a documentation index, Dependencies, Model Setup, a new Quick Start (one build and one run, plus a one-liner for each of the other three backends), Backend Selection, Build Options, five usage examples, the CI table, and Acknowledgements. |
| `docs/building.md` | New, 256 lines. Toolchain install, dependency-resolution modes, and every build configuration: ONNX Runtime, TensorRT, ExecuTorch (including the install-prefix build), the GPU pipeline, and the OpenCV media backend. |
| `docs/usage.md` | New, 167 lines. Every run mode, the tuning flags, how detections are ranked, the GPU pipeline flags, and the `Config` reference. |
| `docs/architecture.md` | New, 153 lines. The GPU pipeline, the four-stage video ring buffer, model output shapes, C++ result types, and the processing stages. |
| `docs/development.md` | New, 175 lines. clang-format, clang-tidy, cppcheck, the three sanitizer modes, Valgrind/profiling, pre-commit, strict compilation, unit and integration tests, cross-backend checks, and benchmarks. |
| `docs/docker.md` | New, 62 lines. The parametric `Dockerfile` and its inference-backend × media-backend matrix. |
| `docs/export.md` | Two links into README sections that no longer exist now point at `usage.md` and `building.md`. |

No prose was dropped: the move was mechanical, and the only old lines without a home in the new
tree are the nine replaced table-of-contents entries. All 83 relative links and heading anchors
across the README and `docs/` resolve.

### CI: compile the TensorRT and GPU paths, and fix what that immediately caught

`src/backends/tensorrt_backend.cpp` and `src/gpu/` were never compiled by CI. The first build of
them happened on a rented GPU box, on the metered clock, and it failed before it ran a single
kernel: eight `-Werror` errors — `-Wsign-conversion` on the binding-volume and file-size
arithmetic, `-Wdeprecated-declarations` on three TensorRT 10 APIs, and an unused parameter. None
of that needs a GPU to catch. It needs headers.

`gpu-compile.yml` stages headers-only TensorRT and DALI prefixes and builds the static library
target with `-DWERROR=ON` across all four `USE_DALI`/`USE_CUDA_POSTPROCESS` combinations, since the
two GPU halves are independent options and each combination is a distinct set of `#if` branches. It
does not link and it does not run — the staged shared objects are empty stubs, present only because
the dependency resolver checks that the files exist. This is a compile gate, not a verification
gate: [gpu-verify](.claude/skills/gpu-verify/SKILL.md) still owns behaviour, and Phase 2 still
gates Phase 4.

Its first run on `develop` earned its keep immediately: three of the four combinations passed and
the plain `-DUSE_TENSORRT=ON` one did not. TensorRT's own headers `#include <cuda_runtime_api.h>`,
but the `PROVIDED_CUDA ON` branch of the dependency resolver contributed only the CUDA *library*
and its rpath — never the CUDA include directory. That build therefore compiled only on machines
where CUDA happens to sit on the default include path, which is why nobody had hit it: every
configuration anyone actually builds also sets `USE_DALI` or `USE_CUDA_POSTPROCESS`, and those
resolve `CUDAToolkit` separately and carry its includes along. The matrix is what separated them.

#### Fixed

| File | Change |
|------|--------|
| `cmake/deps/strategies/ProvidedPackageManager.cmake` | `PROVIDED_CUDA ON` now contributes the CUDA include directory alongside the library, from `CUDAToolkit_INCLUDE_DIRS` when `CUDAToolkit` resolves and otherwise from the directory holding the located `cudart`. Found by `gpu-compile.yml`'s plain-TensorRT job on its first run. |
| `src/backends/tensorrt_backend.cpp` | Binding and output volumes cast to `size_t` explicitly; `serialize_engine`/`deserialize_engine` cast to `std::streamsize`, and `deserialize_engine` now rejects a negative `tellg()` rather than converting it into a huge allocation. |
| `src/backends/tensorrt_backend.cpp` | `kEXPLICIT_BATCH` is skipped on TensorRT 10, where explicit batch is the only mode and `createNetworkV2` takes no flag. `platformHasFastFp16()` is dropped there too — every GPU TensorRT 10 supports has fast FP16. `BuilderFlag::kFP16` keeps a localized deprecation suppression: it is deprecated in favour of strongly-typed networks, but this build is deliberately weakly typed so that an FP32 ONNX still gets FP16 kernels, and the flag is the only way to ask for that. Migrating would silently cost FP16. |
| `src/backends/tensorrt_backend.cpp` | `build_engine_from_onnx` marks `input_shape` unused, matching the idiom already used elsewhere in the file. |

#### Added

| File | Change |
|------|--------|
| `.github/workflows/gpu-compile.yml` | New. Compile-only matrix — TensorRT alone, +DALI, +CUDA postprocess, +both — with `-DWERROR=ON`. Installs `cuda-nvcc-13-0` and `cuda-cudart-dev-13-0` from NVIDIA's apt repo, and caches the staged headers on the hash of the staging script so a pin change invalidates them. |
| `scripts/ci/stage_gpu_headers.sh` | New. Assembles the two prefixes from the smallest artifacts carrying the pinned headers: the `libnvinfer-headers-dev` / `libnvonnxparsers-dev` debs (~130 KB, unpacked with `dpkg-deb -x` so the 2 GB `libnvinfer10` runtime is never pulled in), and `include/` out of the DALI wheel — the same tree `scripts/fetch_dali.sh` takes from the Triton container. The real distributions are 6.2 GB and 380 MB. |
| `AGENTS.md`, `README.md`, `specs/tech-stack.md` | The claim that CI never builds these paths is no longer true. The CI coverage tables and the pin-duplication list record the new workflow and the versions `stage_gpu_headers.sh` has to keep in step with `cmake/deps/packages/TensorRT.cmake` and `scripts/fetch_dali.sh`. |

### RF-DETR 1.9.3 / 1.9.4 alignment

**Upstream releases**: [1.9.3](https://github.com/roboflow/rf-detr/releases/tag/1.9.3),
[1.9.4](https://github.com/roboflow/rf-detr/releases/tag/1.9.4)

Continues from the 1.9.2 alignment below, which correctly found nothing for C++ to do.
1.9.3 and 1.9.4 are different: two changes land on the decode path, and both were wrong
here in the same way they were wrong upstream:

1. **Detections were being dropped.** Class scores are independent sigmoids, so one
   query can score above threshold on several classes at once. The detection and
   keypoint paths took a per-query `argmax` and kept only the strongest, exactly the bug
   1.9.3 (PR #1320) fixed in upstream's own ONNX/TFLite decoders. All three heads now
   rank the flattened *(query, class)* grid the way `PostProcess._select_topk` does.
2. **The background logit slot was hardcoded.** 1.9.4 (PR #1397) made it an explicit
   argument, because the assumption is checkpoint-dependent and a wrong guess shifts
   every reported label. Exposed here as `Config::background_class_id` and
   `--background-class-id`.

Ordering also became contractual: descending score, then ascending flattened query/class
index. That fell out of 1.9.3 replacing `torch.topk` with a stable `argsort`.

#### Fixed

| File | Change |
|------|--------|
| `src/rfdetr_inference.cpp` | `postprocess_outputs()` and `postprocess_keypoint_outputs()` rank the flattened *(query, class)* grid instead of taking a per-query `argmax`, so a query above threshold on more than one class now yields one detection per class rather than only its strongest. Both also honour `max_detections` as the ranking cap, which the detection path previously ignored entirely. |
| `src/rfdetr_inference.cpp` | The segmentation path excludes the background column **before** ranking. It used to rank every column and skip background candidates afterwards, so background pairs consumed slots of the `max_detections` cap and could push real detections out of a crowded frame. |
| `src/rfdetr_inference.cpp`, `src/gpu/rfdetr_postprocess.cu` | Kept scores are now tested with `score > threshold` rather than `!(score <= threshold)`. A NaN score fails the first and passes the second, so a malformed logit used to be emitted as a detection with a NaN confidence. |
| `src/processing_utils.cpp` | The segmentation ranking comparator was `all_scores[i1] > all_scores[i2]`, which is not a strict weak ordering when any score is NaN — undefined behaviour in `std::partial_sort`. The shared `select_topk_multiclass()` substitutes `+inf` for NaN before comparing, which both restores the ordering and reproduces torch's NaN-ranks-first behaviour. |

#### Added

| File | Change |
|------|--------|
| `src/processing_utils.{hpp,cpp}` | `resolve_background_slot()`, `foreground_class_count()`, `slot_for_foreground_column()`, `build_foreground_scores()`, `select_topk_multiclass()` — C++ mirrors of upstream's `export/_class_layout.py` (1.9.4) and `export/_topk.py` (1.9.3). One selection rule now serves detection, segmentation, keypoint, and the CUDA kernels. |
| `src/rfdetr_inference.hpp` | `Config::background_class_id` (`std::optional<int>`, default `0`): the exported logit slot holding background, excluded before ranking. Negative values count from the end, `std::nullopt` keeps every slot — upstream's `background_class_id` semantics. |
| `src/main.cpp` | `--background-class-id <n\|none>`. |
| `tests/unit/test_rfdetr_inference.cpp` | Ten tests: multi-label selection (a query yielding two classes), descending-score output order, the ascending-flat-index tie rule, `max_detections` capping candidates before thresholding, NaN dropped rather than kept, `background_class_id` as `none` / negative / out-of-range, negative `max_detections` rejected at construction, and the segmentation path sharing the same selection. |

#### Changed

| File | Change |
|------|--------|
| `src/rfdetr_inference.cpp` | Construction rejects a negative `max_detections` with `std::invalid_argument`, mirroring 1.9.3 making `PostProcess(num_select=<negative>)` a construction-time `ValueError` instead of a silently accepted no-op. |
| `src/gpu/rfdetr_postprocess.{hpp,cu}` | `SegPostprocessParams::background_slot`; `decode_scores` compacts the foreground columns before the sort so the GPU ranks the same candidate set as the CPU. CUB's radix sort is stable, so ties already came out in ascending flattened index order — the CPU rule is now written down beside it. |
| `deploy/requirements.txt`, `deploy/export_executorch.py` | `rfdetr` 1.9.2 → 1.9.4. |
| `specs/mission.md`, `specs/tech-stack.md` | Version statement and export-tooling pin row to 1.9.4, propagated in this same commit per the Spec Sync rule. |
| `specs/features/2026-08-25-rfdetr-1.9.4-alignment/` | Spec triple for this pass: the skill's Step 1 classification table, the decisions behind the `background_class_id` default and the pre-ranking exclusion, and a validation sheet recording what was and was not verified. Written after the implementation rather than before it — the reason is in its `requirements.md` → Context, and the `AGENTS.md` rule below is the fix. |
| `AGENTS.md` | New mandatory rule: `git fetch` and check against upstream **before starting work**, not only before pushing. This alignment was written against a `develop` four commits stale, so it missed the completed 1.9.2 pass, the `specs/` tree, and the `rfdetr-alignment` skill governing the task — and then conflicted in six files. |
| `README.md`, `docs/export.md` | Version statements to rfdetr 1.9.4. New "How detections are selected" README section (multi-label ranking, the cap-then-threshold order, the tie rule) and a `--background-class-id` entry explaining when the default is wrong. `docs/export.md` notes what 1.9.2–1.9.4 do and do not change for an exported model. |

#### Why the class ids did not move

Upstream's own decoder defaults to `background_class_id=-1` — background in the *final*
logit slot — and 1.9.4's release notes say plainly that this mis-decodes the official
pretrained COCO weights, whose final slot holds real category 90 (`toothbrush`). This
project has always used the other convention: slot 0 is background, slot *n* is COCO
category *n*, and `data/coco-labels-91.txt` is indexed by COCO id so slot *n* reads
entry *n*. That is upstream's `background_class_id=0` case, so the default stays `0` and
decoded labels are unchanged. The flag exists for checkpoints that are laid out
differently — a fine-tuned model with contiguous 0-based ids wants `none`.

#### Not mirrored

The rest of 1.9.3–1.9.4 does not reach an exported model: evaluation-sweep and matcher
performance, `metrics.csv` history across resumes, `BestModelCallback` scoring the
Lightning sanity check as a real epoch, EMA epoch-boundary double counting, YOLO
test-split resolution, AdamW-aware auto-batch probing, keypoint flip-pair and
Albumentations `TimeReverse`/`SquareSymmetry` augmentation fixes, non-square training
resize, and the TFLite conversion fixes. One is worth knowing about anyway and is
written up in `docs/export.md`: 1.9.3's `SegmentationHead.skip_blocks` fix is
training-only — the export path already applied the projection, so no re-export is
needed. (1.9.2's label-space change is covered in its own entry below.)

---

### RF-DETR 1.9.2 alignment

**Upstream release**: https://github.com/roboflow/rf-detr/releases/tag/1.9.2

RF-DETR 1.9.2 is a fix-and-performance release with no new public APIs and no
changes to exported model inputs, outputs, preprocessing, or runtime operator
requirements. The C++ inference implementation therefore needs no compatibility
change; the export package pins and current release guidance move to 1.9.2.

#### Breaking upstream dataset change

Hierarchical COCO datasets now share one label mapping derived from the training
split, and unannotated grouping/root categories no longer consume a class slot.
This fixes Roboflow COCO exports that previously produced an N+1-class head or
split-dependent label indices. Checkpoints trained before 1.9.2 retain their old
head width and label ordering; evaluating them against a dataset re-filtered by
1.9.2 can misalign per-class metrics. Retrain those checkpoints when adopting the
new label space. This affects Python training and dataset evaluation, not C++
postprocessing of an already-exported model.

#### Changed

| File | Change |
|------|--------|
| `deploy/requirements.txt` | `rfdetr[onnx]` 1.9.1 → 1.9.2. |
| `deploy/export_executorch.py` | Current ExecuTorch exporter install guidance moved to `rfdetr[executorch]==1.9.2`; the 1.9.1 `aten::linear.out` compatibility explanation remains applicable. |
| `README.md`, `docs/export.md` | Current ONNX, ExecuTorch, and TensorRT export pins moved to 1.9.2; historical 1.9.1 behaviour and runtime requirements remain documented. |
| `tests/integration/integration_test_rfdetr_inference.cpp` | Missing-model guidance now names rfdetr 1.9.2. |
| `AGENTS.md` | Added a mandatory rule to inspect repository documentation and verify official upstream release information before acting on version-alignment requests. |

The upstream matcher memory/time improvements, checkpoint-resume fixes, training
determinism changes, and Python-side prediction optimizations do not map to code
paths implemented by this repository.

---

### RF-DETR 1.9.1 alignment

**Upstream release**: https://github.com/roboflow/rf-detr/releases/tag/1.9.1

A quiet upstream release, but two things here needed fixing:

1. **Mask borders were slightly wrong.** Our image and mask resize extrapolated past
   the top and left edges instead of holding them. 1.9.1 wrote down the resize
   convention upstream uses, which made the mismatch obvious.
2. **ExecuTorch `.pte` files from 1.9.1 would not run.** The new export needs an
   operator our runtime did not have. Fixed by building ExecuTorch with its
   optimized kernels and moving the pinned runtime to v1.4.0.

#### Fixed

| File | Change |
|------|--------|
| `src/media.cpp` | Bilinear resize now clamps the source *coordinate* to the edge instead of clamping the sample *index* (new `clamp_source_coord` helper, used by `preprocess_bgr_image` and `resize_threshold_mask`). Clamping the index left a negative blend weight on output pixels that fall before the first source pixel, so those pixels were extrapolated instead of copied from the edge: upscaling `[0, 4, 4, 0]` to 12 wide gave `-1.333` at the first pixel where torch gives `0.0`. Masks are always upscaled, so this affected the first few rows and columns of every mask (about 5 of them for a 96px mask head on a 1080p frame) and could flip those pixels to the wrong side of the mask threshold. Preprocessing only hit it when the image was smaller than the model resolution. |
| `src/gpu/rfdetr_postprocess.cu` | The same fix in the CUDA mask kernel, which must stay bit-identical to the CPU version — `tests/unit/test_gpu_postprocess.cpp` compares them. |

#### Added

| File | Change |
|------|--------|
| `tests/unit/test_rfdetr_inference.cpp` | Three tests that pin the resize behaviour: `MaskResize.HalfPixelCenterBilinear` (sample positions), `MaskResize.LeadingEdgeClampsInsteadOfExtrapolating` and `PreprocessFrame.UpscaleDoesNotExtrapolatePastEdge` (the border fix — both fail without it). |
| `CMakeLists.txt` | Links ExecuTorch's `optimized_native_cpu_ops_lib` when the install prefix has it, otherwise `portable_ops_lib` plus a warning explaining that 1.9.1 models will not load. Only one of the two may be linked — both register kernels at startup, and registering an operator twice aborts the process. |

#### Changed

| File | Change |
|------|--------|
| `cmake/deps/packages/ExecuTorch.cmake` | ExecuTorch runtime `v1.3.1` → **`v1.4.0`**, the version a `.pte` must be exported with to run on this build, and the source-build fallback now enables `EXECUTORCH_BUILD_KERNELS_OPTIMIZED`. The operator library is chosen in `CMakeLists.txt` instead of being listed here, because which one exists depends on how the prefix was built. |
| `Dockerfile` | `EXECUTORCH_VERSION` default `v1.3.1` → `v1.4.0`; the source build enables the optimized kernels. The `extension_evalue_util` install patch is fixed upstream in v1.4.0, so the `sed` is now a no-op kept only for older tags. |
| `deploy/requirements.txt` | `rfdetr[onnx]` 1.9.0 → 1.9.1. |
| `deploy/export_executorch.py` | The closing note points at `rfdetr[executorch]==1.9.1` and its faster export, and replaces the claim that the extra installs a matching ExecuTorch with instructions to check `pip show executorch` against the v1.4.0 runtime. |
| `README.md`, `docs/export.md`, `AGENTS.md` | Versions updated to rfdetr 1.9.1 / ExecuTorch v1.4.0. New "Preprocessing parity" section in `docs/export.md` describing the resize convention and the tests that hold it. Documented that `EXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON` is required (it defaults to off), that pinning `rfdetr` does **not** pin ExecuTorch — the extra allows `>=1.3,<2.0`, so the installed version must be checked with `pip show executorch` and matched to the runtime — and 1.9.1's new install rules (`onnxruntime<1.24` on Python 3.10, `[executorch]` empty on 3.14). |

#### Why the ExecuTorch runtime had to change

1.9.1 rewrites the `addmm` operations XNNPACK does not accelerate back into
`aten.linear`; that is where its ~2.5× speedup comes from. An `RFDETRNano` export at
384×384 ends up with 6 `aten::linear.out` calls that run outside the delegate — and
`linear.out` ships only in ExecuTorch's *optimized* kernels. The portable kernels we
linked before have `addmm.out` and `mm.out` but no `linear.out`, in both v1.3.1 and
v1.4.0. Running a 1.9.1 model on the old setup fails immediately:

```
E executorch:method.cpp:819] Missing operator: [29] aten::linear.out
E executorch:method.cpp:1125] There are 6 instructions don't have corresponding operator registered
Error: ExecuTorch forward() failed with error code 20
```

So the optimized kernels are now enabled in the ExecuTorch builds this project drives
itself — the CMake source-build fallback and the Docker image — and the runtime pin is
v1.4.0. A prefix you built yourself and pass with `-DEXECUTORCH_ROOTDIR` is outside
that: if it has no optimized kernels the build still configures, links the portable
ones, and warns that 1.9.1 models will not load. Exporter and runtime must also be the
same ExecuTorch version; `rfdetr[executorch]` does not guarantee that on its own.

Checked end to end: `rf-detr-nano.pth` exported at 384×384 with rfdetr 1.9.1 to both
`.onnx` and `.pte` and run on `data/dog.jpg`. Against a v1.4.0 prefix with optimized
kernels, the `.pte` loads and gives the same 3 detections as the ONNX Runtime build —
identical boxes, scores within 1e-6. See
[docs/export.md](docs/export.md#verified-onnx--executorch-parity).

#### Why nothing else was ported

The other three upstream changes need no C++ work: the faster segmentation
postprocessing (PR #1268) avoids a PyTorch temporary this code never creates; skipping
the upsample of low-scoring masks (PR #1265) is what both the CPU and CUDA paths
already did; and the export resize parity work (PR #1269) only touches upstream's own
Python inference and calibration paths — its value here is the resize convention it
documents, which is what the border fix above brings this project in line with.

---

### Upstream releases with no dedicated entry

Seven `roboflow/rf-detr` releases have no alignment entry above. Six needed none; **1.8.2
did**, and the gap is still open — see the keypoint row in Known Issues below.

| Upstream | Why no entry |
|----------|--------------|
| [1.5.0](https://github.com/roboflow/rf-detr/releases/tag/1.5.0), [1.5.1](https://github.com/roboflow/rf-detr/releases/tag/1.5.1), [1.5.2](https://github.com/roboflow/rf-detr/releases/tag/1.5.2), [1.6.1](https://github.com/roboflow/rf-detr/releases/tag/1.6.1) | Custom augmentations, nested transforms, GPU-memory reporting, checkpointing — training-side, no export or runtime contract change. Predate the per-release habit; `v0.1.2` jumped 1.4.3 → 1.6.5.post0 in one hop and summarized 1.6.0–1.6.5 as bullets. |
| [1.7.1](https://github.com/roboflow/rf-detr/releases/tag/1.7.1) | BF16 segmentation training crash, `from_checkpoint` on starter weights, a NumPy 2.x import shim. Python-side only. Landed 2026-05-28, one day before `v0.1.3` shipped against 1.7.0. |
| [1.8.1](https://github.com/roboflow/rf-detr/releases/tag/1.8.1) | Albumentations/TensorBoard compatibility, metric plotting, checkpoint selection — training-side. One caveat for this project: [#1135](https://github.com/roboflow/rf-detr/pull/1135) fixed keypoint query routing *in eval mode*, and export traces the eval graph, so a keypoint ONNX exported with 1.8.0 can carry the bug. Re-export keypoint models with 1.8.1 or later. |
| [1.8.2](https://github.com/roboflow/rf-detr/releases/tag/1.8.2) | **Not benign.** Default `num_keypoints_per_class` changed from background-first `[0, 17]` to active-first `[17]` ([#1160](https://github.com/roboflow/rf-detr/pull/1160)), shifting person from `class_id=1` to `class_id=0`. `Config::keypoint_counts` still defaults to `{0, 17}`. Also [#1155](https://github.com/roboflow/rf-detr/pull/1155): `spatial_shapes` is now built from symbolic Shape ops, removing the `ScatterND` node that made `trtexec` fail with `IScatterLayer cannot be used to compute a shape tensor` — ONNX files exported before 1.8.2 may not build a TensorRT engine, and re-exporting is the fix. |

The 1.8.2 keypoint schema change was missed because 1.8.1 and 1.8.2 were skipped between
`v0.2.0` (1.8.0) and `v0.2.1`/`v0.2.2` (1.8.3), which backported the box-clamping fix without
re-reading the two releases in between. The `rfdetr-alignment` skill now requires saying
explicitly when a release needs no C++ change, so from 1.9.2 onward "considered and dismissed"
is on the record; these predate it.

---

### GPU gate: first execution of `run_gate.sh` against real hardware

The [gpu-verify](.claude/skills/gpu-verify/SKILL.md) gate had never been run — the script's own
header said so ("no execution history against real hardware"). Running it on a local RTX 3060
Laptop, following [docs/rented-gpu-runbook.md](docs/rented-gpu-runbook.md), took six fixes before
it produced a usable result. Two of them are bugs anyone would have hit; four are in the script.

#### Result

Hardware and versions from `gate-results/environment.txt`, per gate step 7:

| | |
|---|---|
| GPU / driver | NVIDIA GeForce RTX 3060 Laptop, 6 GB, driver 580.173.02 |
| CUDA | `nvcc` 12.0.140 (Ubuntu `nvidia-cuda-toolkit`); driver reports CUDA 13.0 |
| TensorRT | 10.13.3.9, `Linux.x86_64-gnu.cuda-12.9` tarball via `TENSORRT_ROOTDIR` |
| DALI | staged from `nvcr.io/nvidia/tritonserver:25.12-py3` |
| Model | `rfdetr-seg-medium.onnx`, rfdetr 1.9.4, 432×432 |
| `CMAKE_CUDA_ARCHITECTURES` | 86 |

**9 PASS, 0 FAIL, 5 UNRUN.** Passing: the full TensorRT + DALI + CUDA build, `USE_DALI=ON` alone,
`USE_CUDA_POSTPROCESS=ON` alone, both configure-time guards rejecting the ONNX Runtime combination,
all four pre/post combinations as smoke runs, and benchmarks.

Unrun, stated plainly as unrun rather than implied to have passed:

- The parity tolerances (tensor 2e-2, scores 1e-3, box centres 1 px, mask IoU 0.999) — no
  `tests/data/gpu_parity/`, roadmap Phase 2.
- The dense >100-detection fixture — roadmap Phase 2.
- The per-stage four-combination benchmark — `bench_gpu_pipeline.cpp` does not exist, Phase 2.
- **`compute-sanitizer` over 1000 frames** — the box's sanitizer (CUDA 12.0, Jan 2023) cannot
  instrument the app. The `daliOutputRelease` ordering check the pipeline spec calls for is
  therefore still unperformed; a passing smoke run is not a substitute for it.
- Gate step 6 on the GPU box — run locally instead, where it passed.

Also unrun and not attempted: `-DWERROR=ON`. See Known Issues.

#### What the four combinations showed

Not a parity check — one image, final scores rather than the preprocessed tensor — but the first
real signal on the question Phase 2 exists to answer:

| Combination | bicycle | dog | car | motorbike |
|---|---|---|---|---|
| cpu-cpu | 0.952574 | 0.891811 | 0.816406 | 0.580352 |
| cpupre-gpupost | 0.952574 | 0.891811 | 0.816406 | 0.580352 |
| gpupre-cpupost | 0.956145 | 0.888372 | 0.800068 | 0.571767 |
| gpupre-gpupost | 0.956145 | 0.888372 | 0.800068 | 0.571767 |

They split by **preprocessing**, not postprocessing. CUDA postprocessing is bit-identical to the CPU
path on real hardware — the `src/gpu/` correctness rule holding outside `test_gpu_postprocess.cpp`
for the first time. DALI preprocessing is not: max score delta 0.0163 on `car`, **16× the gate's
1e-3 score tolerance**. Phase 2 should be written expecting to find a discrepancy, not to confirm
its absence.

#### Fixed

| File | Change |
|------|--------|
| `cmake/deps/packages/TensorRT.cmake` | The pinned download URL was a hard 404: `...Linux.x86_64.gnu.cuda-13.0.tar.gz`, where NVIDIA's path is `x86_64-gnu`. This blocked `-DUSE_TENSORRT=ON` from a clean tree for everyone, including the runbook's own provisioning script, which installs only the CUDA toolkit and lets CMake fetch TensorRT. Undetected because CI never builds this backend. |
| `scripts/run_gate.sh` | `arm_watchdog()` falls back to `sudo shutdown -h "+N"` when no `brev` CLI is present, so running the gate on a local machine would halt it — `SELF_STOP` does not cover that path. New `WATCHDOG` variable (default `1`, rented behaviour unchanged) skips arming it. |
| `scripts/run_gate.sh` | Exports the TensorRT lib directory on `LD_LIBRARY_PATH`. `libnvinfer` dlopens `libnvinfer_builder_resource.so` at engine-build time, and a dlopen from inside a dependency does not use the executable's `RUNPATH`, so every combination died with `Unable to load library` before building an engine. Covers both an existing prefix and the tarball `cmake/deps` downloads into `build-gpu/_deps`. |
| `scripts/run_gate.sh` | `probe_tensorrt()` read `$3` of `#define NV_TENSORRT_MAJOR`, which in TensorRT 10.x expands to `TRT_MAJOR_ENTERPRISE`, not a literal — so `environment.txt`, the file step 7 says to copy into this changelog, recorded `TensorRT TRT_MAJOR_ENTERPRISE.TRT_MINOR_ENTERPRISE...`. Now resolves one level of indirection. |
| `scripts/run_gate.sh` | New `EXTRA_CMAKE_ARGS`, appended to the four TensorRT configures and deliberately not to the ONNX Runtime one. A box with its own TensorRT needs `-DTENSORRT_ROOTDIR=<prefix>`, which `ProvidedPackageManager.cmake` reads as a CMake variable, not an environment variable. `TRT_PREFIX` derives one prefix from either source for both the loader path and the version probe. |
| `scripts/run_gate.sh` | `step_sanitizer()` treated "no `ERROR SUMMARY: 0` line" as findings, so a sanitizer that never attached was reported `FAIL` — the same column as a real memory error. A tool that could not instrument the app is now `UNRUN`. |

#### Not fixed

`environment.txt` also showed the app's `RUNPATH` carrying raw link items —
`/usr/lib/x86_64-linux-gnu/libcudart_static.a:Threads::Threads:dl:` — which means unresolved link
targets are reaching the RPATH computation in the TensorRT build. Harmless here, recorded rather
than chased.

---

### Tooling: `scripts/run_gate.sh`, an unattended driver for the gpu-verify gate

The [gpu-verify](.claude/skills/gpu-verify/SKILL.md) checklist is the only thing standing between
the TensorRT, DALI and CUDA paths and an unverified release, and running it means renting a GPU by
the hour. Two things made that awkward in practice: the checklist is a document, so every run was
hand-driven from an SSH session that dies when the laptop closes, and a hung or finished run keeps
billing until someone notices. This turns the executable part into one script.

What it does not do is the point. `tests/data/gpu_parity/` does not exist and `src/main.cpp` has no
output-path flag — roadmap Phases 2 and 1 respectively — so the numeric tolerances the gate is
actually built around (preprocessed tensor `max |Δ| ≤ 2e-2`, scores within `1e-3`, box centres
within 1 px, mask IoU ≥ 0.999) have nothing to run against. The script reports those as `UNRUN`
rather than skipping them silently, and the four combinations it *can* run are labelled smoke
tests, not parity checks. That follows the skill's own rule: an unrun check is reported as unrun,
never implied to have passed. Running this script is not passing the gate; Phase 2 still gates
Phase 4.

| File | Change |
|------|--------|
| `scripts/run_gate.sh` | New. Drives step 1 (full TensorRT+DALI+CUDA build, both halves independently, and the two `USE_DALI`/`USE_CUDA_POSTPROCESS` + ONNX Runtime configure guards that must `FATAL_ERROR`), step 2 as smoke runs, step 4 (`compute-sanitizer` memcheck over a long video), step 5 (benchmarks), and step 6 (default ONNX Runtime build + UnitTests). Writes a `PASS`/`FAIL`/`UNRUN` summary plus per-check output into `~/gate-results/` for the CHANGELOG entry step 7 asks for. Not `set -e`: a failing check is data, so the remaining checks still run. |
| `scripts/run_gate.sh` | Cost control, because the gate runs on metered hardware. A `systemd-run` deadline watchdog fires `brev stop` after `DEADLINE_HOURS` whether the run finished, crashed, or hung — the self-stop at the end only covers the clean path. `SKIP_DEFAULT_PATH=1` moves step 6's default ONNX Runtime build, which needs no GPU, back to a local machine and off the metered clock. `CUDA_ARCH` defaults to `89` (L4/L40S/RTX-Ada) rather than the CMake default `86`, since the arch is a property of the rented box, not the project. |
| `docs/rented-gpu-runbook.md` | New. The operational half the skill deliberately leaves out: how to choose an instance, what to prepare at home before the meter starts, the provisioning setup script, running under `tmux`, and collecting results. Leads with what a rented hour actually buys today — steps 1, 2 (smoke), 4 and 5 — so nobody reads a green summary as a passed gate. Records the selection rule the hard way round: this workload is build-bound, so rank instances by CPU count and provisioning time, not VRAM, and avoid `highcpu`-class families whose ~0.9 GB per vCPU will OOM-kill a parallel C++ build on a swapless cloud VM. |
| `AGENTS.md`, `.claude/skills/gpu-verify/SKILL.md` | Pointers to the script and the runbook, alongside the existing `fetch_dali.sh` and `generate_dali_pipelines.sh` entries. |

The version probes behind step 7 are the part most likely to be quietly useless, so they were
written against how this project actually resolves its dependencies rather than against convention.
TensorRT is normally *not* a system package here — `cmake/deps/packages/TensorRT.cmake` downloads
the pinned tarball into `${CMAKE_BINARY_DIR}/_deps` — so a probe reading `/usr/include` finds
nothing on a correct build. `probe_tensorrt` searches the build trees, then `TENSORRT_ROOTDIR`,
then the system paths, and reads the version from the `NV_TENSORRT_*` macros rather than parsing
the extracted directory name, which can drift from the pin. It runs *after* step 1, since that
configure is what performs the download. DALI ships no version header in the extracted wheel, so
`probe_dali` records the provenance that does identify it: the Triton image pinned in
`fetch_dali.sh`, read out of that script so the two cannot disagree. Both probes were exercised
against a synthetic tree, present and absent.

Two checks are untestable from the script by construction and say so: the `GTEST_SKIP()`-without-a-
device behaviour needs a machine with no CUDA device, and the guest-shutdown watchdog fallback
needs one manual confirmation on the provider console that a halted VM bills as *stopped* rather
than billed-but-off.

Three review findings on the gate driver were fixed before it ran again; all three would have shown
up as a green summary hiding an unrun check, which is the one failure mode a gate cannot have.

| File | Change |
|------|--------|
| `scripts/run_gate.sh` | The `compute-sanitizer` step now requires the run to *finish*. It discarded the exit status and judged on the summary line alone, so an app that died in the first second — a `MODEL` path that does not exist, a video the decoder rejects — still printed `ERROR SUMMARY: 0 errors` and was recorded `PASS` for a 1000-frame run that never happened. `MODEL` is now `-f`-checked here as it already was in step 2, the exit status is captured, and the not-instrumented branch is tested first so a toolkit/driver mismatch stays `UNRUN` rather than being reclassified `FAIL` by its non-zero status. |
| `scripts/run_gate.sh` | The zero-findings match is anchored to `ERROR SUMMARY: 0 errors`. The old pattern also matched `0 errors` as a substring, so `ERROR SUMMARY: 10 errors` — or 20, or 100 — passed. |
| `scripts/run_gate.sh` | `SKIP_DEFAULT_PATH=1` no longer suppresses the GPU build's UnitTests. The early return sat above them, so the documented way to move the CPU build off the metered clock also skipped `test_gpu_postprocess` — the one test that needs the rented device to run rather than `GTEST_SKIP()`, and the reason the script is on a GPU box at all. |
| `scripts/run_gate.sh` | Step 6 reports the bit-identical baseline comparison as `UNRUN`. A green default build and green UnitTests are not the checklist's first step-6 box, which asks for output bit-identical to the pre-change baseline; that needs an inference run and a baseline to diff against, and `src/main.cpp` still has no output-path flag (Phase 1). It was previously absorbed into the `PASS`. |
| `scripts/run_gate.sh` | `step_build` returns explicitly. Its status on the success path was whatever the trailing guard loop's `rm -rf` left, and `main()` reads it to decide whether steps 2–5 have a binary to run. |

No `specs/features/` directory: per [AGENTS.md](AGENTS.md), a spec is required for a roadmap phase,
a release, an rfdetr alignment, or a change to a path CI cannot execute. This adds a helper script
and touches none of `src/`, so it is CHANGELOG-only. No README change either — it alters no code,
build option, backend version, Docker image, or export package; `docs/rented-gpu-runbook.md` is
reached from `AGENTS.md` and the skill, which is where someone running the gate is already looking.

---

### Fixed: ONNX Runtime download ignored the target platform

`cmake/deps/packages/OnnxRuntime.cmake` hard-coded the `onnxruntime-linux-x64`
archive URL, extracted directory, and `.so` path, so a provided-download build on
any non-x86-64 host fetched the x86-64 archive: every source file compiled and the
link then failed on incompatible objects. The catalog entry now derives the
archive from `CMAKE_SYSTEM_NAME` and `CMAKE_SYSTEM_PROCESSOR`, covering the
official CPU packages for Linux (`x64`, `aarch64`) and Windows (`x64`, `arm64`)
with the right extension (`.tgz`/`.zip`) and link/runtime library names
(`libonnxruntime.so.<ver>` vs `onnxruntime.lib` + `onnxruntime.dll`).
Because the selection follows the *target* processor, cross-compilation resolves
the target's archive rather than the host's. An unsupported target now fails at
configure time with a message pointing at `ONNXRUNTIME_ROOTDIR`, conan, or vcpkg,
instead of producing a link error at the end of a full build.

| File | Change |
|------|--------|
| `cmake/deps/packages/OnnxRuntime.cmake` | OS/architecture-derived archive, extension, and library paths; `FATAL_ERROR` on unsupported targets. |
| `Dockerfile` | ONNX Runtime staging glob is architecture-neutral (`onnxruntime-linux-*`), so the `onnx` image builds on arm64. |
| `README.md` | Documents the per-target archive table and the supported-platform behaviour. |
| `docs/package-manager-architecture.md` | Catalog example matches the computed declaration and notes that entries may compute values from the target platform. |

---

### GPU pipeline: DALI preprocessing + CUDA segmentation postprocessing

Opt-in GPU pipeline on the TensorRT backend, hosting DALI in-process through the
DALI C API and launching custom CUDA kernels on the same stream as the TensorRT
execution context — no Triton server. Both halves are compile-time optional
(`-DUSE_DALI`, `-DUSE_CUDA_POSTPROCESS`, or `-DUSE_GPU_PIPELINE` for both) and
runtime opt-in (`--gpu-preprocess`, `--gpu-postprocess`,
`--dali-pipeline-dir <dir>`); the CPU paths stay the default, and the ONNX
Runtime build is untouched. Design and phase plan: `docs/GPU_PIPELINE_ROADMAP.md`
at the time, since folded into `specs/roadmap.md`.

#### Added

| File | Change |
|------|--------|
| `src/gpu/gpu_context.{hpp,cpp}` | RAII device/stream owner + `CUDA_CHECK`; opaque stream handle keeps CUDA types out of non-CUDA translation units. |
| `src/gpu/dali_preprocessor.{hpp,cpp}` | RAII wrapper over the DALI C API: deserializes the `.dali` pipelines, feeds encoded bytes (still images) or device BGR frames (video), hands the output device pointer to the backend. |
| `src/gpu/rfdetr_postprocess.{hpp,cu}` | CUDA segmentation postprocess: sigmoid score decode, CUB top-k, box decode, per-instance mask resize + threshold, single packed D2H (count/boxes/scores/classes/mask_offsets/mask_data). |
| `src/backends/inference_backend.{hpp,cpp}` | Optional device-side I/O virtuals: `supports_device_io()`, `run_inference_device()`, `get_input_device_ptr()`, `get_output_device_ptr()`, `device_stream()`, `synchronize_device()` — default-false/throwing so ONNX Runtime is unaffected. |
| `src/backends/tensorrt_backend.{hpp,cpp}` | Implements device-side I/O; inference moves off the default CUDA stream onto a dedicated context stream. |
| `src/rfdetr_inference.{hpp,cpp}` | `run_gpu_image()`, `run_gpu_frame()`, `postprocess_segmentation_outputs_gpu()`, `fetch_device_outputs()`, `gpu_pre/postprocess_active()`; new `Config` fields `gpu_preprocess`, `gpu_postprocess`, `dali_pipeline_dir`, `gpu_device_id`. |
| `src/video_pipeline.cpp` | GPU-preprocess mode: preprocess stage becomes a passthrough; DALI runs on the backend stream in the inference stage. |
| `deploy/dali/generate_preprocess_pipeline.py` | Serializes the `encoded` (nvJPEG decode) and `frame` (BGR→RGB) DALI pipelines; documents the deliberate divergences from letterboxed YOLO pipelines (no `fn.paste`, `antialias=False`, ImageNet mean/std folded ×255). |
| `data/dali/preprocess_{encoded,frame}_{432,576}.dali` | Pre-generated pipelines for the 432 and 576 model resolutions. |
| `scripts/fetch_dali.sh` | Stages DALI C++ libraries/headers from the pinned `nvcr.io/nvidia/tritonserver:25.12-py3` container into `-DDALI_ROOT`. |
| `scripts/generate_dali_pipelines.sh` | Regenerates the `.dali` files for a given resolution inside the same pinned container (no local DALI pip install). |
| `cmake/deps/packages/CUDAToolkit.cmake` | Facade entry: apt handler via `FindCUDAToolkit` (`CUDA::cudart` + header-only CUB). |
| `cmake/deps/packages/DALI.cmake` | Facade entry: ROOT-only acquisition (`-DDALI_ROOT`), links `libdali.so` + `libdali_operators.so`. |
| `tests/unit/test_gpu_postprocess.cpp` | CPU-versus-GPU parity gate for the segmentation postprocessor; synthetic tensors via mock backend, `GTEST_SKIP()` when no CUDA device is present. |
| `CMakeLists.txt` | `USE_DALI` / `USE_CUDA_POSTPROCESS` / `USE_GPU_PIPELINE` options (TensorRT-only, `FATAL_ERROR` otherwise); `enable_language(CUDA)` and `CMAKE_CUDA_ARCHITECTURES` (default `86`) only when needed; C++-only warning flags scoped with `$<COMPILE_LANGUAGE:CXX>`; DALI RPATH alongside TensorRT. |

#### Why

The segmentation mask resize was the dominant CPU cost: single-threaded bilinear
resampling of one full-resolution mask per detection, after a ~14 MB D2H of mask
data that is mostly discarded. The GPU path keeps TensorRT outputs on the device,
postprocesses in place, and transfers only the packed final results. DALI
preprocessing removes the float tensor H2D (still images upload only the
compressed bytes) and frees the video pipeline's preprocess thread.

---

### CLI: inference parameters overridable without recompiling

`src/main.cpp` previously hard-coded `resolution`, `max_detections`, and
`mask_threshold` over the `Config` defaults, so tuning any of them — or the
`Config` defaults themselves — meant editing and rebuilding.

| File | Change |
|------|--------|
| `src/main.cpp` | Added `--resolution <px>`, `--max-detections <n>`, and `--mask-threshold <val>` alongside the existing `--threshold`. Options are held as `std::optional` and applied only when passed, so unset fields keep their `Config` defaults instead of being overwritten. Numeric arguments are parsed through `parse_int_option()`/`parse_float_option()`, which report the offending flag and value; previously a typo'd `--threshold` value let `std::stof` throw outside the `try` block and terminate the process. Values are range-checked (`--threshold` in `[0, 1]`, `--resolution`/`--max-detections` positive). |
| `README.md` | New "Tuning Flags" table under Usage; the Configuration section now maps each `Config` field to its CLI override (or notes that it has none) instead of describing CLI-settable fields as source-edit only. |

---

### Documentation

| File | Change |
|------|--------|
| `docs/backend-parity-segmentation-video.md` | Record of a manual cross-backend instance-segmentation video test (ONNX Runtime, TensorRT, ExecuTorch) over a 320-frame 1080p clip. All three run the same `RFDETRSegMedium` at 432×432 and agree on detections; ExecuTorch and ONNX Runtime match exactly, TensorRT is ~0.01 lower on four of seven scores from engine precision. Covers the manual gap left by CI, which tests neither TensorRT nor ExecuTorch. |

---

### Known Issues

| Area | Issue |
|------|-------|
| `src/backends/tensorrt_backend.cpp` | **Does not compile under `-DWERROR=ON`**, which is what the gpu-verify gate builds with — so gate step 1 fails outright and every later step is skipped. Nine errors on TensorRT 10.13.3.9: `-Wsign-conversion` at lines 148, 161, 257, 267 and 271; `-Wunused-parameter` on `input_shape` at 171; and `-Wdeprecated-declarations` for `NetworkDefinitionCreationFlag::kEXPLICIT_BATCH` (180), `IBuilder::platformHasFastFp16()` (223) and `BuilderFlag::kFP16` (224). The conversions and the unused parameter are mechanical; the deprecations are a real API migration — `kEXPLICIT_BATCH` is a no-op in TensorRT 10 and the FP16 flags are superseded by strongly-typed networks — so per `AGENTS.md` this needs a spec directory before it is touched. CI compiles only the ONNX Runtime lane, which is why strict warnings never reached this file. The gate results above were obtained with `-DWERROR=OFF`. |
| `src/rfdetr_inference.hpp`, `src/main.cpp` | **Keypoint models exported with `rfdetr` 1.8.2 or later do not decode.** Upstream 1.8.2 changed the default keypoint schema from background-first `[0, 17]` to active-first `[17]` ([#1160](https://github.com/roboflow/rf-detr/pull/1160)); `Config::keypoint_counts` still defaults to `{0, 17}` and there is no CLI override, so the schema can only be changed by editing `Config` and rebuilding. `deploy/requirements.txt` pins `rfdetr[onnx]==1.10.0`, so `deploy/export_keypoint.py` as documented produces the new schema. Expected effect, derived from the release notes and the code rather than observed against an export: the `keypoints` tensor carries one class instead of two, so `postprocess_keypoint_outputs()` raises `Keypoint tensor channels (17) not divisible by number of keypoint classes (2)`; if the `labels` tensor also drops to a single column, `background_class_id`'s default `0` leaves no foreground column and every frame yields zero detections (guarded, not undefined — `build_foreground_scores()` returns empty for a non-positive foreground count). Not yet verified against a real 1.8.2+ keypoint export, and not yet fixed: the choice between a `--keypoint-counts` flag and auto-detecting the schema from the tensor shape needs a model to test against. Pre-1.8.2 exports are unaffected. |
| `deploy/export_executorch.py` | Cannot export segmentation models: `--model_type` offers only the detection classes and the script instantiates `RFDETRNano`…`RFDETR2XLarge`, never `RFDETRSeg*`. Not an upstream or runtime limitation — `rfdetr` 1.9.0 exports `RFDETRSegMedium` to `.pte` without error, `ExecuTorchBackend::validate_output_order()` inspects only outputs 0 and 1 so a third `masks` output passes, and `postprocess_segmentation_outputs()` addresses outputs positionally. Segmentation `.pte` files must currently be exported by hand; see [docs/backend-parity-segmentation-video.md](docs/backend-parity-segmentation-video.md). |
| `src/main.cpp` | Video output is hard-coded to `output_video.mp4` in the current working directory with no override flag, so comparing backends requires running each from its own directory. |
| `.gitignore` | `*.pte` is not ignored (unlike `*.onnx` and `*.engine`), and `*.mp4` is ignored only as the exact root-level `output_video.mp4`. Exported ExecuTorch models and result videos appear as untracked files. |

---

## [v0.4.0] - 2026-08-04

Third inference backend and a unified dependency-resolution layer. **ExecuTorch**
joins ONNX Runtime and TensorRT, running `.pte` programs exported by `rfdetr`
1.9.0+, and the export pin moves to `rfdetr[onnx]==1.9.0`. Dependency resolution
moves behind a single `find_dependency_unified()` facade covering apt, conan,
vcpkg, and provided-download strategies.

**Breaking**: enabling more than one inference backend is now a configure-time
error. Previously two backends configured successfully and silently used ONNX
Runtime.

### RF-DETR 1.9.0 alignment + ExecuTorch backend

**Upstream release**: https://github.com/roboflow/rf-detr/releases/tag/1.9.0

#### Added

| File | Change |
|------|--------|
| `src/backends/executorch_backend.{hpp,cpp}` | Third inference backend, running `.pte` programs via `executorch::extension::Module`. Input/output shapes come from `method_meta("forward")`, so resolution auto-detection and the output-count check both work before the first inference. Input tensors are non-owning views over the caller's buffer (no per-frame copy); outputs are restricted to float32. |
| `cmake/deps/packages/ExecuTorch.cmake` | Catalog entry: `find_package(executorch CONFIG)` against an install prefix first, FetchContent source build of `v1.3.1` as fallback. Pinned to `v1.3.1` to match the ExecuTorch version `rfdetr[executorch]==1.9.0` resolves for the Python exporter — `.pte` schema compatibility across runtime versions is not guaranteed. |
| `cmake/deps/{PackageManager,strategies/ProvidedPackageManager}.cmake` | New `PROVIDED_FC_OPTIONS` / `PROVIDED_FC_SUBMODULES` declaration keys so a FetchContent dependency can have build options seeded into the cache and submodules fetched recursively — both required to build ExecuTorch from source. |
| `CMakeLists.txt` | `USE_EXECUTORCH` and `EXECUTORCH_DELEGATE` (`xnnpack`/`portable`) options; `EXECUTORCH_ROOTDIR` appended to `CMAKE_PREFIX_PATH`; delegate library linked so it self-registers. |
| `deploy/export_executorch.py` | Export script for `.pte` models: `--backend xnnpack/coreml/qnn`, `--soc` required for `qnn`. Warns both before and after export that only `xnnpack` yields a `.pte` this project can run — `-DEXECUTORCH_DELEGATE` offers only `xnnpack`/`portable`, so a coreml/qnn program fails at load with `Backend ... is not registered`. |
| `tests/unit/test_rfdetr_inference.cpp` | `PreprocessFrame.ResizeIsAntialiasFree` — asserts `preprocess_bgr_image` point-samples the bilinear 2x2 footprint rather than averaging over the source area. |
| `docs/export.md` | ExecuTorch export section; native TensorRT export section. |
| `Dockerfile` | `INFERENCE_BACKEND=executorch` variant. Builds the ExecuTorch C++ runtime from source into `/opt/executorch` (tag via `EXECUTORCH_VERSION`, default `v1.3.1`), applying the upstream `extension_evalue_util` install fix. Uses a throwaway venv with the CPU `torch` wheel because ExecuTorch's operator codegen does `import torchgen`, and installs `build-essential` because ExecuTorch's bundled `flatcc` is an ExternalProject that configures with the Unix Makefiles generator and does not inherit `CMAKE_C_COMPILER`. Static-linked, so the runtime stage ships no extra shared libraries and needs no GPU. The ExecuTorch step is placed **before** `COPY . .` so editing a `.cpp` reuses the cached layer (36s rebuild) instead of recompiling ExecuTorch (~11 min). |

#### Changed

| File | Change |
|------|--------|
| `deploy/requirements.txt` | `rfdetr[onnx]` 1.8.3 → 1.9.0. |
| `docs/export.md` | Version bumps to 1.9.0. Noted the relaxed `onnxsim>=0.7.0` pin, the non-native-resolution export fix, and the in-process (polygraphy) TensorRT export path. Pinned the `rfdetr[executorch]` / `rfdetr[tensorrt]` install commands to `==1.9.0` — the extras do not constrain ExecuTorch itself, and an unpinned install can pull an exporter whose `.pte` the pinned v1.3.1 runtime cannot load. |
| `README.md` (accuracy) | ONNX Runtime dependency section no longer claims Windows/macOS and CUDA/DirectML support: the default download is the Linux x64 **CPU** archive, and `OnnxRuntimeBackend` registers no execution provider, so even a GPU build would run on CPU. Other platforms need `-DONNXRUNTIME_ROOTDIR` or the conan/vcpkg coordinates. CMake requirement notes the **3.17+** needed by the ExecuTorch FetchContent fallback (`GIT_SUBMODULES_RECURSE`). |
| `README.md` | Three-backend selection table with model formats; ExecuTorch build section, dependencies, and build options; `rfdetr[executorch]` / `rfdetr[tensorrt]` export extras; version statements to 1.9.0. Fixed the media-backend example `-DUSE_TENSORRT=ON -DUSE_OPENCV=ON`, which the new exactly-one-backend check turned into a configure error (`USE_ONNX_RUNTIME` defaults `ON`). Documented TensorRT's `.onnx` acceptance and per-backend integration-test discovery. |
| `AGENTS.md` | Backend selection covers ExecuTorch; CI-coverage note extended; new Docker section. |
| `docs/package-manager-architecture.md` | ExecuTorch in the catalog listing and dependency-mapping table; documented the new `PROVIDED_FC_OPTIONS` / `PROVIDED_FC_SUBMODULES` keys and the "install-prefix first, source build as fallback" pattern they enable. |
| `docs/glossary.md` | Backend selection is three-way; added ExecuTorch, `.pte`, Delegate, and XNNPACK entries. |
| `.github/copilot-instructions.md` | Backend list covers ExecuTorch. |
| `CMakeLists.txt` | **Behavior change**: enabling more than one backend is now a configure-time error. Previously `-DUSE_ONNX_RUNTIME=ON -DUSE_TENSORRT=ON` configured successfully and silently used ONNX Runtime, because `create_backend()` resolves via an `#ifdef`/`#elif` chain. With a third backend that silent fallthrough became too easy to hit unnoticed. |
| `tests/integration/integration_test_rfdetr_inference.cpp` | Skip message 1.8.3 → 1.9.0. Model discovery now probes the extensions the **compiled-in** backend can load instead of a hardcoded `.onnx`, and the skip message names the expected format. ONNX Runtime probes `.onnx`; ExecuTorch `.pte`; TensorRT `.engine`, `.trt`, then `.onnx` — the last kept deliberately, since `TensorRTBackend::initialize()` builds/caches an engine from ONNX and dropping it would lose that path from coverage. Extension is the outer loop so format preference is global: a prebuilt engine in any search directory beats an ONNX file in an earlier one, since loading an engine is instant while building one costs minutes. Also fixes a pre-existing TensorRT bug: a TRT build previously found a stray `.onnx` before the `.engine` beside it. |
| `src/main.cpp` | Usage text is backend-aware: the example model extension, the backend description, and the rebuild flags are selected at compile time, instead of claiming only ONNX Runtime and TensorRT exist. |
| `.gitignore` | Ignore `data/test_*` and `data/empty_labels.txt` — all six fixtures the detection and keypoint integration fixtures generate (`test_image.jpg`, `test_labels.txt`, `test_output.jpg`, `test_kp_image.jpg`, `test_kp_output.jpg`, `empty_labels.txt`), which are normally removed in `TearDown` but survive an interrupted run. |

#### Why no postprocessing port or model re-export

Diffing `1.8.3...1.9.0` over the code this project mirrors:

- `export/_onnx/exporter.py` (+201/-97) and `models/lwdetr.py` (+78/-57) are **typing-only** — protocols, `cast`, f-strings. Graph structure, output names, ordering, and opset are unchanged, so existing exported models stay compatible and **no re-export is required**.
- `models/postprocess.py` (+63/-18) chunks mask upsampling to cut peak GPU memory (identical results — still bilinear to target size, then `> 0.0`) and adds `upsample_masks_to_image_size`, an opt-in validation-only flag. `media.cpp::resize_threshold_mask` already matches the default path.
- **PR #1206 confirms this project's preprocessing.** Upstream `predict()` had been resizing with antialias enabled, drifting from training; 1.9.0 sets `antialias=False` to match the antialias-free bilinear (`cv2.INTER_LINEAR`) resize used during training. `media.cpp::preprocess_bgr_image` has always been exactly that, so 1.8.3's `predict()` was the side that disagreed, and 1.9.0 closes the gap. Locked in by `PreprocessFrame.ResizeIsAntialiasFree`.

#### End-to-end verification

The ExecuTorch backend was run end-to-end against a real `.pte` before this change was committed. `rf-detr-nano.pth` was exported at 384×384 to both `.onnx` and `.pte` (`format="executorch", backend="xnnpack"`) from the same checkpoint with rfdetr 1.9.0, then run through `inference_app` on both backends:

| Check | Result |
|-------|--------|
| Detections at threshold 0.5 | 3 vs 3 — same classes, same order |
| Detections at threshold 0.05 | 34 vs 34 — same classes, same order |
| Max box delta | 1e-4 px |
| Max score delta | 1e-6 |
| Output-order guard | `method_meta` reports `[1,300,4]` boxes at index 0, `[1,300,91]` logits at index 1 |
| Delegate mismatch (`portable` build vs xnnpack `.pte`) | Fails with `Backend XnnpackBackend is not registered`, exit 1 |
| Integration tests with `RFDETR_TEST_MODEL=<.pte>` | 4 passed, 1 skipped (no keypoint `.pte`) |

Regression status: ONNX Runtime 39/39, TensorRT 39/39, ExecuTorch 39/39; `clang-format` clean; `-DWERROR=ON` clean on the ExecuTorch configuration.

#### Upstream ExecuTorch packaging bug (affects install-prefix builds)

ExecuTorch v1.3.1 `extension/evalue_util/CMakeLists.txt:27` installs its target with
`DESTINATION ${CMAKE_BINARY_DIR}/lib` instead of `${CMAKE_INSTALL_LIBDIR}` — every other
extension uses the latter. Consequences for a `cmake --install`ed prefix:

- `libextension_evalue_util.a` is never copied into `<prefix>/lib`.
- `ExecuTorchTargets-release.cmake` records an absolute **build-tree** path for that one
  target (18 of 19 targets are `${_IMPORT_PREFIX}`-relative; this one is not).
- `find_package(executorch CONFIG)` then hard-fails via its import-check loop as soon as
  the build tree is removed — even though this project never links that target.

Workaround when building ExecuTorch from source, applied before `cmake --install`:

```bash
sed -i 's|DESTINATION ${CMAKE_BINARY_DIR}/lib|DESTINATION ${CMAKE_INSTALL_LIBDIR}|' \
    extension/evalue_util/CMakeLists.txt
```

The FetchContent fallback path is unaffected: `FetchContent_MakeAvailable` consumes the
targets directly from the build tree and never runs the faulty `install()` rule.


### package-manager abstraction

Unified package-manager abstraction layer (`cmake/deps/`) wrapping apt, conan,
vcpkg, and provided-download strategies behind a single `find_dependency_unified()`
facade. All dependencies — including GTest, Google Benchmark, stb, and font8x8 —
now route through the facade instead of ad-hoc `FetchContent` or manual
`target_include_directories` calls.

#### Added

| File | Change |
|------|--------|
| `cmake/deps/Deps.cmake` | Facade: public options (`DEPS_MODE`, `DEPS_CONAN_DIR`, `DEPS_OFFLINE`, `DEPS_PROVIDED_DIR`, `DEPS_DEBUG`), module includes, catalog loader, resolver construction. |
| `cmake/deps/PackageManager.cmake` | Registry (`deps_declare`), chain-of-responsibility driver (`deps_resolve`), non-IMPORTED INTERFACE target builder (`deps_build_target` → `Deps::<Name>`), lockfile writer (`deps_lock_write` → `deps.lock.json`). |
| `cmake/deps/strategies/AptPackageManager.cmake` | apt strategy: `find_package` / `pkg_check_modules` QUIET probe. |
| `cmake/deps/strategies/ConanPackageManager.cmake` | Conan 2 strategy: toolchain mode (auto-detected) + CMakeDeps-only mode (`-DDEPS_CONAN_DIR`); `find_package(<CONAN_FIND> CONFIG)`. |
| `cmake/deps/strategies/VcpkgPackageManager.cmake` | vcpkg strategy: manifest mode (auto-detected via toolchain); CONFIG + MODULE mode fallback; prefers IMPORTED targets, falls back to `<FIND>_LIBRARIES` variables. |
| `cmake/deps/strategies/ProvidedPackageManager.cmake` | Provided strategy: `DOWNLOAD` (pinned URL), `VENDORED` (third_party/), `FETCHCONTENT` (git clone), `ROOT` (user path); shared `deps_provided_finalize`. |
| `cmake/deps/packages/*.cmake` | 10 catalog entries: OnnxRuntime, TensorRT, OpenCV, FFmpeg, SDL2, Threads, GTest, GoogleBenchmark, stb, font8x8 — each declares apt/conan/vcpkg/provided coordinates in one place. |
| `conanfile.txt` | `ffmpeg/6.1`, `sdl/2.28.5`, `gtest/1.12.1`; CMakeDeps + CMakeToolchain generators; OpenCV alternative documented. |
| `vcpkg.json` | `["ffmpeg", "sdl2", "gtest"]` manifest. |
| `.github/workflows/deps-modes.yml` | `workflow_dispatch` CI matrix testing apt/conan/vcpkg modes; conan uses gcc11 CMakeDeps-only mode with prebuilt binaries. |
| `docs/package-manager-architecture.md` | 118-line architecture reference: strategies, catalog format, dependency mapping table, options, lockfile, CMakeDeps-only mode. |
| `CMakeLists.txt` | `DEPS_MODE` cache option; all dependencies resolved via `find_dependency_unified()`; lockfile written at configure end. |

#### Changed

| File | Change |
|------|--------|
| `CMakeLists.txt` | GTest, Google Benchmark: raw `FetchContent` → facade (PROVIDED_ACQUIRE=FETCHCONTENT fallback); stb, font8x8: manual `target_include_directories` → facade (PROVIDED_ACQUIRE=VENDORED); `deps_get_rec()` used for RPATH/runtime libs. |
| `README.md` | New Dependency Resolution section with mode/chain table, conan CMakeDeps-only example, vcpkg manifest example, system-package notes. |
| `AGENTS.md` | Brief DEPS section: backend selection, dependency resolution, DEPS_DEBUG. |

#### Fixed

- **DEPS_OFFLINE honored for FETCHCONTENT**: `can_resolve` returns FALSE when offline, preventing network clone attempts in air-gapped builds (Codex review P2).
- **vcpkg MODULE mode fallback**: vcpkg's FFmpeg port ships a Find module (not a config file); the strategy now tries CONFIG first, then MODULE, and uses `<FIND>_LIBRARIES` variables when IMPORTED targets don't exist.
- **Chain order**: `conan;apt;provided` and `vcpkg;apt;provided` — apt fills in system packages (Threads) even in conan/vcpkg modes.
- **Conan "version conflict" resolved**: OpenCV and FFmpeg are mutually exclusive (`USE_OPENCV` option); each conan graph resolves independently.

#### Why

Decouples dependency acquisition from build logic. Previously, each dependency
had bespoke `find_package`/`FetchContent`/download code interleaved with build
targets in `CMakeLists.txt`. The facade centralizes all acquisition strategies
behind a single declarative interface (`find_dependency_unified(Name REQUIRED)`),
making it trivial to switch between apt (default, zero tooling), conan (prebuilt
binaries), and vcpkg (manifest mode) without touching build logic. The lockfile
records which handler + version resolved each dep for reproducibility.

Also includes expanded runtime-verification toolchain: ThreadSanitizer (data-race
detection) and Valgrind (memcheck correctness + callgrind/massif profiling),
wired into both CI and local CMake targets.

---

## [v0.3.0] - 2026-07-03

Swappable media/display backend and a unified Docker image matrix. The default
image/video I/O + display layer moves from OpenCV to **FFmpeg + SDL2 + stb**;
OpenCV remains selectable via `-DUSE_OPENCV=ON`, orthogonal to the inference
backend (ONNX Runtime / TensorRT).

### Changed

| File | Change |
|------|--------|
| `src/video_reader.*`, `src/video_writer.*`, `src/display.*`, `src/media.cpp` | Replaced the OpenCV media/display layer with FFmpeg (decode/encode), SDL2 (preview), and stb (image I/O). OpenCV implementations retained behind `#ifdef USE_OPENCV`. |
| `CMakeLists.txt` | Added `USE_OPENCV` option (default `OFF`); media backend selected between `pkg-config` FFmpeg+SDL2 and `find_package(OpenCV)`. |
| `Dockerfile` | Unified into a single parametric Dockerfile driven by `INFERENCE_BACKEND` (onnx\|tensorrt) and `MEDIA_BACKEND` (ffmpeg\|opencv), producing all 4 image variants. Reuses NGC-bundled TensorRT (skips the ~1GB tarball download). |
| `README.md` | Documented the media-backend choice, the `USE_OPENCV` build option, and the 4-variant Docker matrix; release badge bumped to `0.3.0`. |
| `.gitignore` | Ignore root-level media artifacts (`output_video.mp4`), review working docs (`hermes_review.md`), and the `Testing/` temp dir. |

### Removed

| File | Change |
|------|--------|
| `Dockerfile.onnx`, `Dockerfile.tensorrt` | Superseded by the unified parametric `Dockerfile`. |

### Fixed

- Stabilized media backend shutdown.
- `VideoWriter::write` return type changed `bool` → `void` (it never returned false; clears cppcheck `knownConditionTrueFalse`).
- CI cppcheck `missingInclude` and Dockerfile matrix build.

### Why

Decouples the media/display stack from OpenCV so the project can run with a
lighter, license-friendly dependency set by default, while keeping OpenCV as a
compile-time option. The Docker matrix makes all four backend combinations
first-class and independently testable. Backward-compatible: existing
`USE_OPENCV=ON` and `USE_TENSORRT=ON` builds are unchanged.

---

## [v0.2.2] - 2026-07-01

**Upstream release**: https://github.com/roboflow/rf-detr/releases/tag/1.8.3

### Changed

| File | Change |
|------|--------|
| `README.md` | Updated release badge, feature summary, Python export package listing to `rfdetr[onnx]==1.8.3`, TensorRT/CUDA version wording. |
| `AGENTS.md` | Added mandatory release documentation sync rule for README dependency, backend, build-option, Docker, and pip package changes. |
| `deploy/requirements.txt` | Bumped export package from `rfdetr[onnx]==1.8.0` to `rfdetr[onnx]==1.8.3`. |
| `docs/export.md` | Updated current RF-DETR export version guidance to `1.8.3`. |
| `tests/integration/integration_test_rfdetr_inference.cpp` | Updated missing-model skip guidance to `rfdetr 1.8.3`. |

### Why

Completes the upstream RF-DETR 1.8.3 release alignment by moving the Python export package and release-facing docs to `rfdetr[onnx]==1.8.3`. Future release work must verify README dependency/version statements against CMake, Docker, export docs, and pip requirements before shipping.

---

## [v0.2.1] - 2026-07-01

**Upstream release**: https://github.com/roboflow/rf-detr/releases/tag/1.8.3 (partial backport)

### Fixed

| File | Change |
|------|--------|
| `src/processing_utils.hpp` | New `clamp_box()` declaration. |
| `src/processing_utils.cpp` | Implement `clamp_box()` using `std::clamp` to constrain boxes to `[0, max_w] x [0, max_h]`. |
| `src/rfdetr_inference.cpp` | Apply `clamp_box()` after `scale_box()` in detection, segmentation, and keypoint postprocess paths. |
| `tests/unit/test_rfdetr_inference.cpp` | 4 `ClampBox` unit tests + 1 end-to-end `PostprocessTest.BoxesClampedToImageBounds` test. |

### Why

Ports the box-clamping fix from upstream rf-detr v1.8.3 (`PostProcess._postprocess_boxes()`, #1168). Predicted boxes are now guaranteed within `[0,width] x [0,height]`, so objects at image edges no longer produce negative `x1/y1` or out-of-frame `x2/y2` coordinates. Partial backport: the fp16 `scale_fct` dtype-cast is N/A in C++; training/export-side 1.8.3 changes do not affect postprocessing, so no model re-export is required.

---

## [v0.2.0] - 2026-06-17

**Upstream release**: https://github.com/roboflow/rf-detr/releases/tag/1.8.0

### Added

| File | Change |
|------|--------|
| `deploy/export_keypoint.py` | **New.** Keypoint model export script using `RFDETRKeypointPreview`. |
| `src/rfdetr_inference.hpp` | `ModelType::KEYPOINT`, `KeypointResult` struct, keypoint Config fields (`keypoint_counts`, `skeleton`, `keypoint_uncertainty_alpha`, `draw_uncertainty`, `keypoint_color`). Declarations for `postprocess_keypoint_outputs()`, `draw_keypoints()`, `get_label_name()`. |
| `src/rfdetr_inference.cpp` | Keypoint postprocessing pipeline: 3-output validation, background-offset class mapping, per-query class selection, bbox decode, image-relative keypoint coordinate decode, sigmoid findability/visibility, Precision Cholesky → pixel covariance, uncertainty-weighted score fusion. `draw_keypoints()`: boxes + labels, keypoint circles (radius ∝ findability), skeleton lines, optional uncertainty ellipses. `get_label_name()` helper. |
| `src/main.cpp` | `--keypoint` CLI flag, dispatch to keypoint postprocessing and drawing. Keypoint result output (coordinates, findability, visibility per keypoint). |
| `src/video_pipeline.hpp` | `keypoints` field in `FrameSlot`. |
| `src/video_pipeline.cpp` | `draw_keypoint_on_frame()` helper. KEYPOINT dispatch in `infer_postprocess_stage()` and `draw_write_stage()`. |
| `tests/unit/test_rfdetr_inference.cpp` | 7 keypoint postprocessing unit tests: 3-output validation, class selection + bbox decode, keypoint coordinate decode, scale factor application, no-detection threshold, Cholesky→covariance math, background column skipping. |
| `tests/integration/integration_test_rfdetr_inference.cpp` | Keypoint E2E integration test with `RFDETR_KEYPOINT_MODEL` env var. |
| `docs/export.md` | Keypoint model export section. |

### Changed

| File | Change |
|------|--------|
| `deploy/requirements.txt` | `rfdetr[onnx]` 1.7.0 → 1.8.0 |
| `deploy/export_detection.py` | Removed `--simplify` arg and deprecation warnings. |
| `deploy/export_segmentation.py` | Same as detection export script. |
| `docs/export.md` | Version bumps 1.7.0→1.8.0. Removed `--simplify` from options table. Removed TRT re-export note (1.7.0-specific). |
| `README.md` | Version bump, keypoint usage examples, keypoint model in ONNX download section. |
| `tests/integration/integration_test_rfdetr_inference.cpp` | Skip message 1.7.0→1.8.0. |
| `.github/workflows/ci.yml` | Add `develop` branch to push/PR triggers. |
| `README.md` | CI section updated to reflect `develop` branch. |

### Why

RF-DETR 1.8.0 introduces keypoint detection models via `RFDETRKeypointPreview`. The `--simplify` flag (deprecated since 1.7.0) is removed entirely in this release.

---

## [v0.1.3] - 2026-05-29

**Upstream release**: https://github.com/roboflow/rf-detr/releases/tag/1.7.0

### Changed

| File | Change |
|------|--------|
| `deploy/requirements.txt` | `rfdetr[onnx]` 1.6.5.post0 → 1.7.0 |
| `deploy/export_detection.py` | `--simplify` deprecated warning; print variant ONNX filename |
| `deploy/export_segmentation.py` | Same as detection export script |
| `deploy/inspect_model.py` | `RFDETRSegPreview` → `RFDETRSegMedium` |
| `docs/export.md` | Version bump, variant ONNX filenames, simplify deprecation, TRT re-export note |
| `README.md` | Version bump + variant ONNX filename examples + integration test model env var |
| `export_trt.sh` | Example path `inference_model.onnx` → `rfdetr-medium.onnx` |
| `Dockerfile.tensorrt` | NGC TRT Docker tag 25.09 → 25.12; variant ONNX examples in usage comments |
| `tests/integration/integration_test_rfdetr_inference.cpp` | Resolve model from `RFDETR_TEST_MODEL` or variant filenames; skip E2E when absent |

### Why

No C++ source changes required. ONNX tensor names and postprocessing unchanged.

Notable upstream changes between 1.6.5.post0 and 1.7.0:

1. **Variant ONNX filenames** (1.7.0) — exports named e.g. `rfdetr-medium.onnx`, `rfdetr-seg-medium.onnx` instead of generic `inference_model.onnx`.
2. **`simplify` deprecated** (1.7.0) — `model.export(simplify=True)` is a no-op; ONNX simplification no longer runs during export.
3. **ONNX/TRT dynamic batch fix** (#950) — re-export with 1.7.0 before building TensorRT engines if you rely on dynamic batch shapes.
4. **Import path deprecations** — `rfdetr.util.*` → `rfdetr.utilities.*`, `rfdetr.deploy.*` → `rfdetr.export.*` (removal in v1.8).
5. **Class deprecations** — `RFDETRBase`, `RFDETRSegPreview` emit warnings; use sized variant classes instead.

---

## [v0.1.2] - 2026-05-12

**Upstream release**: https://github.com/roboflow/rf-detr/releases/tag/1.6.5.post0

### Changed

| File | Change |
|------|--------|
| `deploy/requirements.txt` | `rfdetr[onnxexport]` 1.4.3 → `rfdetr[onnx]` 1.6.5.post0 |
| `deploy/export_detection.py` | Added `--device` flag to allow explicit device selection (e.g. `--device cpu`) |
| `deploy/export_segmentation.py` | Added `--device` flag (same as above) |
| `docs/export.md` | Extra rename note, version bump, TRT Docker tag 25.09 → 25.12 |
| `README.md` | Version bump + extra rename |
| `export_trt.sh` | NGC TRT Docker tag 25.09 → 25.12 |

### Why

The `onnxexport` optional extra was renamed to `onnx` in rfdetr 1.6.0. No C++ source changes required.

Notable upstream changes between 1.4.3 and 1.6.5.post0:

1. **Extra rename** (1.6.0) — `rfdetr[onnxexport]` → `rfdetr[onnx]`.
2. **Non-square inference shapes** (1.6.2) — `export()` and `predict()` now accept `(height, width)` tuples; ONNX model output format unchanged.
3. **Fine-tuned model export fix** (1.6.3) — `reinitialize_detection_head` now replaces `nn.Linear` instead of mutating weights, so custom-class-count models export correctly.
4. **`torch.export.export` fix** (1.6.4) — `spatial_shapes_hw` was not threaded through decoder layers; fixed for models using multi-scale deformable attention.
5. **PTL pin** (1.6.5.post0) — pins PyTorch Lightning ≤ 2.6.1.

---

## [v0.1.1] - 2026-02-17

**Upstream release**: https://github.com/roboflow/rf-detr/releases/tag/1.4.3

### Changed

| File | Change |
|------|--------|
| `deploy/requirements.txt` | `rfdetr[onnxexport]` 1.4.2 → 1.4.3 |
| `docs/export.md` | Version bump |
| `README.md` | Version bump |

### Why

Patch release with no model or API changes affecting C++ inference. Upstream changes:

1. **Segmentation export fix** — resolved `deploy_to_roboflow` issue for segmentation model export (#578).
2. **MD5 checksum validation** — added checksum verification for downloaded pretrained weights (#679).
3. **COCO benchmarks** — added segmentation model benchmarks and updated inference thresholds (#678, #684).

---

## [v0.1.0] - 2026-02-14

**Upstream release**: https://github.com/roboflow/rf-detr/releases/tag/1.4.2

### Changed

| File | Change |
|------|--------|
| `deploy/requirements.txt` | `rfdetr[onnxexport]` 1.3.0 → 1.4.2 |
| `deploy/export_detection.py` | Removed deprecated `RFDETRBase`, default `medium`, added `xlarge`/`2xlarge` |
| `deploy/export_segmentation.py` | Replaced `RFDETRSegPreview` with sized classes + added `--model_type` arg |
| `docs/export.md` | Version bump + updated Python API examples to new class names |
| `README.md` | Version bump, gcc alternative, minimal OpenCV install note, Ninja optional |
| `CMakeLists.txt` | `find_package(OpenCV)` now specifies required components (`core`, `imgproc`, `imgcodecs`) |
| `src/backends/onnx_runtime_backend.cpp` | Fixed `get_output_count()` returning 0 before inference |

### Why

1. **Seg ONNX export was broken** — upstream fixed it in #626. `RFDETRSegPreview` is gone, replaced by `RFDETRSegNano/Small/Medium/Large/XLarge/2XLarge`.
2. **`RFDETRBase` deprecated** — upstream no longer lists it. Use `RFDETRNano/Small/Medium/Large` instead.
3. **XL/2XL models added** — require `pip install rfdetr[plus]` (PML 1.0 license, not Apache).
4. **ONNX Runtime output count bug** — `get_output_count()` checked `ort_output_tensors_` which is only populated after `run_inference()`, but the constructor validates output count before that. Fixed to use `output_name_strings_` (populated during `initialize()`).

---

[Unreleased]: https://github.com/olibartfast/rf-detr-cpp-inference/compare/v0.4.0...develop
[v0.4.0]: https://github.com/olibartfast/rf-detr-cpp-inference/compare/v0.3.0...v0.4.0
[v0.3.0]: https://github.com/olibartfast/rf-detr-cpp-inference/compare/v0.2.2...v0.3.0
[v0.2.2]: https://github.com/olibartfast/rf-detr-cpp-inference/compare/v0.2.1...v0.2.2
[v0.2.1]: https://github.com/olibartfast/rf-detr-cpp-inference/compare/v0.2.0...v0.2.1
[v0.2.0]: https://github.com/olibartfast/rf-detr-cpp-inference/compare/v0.1.3...v0.2.0
[v0.1.3]: https://github.com/olibartfast/rf-detr-cpp-inference/compare/v0.1.2...v0.1.3
[v0.1.2]: https://github.com/olibartfast/rf-detr-cpp-inference/compare/v0.1.1...v0.1.2
[v0.1.1]: https://github.com/olibartfast/rf-detr-cpp-inference/compare/v0.1.0...v0.1.1
[v0.1.0]: https://github.com/olibartfast/rf-detr-cpp-inference/releases/tag/v0.1.0
