# CHANGELOG

All notable changes to this C++ inference project: its own features and fixes, and the
upstream `rfdetr` releases it is kept in step with.

---

## [Unreleased]

### Workflow: spec-driven development loop, four skills, roadmap split

Second pass over the agent workflow, reviewing the previous `specs/` change against the reference
project for the DeepLearning.AI spec-driven-development course
([sc-spec-driven-development-files](https://github.com/https-deeplearning-ai/sc-spec-driven-development-files)).
That workflow is: constitution → feature spec (interview first, then `requirements`/`plan`/
`validation`) → implement → validate → replan → changelog → merge. We already had the constitution
and are ahead of the reference on agent replaceability (`AGENTS.md` canonical, with `CLAUDE.md`,
`.clinerules`, `CONVENTIONS.md`, and `.github/copilot-instructions.md` as thin pointers) and on the
changelog, which is written by hand with reasoning and per-file tables rather than scraped from
`git log`. Four gaps were real: no completion state on roadmap phases, no acceptance criteria
written *before* the work, recurring rituals living as prose that is easy to skip, and a
`roadmap.md` that was three documents in one.

| File | Change |
|------|--------|
| `specs/gpu-pipeline.md` | New. The GPU architecture diagram, the 8-rule model contract, the packed output contract, correctness rules, and the risk table, moved verbatim out of `specs/roadmap.md`. These are standing constraints on `src/gpu/`, not planned work, and were burying the work queue. |
| `specs/roadmap.md` | Phase items are now `[ ]`/`[x]` checkboxes, so "the next phase" is the first section that is all unticked — mechanically findable rather than inferred from prose. Volatile status ("49 commits ahead of `master`") removed; git-flow model and the v0.5.0 gate kept. Phase content and ordering are unchanged: v0.5.0 still waits on Phases 1–4. |
| `specs/features/2026-08-15-gpu-parity-fixtures/` | New. `requirements.md`, `plan.md`, `validation.md` for roadmap Phase 2, seeding the convention with the phase that is actually next. Records one thing the roadmap missed: `tests/unit/test_gpu_postprocess.cpp` already covers segmentation postprocess parity including a dense case, so the new `test_gpu_parity.cpp` is for the uncovered half — DALI preprocess versus CPU preprocess on real image geometry. |
| `.claude/skills/feature-spec/SKILL.md` | New. Next unticked phase → branch from `develop` → three grouped questions before any file is written → the spec triple. Also states when a spec is *not* required, so small fixes stay CHANGELOG-only. |
| `.claude/skills/rfdetr-alignment/SKILL.md` | New. The standing obligation as a checklist: verify the release against upstream first, classify input/output/operator/API change, move CPU and CUDA postprocessing together, update the pins, and say explicitly when no C++ change is needed. |
| `.claude/skills/release/SKILL.md` | New. Spec Sync checks, `[Unreleased]` → `[vX.Y.Z]`, the git-flow cut, and the version reconciliation this release owes: `project()` declares none, `vcpkg.json` says `0.1.0`, the README badge says `0.4.0`. |
| `.claude/skills/gpu-verify/SKILL.md` | New. The gate CI cannot run: the build matrix, the four pre/post combinations with their numeric tolerances, the dense fixture, `compute-sanitizer` over 1000 frames, benchmarks including the flat ones, and confirmation that the default CPU path is bit-identical. |
| `AGENTS.md` | New "Workflow" section: the loop, the four skills, and when a feature spec is required. "Release Documentation Sync" widens into "Spec Sync" — every existing release rule kept verbatim, plus the reference's replanning discipline, that a `specs/mission.md` or `specs/tech-stack.md` change propagates to `README.md`, `AGENTS.md`, and open feature specs in the same commit. |
| `specs/mission.md`, `README.md` | Links follow the roadmap split. |

The skills are plain markdown checklists with no agent-specific mechanics beyond the interview
step, and are linked from `AGENTS.md`, so a Copilot, Cline, or Aider user can follow the same
procedure by hand. That was the point of the existing pointer-file arrangement and this change does
not narrow it.

Deliberately not adopted: the course's `changelog` skill (a `git log` scraper — this file's
per-file tables and reasoning would be lost), and its `TODO.md` / `backlog/` inboxes (the roadmap's
Deferred table already does that job, with a recorded reason per item).

No code, build, or version change, so no README dependency statements move.

---

### Documentation: agent-facing `specs/` folder

Added `specs/mission.md`, `specs/tech-stack.md`, and `specs/roadmap.md`, following the
`specs/` convention from the DeepLearning.AI *AI Coding Workflows* course. `AGENTS.md`
remains the command reference and now links to them; the specs carry intent, pinned
versions with the file that owns each pin, and the phased work queue.

| File | Change |
|------|--------|
| `specs/mission.md` | New. Purpose, the architectural commitments (one backend at a time, backends behind `InferenceBackend`, CPU/GPU postprocessing parity, mock-injected tests), component map, out-of-scope list. |
| `specs/tech-stack.md` | New. Versions with their pin locations, backend and media/GPU tables, every CMake option and default, configure-time constraints, CI coverage, and the known pin duplications (TensorRT in two places; `project()` declares no version). |
| `specs/roadmap.md` | New. Five phases with verification criteria, the deferred list with reasons, and the GPU pipeline design reference. |
| `docs/GPU_PIPELINE_ROADMAP.md` | **Removed**, folded into `specs/roadmap.md`. Completed phases 1–3 collapse into the design reference (architecture, the 8-rule model contract, the packed output contract, correctness rules, risks); the unbuilt phases 0.3/0.4, 4.3, and 5 become roadmap phases 2–4 with their tolerances intact. |
| `AGENTS.md`, `README.md` | Link to `specs/` instead of the removed doc. |

No code, build, or version change, so no README dependency statements move.

Noted while writing: the `.gitignore` known issue below is partly stale — `*.pte` is
now ignored, and `output_video.mp4` has no leading slash so it already matches at any
depth. Tracked in `specs/roadmap.md` Phase 1.

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

### Documentation

| File | Change |
|------|--------|
| `docs/backend-parity-segmentation-video.md` | Record of a manual cross-backend instance-segmentation video test (ONNX Runtime, TensorRT, ExecuTorch) over a 320-frame 1080p clip. All three run the same `RFDETRSegMedium` at 432×432 and agree on detections; ExecuTorch and ONNX Runtime match exactly, TensorRT is ~0.01 lower on four of seven scores from engine precision. Covers the manual gap left by CI, which tests neither TensorRT nor ExecuTorch. |

### Known Issues

| Area | Issue |
|------|-------|
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
