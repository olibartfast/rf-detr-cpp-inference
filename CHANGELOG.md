# CHANGELOG

Tracks upstream `rfdetr` version changes that affect this C++ inference project.

---

## [Unreleased] — RF-DETR 1.9.0 alignment + ExecuTorch backend

**Upstream release**: https://github.com/roboflow/rf-detr/releases/tag/1.9.0

### Added

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

### Changed

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

### Why no postprocessing port or model re-export

Diffing `1.8.3...1.9.0` over the code this project mirrors:

- `export/_onnx/exporter.py` (+201/-97) and `models/lwdetr.py` (+78/-57) are **typing-only** — protocols, `cast`, f-strings. Graph structure, output names, ordering, and opset are unchanged, so existing exported models stay compatible and **no re-export is required**.
- `models/postprocess.py` (+63/-18) chunks mask upsampling to cut peak GPU memory (identical results — still bilinear to target size, then `> 0.0`) and adds `upsample_masks_to_image_size`, an opt-in validation-only flag. `media.cpp::resize_threshold_mask` already matches the default path.
- **PR #1206 confirms this project's preprocessing.** Upstream `predict()` had been resizing with antialias enabled, drifting from training; 1.9.0 sets `antialias=False` to match the antialias-free bilinear (`cv2.INTER_LINEAR`) resize used during training. `media.cpp::preprocess_bgr_image` has always been exactly that, so 1.8.3's `predict()` was the side that disagreed, and 1.9.0 closes the gap. Locked in by `PreprocessFrame.ResizeIsAntialiasFree`.

### End-to-end verification

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

### Upstream ExecuTorch packaging bug (affects install-prefix builds)

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

---

## [Unreleased] — package-manager abstraction

Unified package-manager abstraction layer (`cmake/deps/`) wrapping apt, conan,
vcpkg, and provided-download strategies behind a single `find_dependency_unified()`
facade. All dependencies — including GTest, Google Benchmark, stb, and font8x8 —
now route through the facade instead of ad-hoc `FetchContent` or manual
`target_include_directories` calls.

### Added

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

### Changed

| File | Change |
|------|--------|
| `CMakeLists.txt` | GTest, Google Benchmark: raw `FetchContent` → facade (PROVIDED_ACQUIRE=FETCHCONTENT fallback); stb, font8x8: manual `target_include_directories` → facade (PROVIDED_ACQUIRE=VENDORED); `deps_get_rec()` used for RPATH/runtime libs. |
| `README.md` | New Dependency Resolution section with mode/chain table, conan CMakeDeps-only example, vcpkg manifest example, system-package notes. |
| `AGENTS.md` | Brief DEPS section: backend selection, dependency resolution, DEPS_DEBUG. |

### Fixed

- **DEPS_OFFLINE honored for FETCHCONTENT**: `can_resolve` returns FALSE when offline, preventing network clone attempts in air-gapped builds (Codex review P2).
- **vcpkg MODULE mode fallback**: vcpkg's FFmpeg port ships a Find module (not a config file); the strategy now tries CONFIG first, then MODULE, and uses `<FIND>_LIBRARIES` variables when IMPORTED targets don't exist.
- **Chain order**: `conan;apt;provided` and `vcpkg;apt;provided` — apt fills in system packages (Threads) even in conan/vcpkg modes.
- **Conan "version conflict" resolved**: OpenCV and FFmpeg are mutually exclusive (`USE_OPENCV` option); each conan graph resolves independently.

### Why

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

## v0.3.0

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

## v0.2.2

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

## v0.2.1

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

## v0.2.0

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

## v0.1.3

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

## v0.1.2

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

## v0.1.1

[v0.1.1](https://github.com/olibartfast/rf-detr-cpp-inference/commit/f9028533ad96d79117da2a74a5aa121fd80277c1)

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

## v0.1.0

[v0.1.0](https://github.com/olibartfast/rf-detr-cpp-inference/commit/5ba569b7f7454a2b0fbe3e56ee885d9dad46fc70)

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
