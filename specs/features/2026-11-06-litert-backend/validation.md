# Validation

## Automated — no GPU

- [x] ./scripts/check_version_sync.sh
- [x] cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DWERROR=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
- [x] cmake --build build --parallel
- [x] ctest --test-dir build --output-on-failure -R UnitTests
- [x] LiteRT configure/build with USE_ONNX_RUNTIME=OFF, USE_LITERT=ON, LITERT_ROOTDIR supplied, WERROR=ON; unit and integration tests.
- [x] Configure guards reject zero/two backends and LiteRT with each GPU pipeline option.
- [x] find src tests -name '*.cpp' -o -name '*.hpp' | xargs clang-format-18 --dry-run --Werror
- [x] find src -name '*.cpp' ! -name 'tensorrt_backend.cpp' | xargs clang-tidy-18 -p build
- [x] cppcheck --enable=all --check-level=exhaustive --std=c++20 --suppress=missingIncludeSystem --suppress=unmatchedSuppression --suppress=unusedFunction --error-exitcode=1 -I src src/

## Automated — with a model

- [x] Export Nano detection and Seg-Nano with pinned RFDETR_VERSION and run native CLI on data/dog.jpg.
- [x] Compare native and Python LiteRT outputs on identical tensors: max absolute box error <= 1e-5; class logits <= 1e-3; mask logits <= 1e-3. Same runtime, so this gate checks C++ integration rather than cross-runtime conversion.
- [x] Repeat inference, reject wrong shape/size and invalid paths, and reinitialize safely.

## Compile without device

LiteRT is CPU only and needs no GPU. Existing GPU skip rules remain unchanged; no src/gpu/ edit is planned.

## Manual

- [x] Inspect annotated detection and segmentation outputs.
- [x] Confirm dependency staging needs Python only at setup and exported inference executes natively.

## Definition of done

- [x] CHANGELOG and all backend/constitution documentation updated.
- [x] Validation recorded honestly; unavailable checks are UNRUN with reasons.
- [ ] Roadmap ticked only after merge; feature branch merged and deleted (requires separate authorization for outward-facing publication).

## Results

Verified locally on 2026-11-06, branch `feature/litert-backend`, based on current `origin/develop`.
No commit, merge, push or tag was made. Roadmap boxes intentionally remain open until merge.

### Environment and artifact provenance

- Linux x86_64, GCC 15.2.0; Python 3.12.16 in `/tmp/rfdetr-litert-venv`.
- Runtime: `LITERT_VERSION`; exporter: `RFDETR_VERSION`. Resolved transitive exporter versions during this verification: litert-torch 0.10.0, torch 2.14.1+cpu, torchvision 0.29.2+cpu. These are observations, not additional project pins.
- SDK staged successfully twice with `PYTHON=/tmp/rfdetr-litert-venv/bin/python ./scripts/fetch_litert.sh <prefix>`.
  The second staging exercised the final helper including generated build_config.h.
- Models: `/tmp/rfdetr-litert-models/rfdetr-nano.tflite` (384 square input),
  `/tmp/rfdetr-litert-models/rfdetr-seg-nano.tflite` (312 square input, 100 masks at 78 square).
- Detection model SHA256: `681c08430596930603b6119fc5332a28de028bf1bedc2d847edd2596a8a9795b`.
- Segmentation model SHA256: `6b1c4b18bc68edb08bf8fffb5c6a74d3c2eb7b9b0c7bd753f0d6be5bd74cb2dc`.
- Large models/reference tensors and logs stay outside version control in `/tmp/rfdetr-litert-models` and `/tmp/rfdetr-litert-reference`.

### Passing checks

- Release builds with `-DWERROR=ON` for ONNX Runtime and LiteRT, using `-DUSE_OPENCV=ON`.
  The unmodified default FFmpeg/SDL2 configuration could not configure: SDL2 development files are absent and sudo requires interactive authentication. This specific media configuration is **UNRUN**, not passing.
- Unit suites pass: 110 tests for LiteRT and 109 for ONNX Runtime; all nine existing ONNX catalog CTests pass.
- LiteRT integration CTest passes without fixtures by skipping model-dependent cases.
  Separately, the real-model parity test passes for each of the two exported models, with no skips.
- Six configure guards pass: no backend, ONNX+LiteRT, and LiteRT with each of DALI, CUDA preprocess,
  CUDA postprocess, and the combined GPU pipeline all fail for the expected reason.
- Final format check with clang-format 18.1.8, version sync, `bash -n`, Python compile checks and `git diff --check` pass.
- The LiteRT backend, contract helper and changed CLI sources pass clang-tidy 18.1.8 using the LiteRT compilation database.
- ASan+UBSan Debug LiteRT build passes unit tests and both real-model lifecycle/parity cases.
  The prebuilt native runtime is not sanitizer-instrumented; these runs check this project's boundary/ownership code.
- `ldd build-litert/inference_app` resolves the native libLiteRt.so from the prefix and has no Python library dependency.
- CLI detection finds bicycle, dog and car; segmentation finds the same three with masks.
  Both rendered images were visually inspected. The initial two-task CLI rejected keypoints; the verified pose extension below supersedes that restriction.

### Reproduce real-model parity

```bash
python deploy/export_litert.py --model_type nano --output_dir /tmp/rfdetr-litert-models
python deploy/export_litert.py --model_type nano --segmentation --output_dir /tmp/rfdetr-litert-models
python tests/integration/generate_litert_reference.py /tmp/rfdetr-litert-models/rfdetr-nano.tflite /tmp/rfdetr-litert-reference/detection
RFDETR_LITERT_MODEL=/tmp/rfdetr-litert-models/rfdetr-nano.tflite \
RFDETR_LITERT_REFERENCE=/tmp/rfdetr-litert-reference/detection \
./build-litert/integration_tests --gtest_filter='LiteRTIntegration.*'
# Repeat with rfdetr-seg-nano.tflite and a separate reference directory.
```

All outputs pass the numeric limits above. Both iterations compare every output element, verify finite values,
check invalid shape/count/output access, retain the old model after failed reinitialization, and perform a successful reload.

### Existing lint findings and remaining gates

- **Fixed during release preparation:** the five `functionStatic` cppcheck findings and the
  OpenCV `bugprone-empty-catch` clang-tidy finding were resolved with behaviour-preserving
  changes. The release gate re-ran both tools clean; see the
  [v0.7.0 release validation](../2026-11-06-release-v0.7.0/validation.md).
- **UNRUN:** hosted CI job (added here, not dispatched), other platforms/architectures,
  alternate backends requiring external SDKs, and real-display/video checks. The release
  preparation built the Docker matrix on this host; its record is in the release validation.
- The phase is implemented and locally verified for Linux CPU detection, segmentation and pose.
  Merge into `develop`, release tagging and branch deletion are tracked by the
  [v0.7.0 release spec](../2026-11-06-release-v0.7.0/).


## User-requested pose extension

- [x] Real keypoint export succeeds with the project Linear reshape adapter; the unmodified direct converter fails.
- [x] CLI and backend accept pose; keypoint tensor parity <= 1e-3 against Python LiteRT.
- [x] Skeletons visually inspected on real footage.
- [x] Same-video pose run checked for frame count and readable output: 600/600 frames, 50 fps, 12 seconds.
- [x] Final showcase contains detection and segmentation only; the rejected pose section is absent.

The earlier two-task results remain valid. The pose video and final showcase checks pass.

### Pose export and boundary checks

The unmodified exporter fails with `tfl.batch_matmul ... expected 3 but got 4`, including
with the installed converter. `deploy/litert_keypoint_export.py` applies an export-only
Linear reshape to the prepared private graph and restores its temporary exporter hook.
On a real city frame, the original and rewritten PyTorch outputs had max absolute
differences `[0, 0, 0]` over shapes `[1,100,4]`, `[1,100,2]`, `[1,100,34,8]`.
Native C++ passes the existing integration test against both Python LiteRT random-input
references and the unmodified PyTorch city-frame reference, with keypoint tolerance 1e-3.
The final artifact was exported again through the repository CLI and passed native parity.
ASan+UBSan unit tests and native pose parity/lifecycle checks pass with the final pose-enabled build.
Two Python adapter tests pass: rank-4 output/parameter preservation and restoration on failure.
No installed upstream package files or trained parameters were modified.

### Pexels video checks

Source: https://www.pexels.com/video/a-city-street-with-cars-and-buses-on-it-18437773/
(Enrique B / Pexels). Original: 3840x2160, 50 fps, 44.16 seconds, 2,208 decoded frames.
The test input is 1280x720, preserving all 2,208 frames and the original frame rate.
Its SHA256 is `d6a7a0da8bf9757b68382a9bba915f4fe52cbe242c8d4567e8ff14d7ffc63c28`.

- Detection: PASS, exit 0, all 2,208 frames; native output and shareable H.264 copy decoded
  and frame-count checked. Wall time 24:48.67, peak RSS 647224 KiB.
- Segmentation: PASS, exit 0, all 2,208 frames; native output and H.264 copy decoded
  and frame-count checked. Detailed timings are in output/litert-pexels-18437773/results.json.
- Pose: PASS, exit 0, source seconds 8–20, 600/600 frames at 50 fps, threshold 0.20, model resolution 384. Wall time 25:15.82, peak RSS 873916 KiB. Native and H.264 outputs decoded and frame-count checked.
- Superseded showcase: functional decode PASS, one 70-second three-task video, 1920x1080 at 30 fps, H.264 yuv420p + stereo AAC. All 2,100 video frames decode cleanly; the full audio/video decode has no errors. Task, comparison, results and thumbnail frames were visually inspected.

Runs are functional tests rather than isolated performance benchmarks: tasks and video
editing overlap. Outputs are ignored artifacts under `output/litert-pexels-18437773/`;
raw logs/models remain outside version control. Visual inspection is a smoke test, not
a ground-truth accuracy measurement.

Pose artifact SHA256: `c942ab3b46cb7c7cbe0602f0f3c184a0540028feee39428ca01a6a5a80184594`.
Local video hardware: Intel Core i5-6500T @ 2.50 GHz, 4 logical CPUs.
Pose at threshold 0.20 includes noisy low-confidence predictions around occluded people and
vehicle occupants; the raw output is retained and no accuracy claim is made.

Final showcase: `output/litert-pexels-18437773/rfdetr-litert-youtube-showcase.mp4`. `showcase-validation.json` records ffprobe metadata;
`youtube-description.md` contains title, chapters, model/settings information and source credit;
`youtube-thumbnail.jpg` is the 1280x720 thumbnail. The render script and original footage are
retained under the ignored `showcase-edit/` directory. No upload/publication was performed.

### Final showcase revision for release

The user rejected the three-task edit because pose predicted a person on a vehicle.
The final video replaces it with detection/segmentation only: 64 seconds, 1920x1080 at 30 fps,
H.264 yuv420p and stereo AAC. All 1,920 video frames and the complete audio/video stream decode
without errors. Its thumbnail, title, description and chapters contain only these two tasks.
The earlier 70-second artifact and pose previews are archived under the ignored
`superseded-three-task/` directory, and are not the release showcase.
The experimental pose code remains available; the release changelog explicitly notes its
false-positive risk. Earlier cppcheck/clang-tidy findings in this record were subsequently
resolved during release preparation; see the release validation record for the final gates.
