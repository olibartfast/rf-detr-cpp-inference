# Validation

## Automated — no GPU
- [x] ./scripts/check_version_sync.sh
- [x] python -m unittest discover -s tests/python -v: 11 tests, including 4 adapter regressions.
- [x] ./scripts/scoreboard.sh: all 11 CTest entries pass. Its fallback used clang-format 21; separate pinned 18.1.8 format check passes.
- [x] clang-tidy 18.1.8 and cppcheck exhaustive analysis pass on the default compile database.

## Automated — runtime/device
- [x] Re-export pretrained Nano with installed rfdetr==1.12.0 and run default ONNX end to end; RFDETR_TEST_MODEL integration CTest passes.
- [ ] ExecuTorch detection/segmentation exports execute on pinned native runtime.
- [x] LiteRT detection/segmentation/pose exports execute with native/Python raw boxes within 1e-5 and other raw outputs within 1e-3 (existing integration gate); adapter rewrite atol=rtol=1e-5.
- [ ] TensorRT engine execution and GPU parity per gpu-verify if regenerated artifacts implicate those paths.

## Compile without device
No C++/CUDA changes planned; GPU tests must retain GTEST_SKIP without hardware.

## Manual
- [x] Regression exercises base exporter sidecar lifecycle and retains original keypoint names/metadata; source review accepted.
- [x] Review dependency extras, exporter return paths and output names against tag source.

## Definition of done
- [x] CHANGELOG and README updated; all required validation passed or exact UNRUN limitations recorded.
- [ ] Alignment merged to develop and branch deleted after review. No numbered roadmap phase to tick.

## Evidence (2026-11-07)

Isolated checkout based on origin/develop ddc2a90. GCC 15.2.0, default FFmpeg/SDL2 media, strict warnings. Installed upstream wheel RFDETR_VERSION; CPU torch initially 2.15.0 for ONNX, 2.14.1 for the pinned LiteRT converter. Python 3.11.15. Tool wheels clang-format/clang-tidy 18.1.8 are validation tools only, not new project dependencies.

ONNX artifact `/tmp/rfdetr-align-models/rfdetr-nano.onnx`: float32 input [1,3,640,640], dets [1,300,4], labels [1,300,91]. C++ dog.jpg CLI finds bicycle (0.939584), dog (0.926562), car (0.829679), motorbike (0.598268); output `/tmp/rfdetr-align-detection.jpg`. Integration fixtures requiring other task models skip; the provided detection case executes.

Logs `/tmp/rfdetr-align-{build,cppcheck,tidy,export,inference,integration,python-tests}.log`. Source review and test doubles do not prove conversion accuracy of pose.

UNRUN: TensorRT/CUDA/DALI runtime and compute-sanitizer: no NVIDIA device/toolkit available. ExecuTorch runtime: no native ExecuTorch SDK/install prefix in this environment; unchanged native runtime pin must still be checked against regenerated .pte operators. No Docker files/build options/runtime pins changed; Python export requirements are not installed by these Dockerfiles, so no Docker build matrix is implicated.

LiteRT strict build and all 11 CTest entries pass. Pretrained Nano at native 384 resolution exported successfully using the pinned converter and upstream pixel-row sampling; native/Python integration parity passes with zero reported maximum absolute error for both boxes and logits, including reload/lifecycle checks. Native dog.jpg CLI finds bicycle/dog/car. ONNX helper defaults to 640, so these CLI scores are not a cross-format parity comparison.

The converter pins typing-extensions to an older release that does not provide Sentinel, while the newest pydantic-core imports it. The temporary LiteRT validation environment uses pydantic 2.11.7/core 2.33.2 (within upstream's declared range) to run; this is an environment workaround, not a new repository pin. Plain ONNX export succeeded with the initially resolved dependencies.

Official RFDETRKeypointPreview exported at native 576 resolution through the migrated adapter. Full-model Linear rewrite parity passes at atol=rtol=1e-5. Generated sidecar task is keypoints with pred_boxes=0, pred_logits=1, pred_keypoints=2. Interpreter shapes are [1,100,4], [1,100,2], [1,100,34,8]. Native/Python integration passes with zero reported maximum absolute error on all three outputs, including reload checks. Pose CLI completes on dog.jpg with no detections at threshold 0.5 (no person in fixture); this is execution evidence, not a pose-accuracy benchmark.

Artifact SHA256 provenance:

- `rfdetr-keypoint-preview.tflite`: `5561b634e4dfc1422c3d9dceaa666c6ddd0476a675ab982e796cc107ba7962c5`
- `rfdetr-nano.onnx`: `cdf961585e18d9ded9189ead98b52ea9ef7bfa059368c481e86d666078b596dd`
- `rfdetr-nano.tflite`: `9e3c71ad381eba707501f78cac268c08c7712196ad8804646a033a686d3439af`

Seg-Nano at native 312 resolution also exports, passes native/Python parity with zero reported maximum absolute error across all three outputs, and completes native dog.jpg segmentation CLI. All three LiteRT task artifacts use upstream pixel-row sampling. This is native-versus-interpreter parity on identical tensors; no claim of COCO accuracy or cross-version/cross-runtime equivalence is made.

Reproduce task parity with the existing helpers (replace MODEL and REF with each task artifact/reference directory):

```bash
/tmp/rfdetr-align-venv/bin/python tests/integration/generate_litert_reference.py MODEL REF
RFDETR_LITERT_MODEL=MODEL RFDETR_LITERT_REFERENCE=REF ./build-litert/integration_tests --gtest_filter='LiteRTIntegration.*'
```

Real-model parity XML: `/tmp/rfdetr-align-litert-{detection,pose,seg}.xml`; conversion logs: `/tmp/rfdetr-align-{litert,pose,seg}-export.log`. Pose checkpoint and segmentation checkpoint downloaded from the official upstream URLs into `/tmp/rfdetr-align-weights`; Nano reused the already-cached official checkpoint. No generated model, SDK or validation environment is committed.

Implementation and validation files are copied into the primary workspace on feature/rfdetr-1.12.0, based on origin/develop. The divergent local develop ref is preserved. The maintainer authorized pushing the alignment and a release bump after green CI on 2026-11-07; alternate runtime gates remain explicitly UNRUN. Merge/publication evidence is recorded below as it becomes available.

- `rfdetr-seg-nano.tflite` SHA256: `cd2f759e956a1070985a4909bd6f44e8b06543cbb24b7bdfc92dee6fccd82573`
