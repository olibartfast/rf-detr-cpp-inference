# Validation: RF-DETR 1.11.2

Validated on 2026-10-06, branch `feature/rfdetr-1.11.2-alignment`, Python 3.12 venvs made with
`uv` (CPU torch wheels), RTX 3060 Laptop (sm_86, driver 610.43.02) for the TensorRT row.

## Upstream review

- PASS: release notes for 1.11.0, 1.11.1 and 1.11.2 read. The 1.10.1...1.11.2 diff of
  `src/rfdetr/export`, `src/rfdetr/models/{lwdetr.py,heads/segmentation.py}` and `pyproject.toml`
  was reviewed. Classification is in requirements.md.

## ONNX (default backend)

- PASS: fresh exports with `rfdetr[onnx]==1.11.2` (torch 2.14.1, onnx 1.23.2):
  `deploy/export_detection.py --model_type nano --input_size 384`,
  `deploy/export_segmentation.py --model_type nano --input_size 312`, `deploy/export_keypoint.py`.
- PASS: ONNX checker; all opset 17, float32 `input`:

  | Model | Input | Outputs |
  |-------|-------|---------|
  | detection nano | `[1,3,384,384]` | `dets [1,300,4]`, `labels [1,300,91]` |
  | segmentation nano | `[1,3,312,312]` | `dets [1,100,4]`, `labels [1,100,91]`, `masks [1,100,78,78]` |
  | keypoint preview | `[1,3,576,576]` | `dets [1,100,4]`, `labels [1,100,2]`, `keypoints [1,100,34,8]` |

  The segmentation graph has no `Einsum` any more, as the 1.11.2 notes say.
- PASS: the same three models exported with `rfdetr[onnx]==1.10.1` at the same resolutions, and
  both run in ONNX Runtime on one seeded random input: max abs difference **0** on every output
  (`dets`, `labels`, `masks`, `keypoints`).
- PASS: C++ ONNX Runtime app on `data/dog.jpg`. Detection: bicycle 0.950795, dog 0.940137, car
  0.854236. Segmentation: bicycle 0.94919, dog 0.925574, car 0.800418, masks 57391/35538/12554 px.
- PASS: `RFDETR_TEST_MODEL=<det-nano-1112.onnx> RFDETR_KEYPOINT_MODEL=<rfdetr-keypoint-preview.onnx>
  ./build/integration_tests`: 5/5 pass, keypoint included. `ctest -R 'UnitTests|IntegrationTests'`: 2/2.
- PASS: `python -m unittest discover -s tests/python`: 7 tests.

## ExecuTorch

- PASS: `pip install "rfdetr[executorch]==1.11.2"` resolves **executorch 1.3.1** (torch 2.14.1).
  `deploy/export_executorch.py --model_type nano --input_size 384` writes `rfdetr-nano.pte`.
  The same export with `torch==2.12.1` (upstream's CI pairing) also succeeds.
- Finding: run without the venv's `bin/` on `PATH`, the export fails with
  `No such file or directory: 'flatc'`. ExecuTorch calls the `flatc` installed in the venv.
  Documented in `docs/export.md` and the script's closing note.
- PASS: `-DUSE_EXECUTORCH=ON -DEXECUTORCH_ROOTDIR=~/dependencies/executorch-1.4.0` (v1.4.0, optimized
  kernels) builds. The 1.3.1-exported `.pte` runs on it at `--resolution 384`: bicycle 0.950795,
  dog 0.940137, car 0.854237, matching the ONNX export to 1e-6. So an exporter older than the
  runtime is compatible, and `EXECUTORCH_VERSION` stays at v1.4.0.

## TensorRT

- PASS: detection and segmentation ONNX cast with 1.11.2's
  `rfdetr.export._tensorrt.exporter._cast_onnx_to_fp16` (the `fp16=True` path on TensorRT 11).
  Graph I/O stays float32 with unchanged names and shapes.
- PASS: run in the C++ TensorRT backend inside `rfdetr-p6:ffmpeg-off` (`dockerfile.trt`, NGC 26.08,
  TensorRT 11.2.1.2; the app was built from the Phase 6 branch merged as PR #18) with `--gpus all`.
  The backend built engines from the ONNX and accepted the float32 I/O. Detection FP16 vs FP32:
  0.950411/0.940353/0.858244 vs 0.95089/0.940194/0.854531. Segmentation FP16: 0.948728/0.922482/0.800068,
  masks 57446/35536/12558 px. The engine's per-layer precision was not inspected; the small score
  shift is consistent with FP16 compute.
- Finding: the local TensorRT 11.2.1.2 tarball cannot build engines on this machine, because
  `libnvinfer_builder_resource_sm86.so.11.2.1` is missing from it. This is a local environment gap
  and not caused by this change; the Docker image was used instead.
- UNRUN: `model.export(format="tensorrt", fp16=True)` end to end (needs the `[tensorrt]` extra's
  pip TensorRT matching the runtime), dynamic-batch engines (not consumed), the GPU pipeline
  parity gate. This alignment changes no C++ or CUDA code, and the pre/post contract is unchanged.

## Repository gates

- PASS: `./scripts/check_version_sync.sh`; all pins agree.
- PASS: clang-format, clang-tidy and cppcheck (the AGENTS.md pre-commit gate).
- Docker rebuild not applicable: the images compile the C++ runtime and do not install rfdetr.
  No Dockerfile, runtime pin or build option changed.
- README updated: the export-tooling version table and its ExecuTorch note.
