# Validation: RF-DETR 1.10.0 Alignment

Status is updated only from observed commands. An unchecked item is not implied to pass.

## Static and hermetic

- [x] `./scripts/check_version_sync.sh` — PASS (all pins agree with `versions.env`, RFDETR 1.10.0, opset 17).
- [x] `python3 -m unittest tests/python/test_export_scripts.py` — 6/6 PASS.
- [x] `git diff --check` — clean.
- [x] Review every remaining `1.9.4` occurrence as intentional historical context — CHANGELOG history, `docs/export.md` historical note, `docs/usage.md` background-class-id note, and the `2026-08-25-rfdetr-1.9.4-alignment/` spec triple are all historical; no live version guidance still says 1.9.4.

## Clean 1.10.0 export

- [x] Clean Python 3.11 environment reports `rfdetr==1.10.0` (`importlib.metadata` in `/tmp/rfdetr-110-venv`).
- [x] Explicitly named ONNX export returns the same path the script prints, and that file exists — `/tmp/rfdetr-110-export/rfdetr-nano-110.onnx`.
- [x] ONNX checker passes; input/output names, order, dtypes, ranks, and shapes are recorded — opset 17; `input [1,3,640,640]` (f32); `dets [1,300,4]`, `labels [1,300,91]` (f32).
- [x] Equivalent 1.9.4 and 1.10.0 signatures match — `rfdetr-nano.onnx` (1.9.4) and `rfdetr-nano-110.onnx` (1.10.0) have identical opset and input/output names/shapes/dtypes.

## C++ runtime

- [x] Default Release build succeeds — `./scripts/scoreboard.sh` PASS (`-DWERROR=ON` configure + build).
- [x] Unit tests pass — `ctest` UnitTests + IntegrationTests 2/2 PASS.
- [x] Fresh 1.10.0 ONNX artifact completes one image inference with expected outputs — `./build/inference_app /tmp/rfdetr-110-export/rfdetr-nano-110.onnx data/dog.jpg data/coco-labels-91.txt` → 4 detections (bicycle 0.94, dog 0.93, car 0.83, motorbike 0.60).

## Optional backends

- [x] ExecuTorch export/operator/runtime validation: `UNRUN` (matching exporter/runtime not exercised; recorded, not inferred).
- [x] TensorRT native-export validation: `UNRUN` (Python `tensorrt` extra not installed).
- [x] Docker matrix: one build per backend run locally — `dockerfile.onnxrt`, `dockerfile.executorch`, `dockerfile.trt` (`GPU_PIPELINE=on`). Other `MEDIA_BACKEND`/`GPU_PIPELINE` combinations and any `--gpus all` run are `UNRUN` (no NVIDIA hardware on this machine).

## Definition of done

- [x] Pins, live documentation, README, mission, and tech stack agree on 1.10.0.
- [x] CHANGELOG records what does and does not reach C++.
- [x] No C++ source, CUDA source, or DALI pipeline changed — `git status` shows no `src/`, `cmake/`, or `third_party/` modifications.
- [x] Scoped staged diff reviewed and committed to `develop` after locally runnable gates pass.
