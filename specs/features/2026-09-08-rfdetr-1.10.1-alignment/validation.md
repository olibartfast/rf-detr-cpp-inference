# Validation: RF-DETR 1.10.1

Validated on 2026-09-08, branch feature/rfdetr-1.10.1-alignment.

- PASS: official 1.10.0...1.10.1 diff reviewed; training/device sampling changes do not alter exported inputs, outputs, operators or deploy APIs.
- PASS: `./scripts/check_version_sync.sh`; all pins agree, ONNX opset remains 17.
- PASS: `python3 -m unittest tests/python/test_export_scripts.py`, 6 tests.
- PASS: `cmake --build build --parallel 4` and CTest; 52 unit tests.
- PASS: fresh Nano export using `/tmp/rfdetr-1101-venv/bin/python deploy/export_detection.py --model_type nano --device cpu --input_size 640 --output_dir /tmp/rfdetr-1101-export --output_name rfdetr-nano-1101`.
- Environment: Python 3.12 venv with system-site-packages, rfdetr 1.10.1 installed in the venv, existing torch 2.12.0 reused. This was not a clean dependency-resolution test.
- PASS: ONNX checker; float32 input `input [1,3,640,640]`, outputs `dets [1,300,4]`, `labels [1,300,91]`, opset 17.
- PASS: C++ ONNX Runtime image inference on data/dog.jpg: bicycle 0.939584, dog 0.926562, car 0.829679, motorbike 0.598268. Output: `/tmp/rfdetr-1101-export/output_image.jpg`.
- PASS: `RFDETR_TEST_MODEL=/tmp/rfdetr-1101-export/rfdetr-nano-1101.onnx ctest --test-dir build --output-on-failure -R IntegrationTests`; 4 tests passed, 1 keypoint test skipped (no keypoint model).
- PASS: live README/docs, mission, tech stack, AGENTS and version pin synchronized. Historical 1.10.0 references and completed alignment records retained.
- PASS: `git diff --check`.

## Remaining acceptance

- UNRUN: TensorRT/ExecuTorch exports and GPU parity; no validation of those backends is inferred from the Nano CPU run.
- Docker rebuild not applicable to this export-only pin: images compile the C++ runtime and do not install rfdetr. Dockerfiles, runtime dependency pins and build options did not change.
- No tag, commit, push or merge performed. Full release gates remain open in [roadmap](../../roadmap.md).
