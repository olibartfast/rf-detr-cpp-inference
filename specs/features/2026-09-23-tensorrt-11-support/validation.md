# Validation: TensorRT 11 support

Validated on 2026-09-23, branch `claude/focused-dirac-wxgnu7`, without a GPU.

Headers: NVIDIA/TensorRT OSS tags `v10.13.3` and `v11.3` (`include/`); CUDA 13.0 headers and
`nvcc` from the `nvidia-cuda-runtime`, `nvidia-cuda-nvcc`, `nvidia-cuda-crt` and
`nvidia-cuda-cccl` 13.0 wheels.

- PASS (reproduced): before the change, `rfdetr_inference_lib` fails against 11.3 headers:
  `'kFP16' is not a member of 'nvinfer1::BuilderFlag'`.
- PASS: after the change, CMake configure + `cmake --build --target rfdetr_inference_lib` with
  `-DUSE_TENSORRT=ON -DWERROR=ON` succeeds against both 10.13.3 and 11.3 headers
  (`TENSORRT_ROOTDIR` pointing at each; stub shared objects, as in CI).
- PASS: `TRT_HEADERS=compat ./scripts/ci/stage_gpu_headers.sh` stages `NvInfer.h`,
  `NvOnnxParser.h` and `NvInferVersion.h` (`NV_TENSORRT_MAJOR 11`) into `tensorrt-compat-headers`.
  The DALI half of the script could not be exercised locally (pypi.nvidia.com blocked by the
  sandbox proxy); it is unchanged and runs in CI.
- PASS: `export_trt.sh` inner command against fake `trtexec` binaries — `--fp16` passed when
  `--help` lists it, omitted when it does not.
- PASS: `./scripts/check_version_sync.sh`; clang-format-18 on the backend; `bash -n` on the scripts.
- UNRUN (no GPU): engine build and inference on TensorRT 11.x; FP16-converted ONNX parity against
  the FP32 engine; the float32 I/O guard on a real engine; `gpu-compile.yml` compat job (runs on
  the PR). Run [gpu-verify](../../../.claude/skills/gpu-verify/SKILL.md) on an 11.x prefix before
  any pin bump.
- UNRUN: Docker builds — no `dockerfile.*`, Docker build argument or Docker-used pin changed
  (`TENSORRT_COMPAT_VERSION` is read by CI staging only).
