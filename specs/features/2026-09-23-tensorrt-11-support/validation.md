# Validation: TensorRT 11 support

Validated on 2026-09-23, branch `feature/tensorrt-11-support`, without a GPU.

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

## Pin bump — 2026-09-25, local RTX 3060 Laptop (driver 610.43.02, CUDA 13.3)

- PASS: clang-format-18, clang-tidy-18, cppcheck (the `lint.yml` flags; the backend's raw loops
  moved to `std::accumulate`/`std::transform`), `check_version_sync.sh`,
  `check_dockerfile_parity.sh`.
- PASS: `-DWERROR=ON` builds against the TensorRT 11.2.1.2 tarball (56 unit tests) and 10.13.3.9
  with `-DUSE_GPU_PIPELINE=ON` and a DALI 1.51.2 prefix (72 unit tests). Host `nvcc` was 12.0,
  not the pinned CUDA.
- PASS: `dockerfile.trt` builds for all eight `MEDIA_BACKEND` × `GPU_PIPELINE` combinations on
  the 26.08 images.
- PASS: `--gpus all` in the `ffmpeg`/`on` image — engines built from `rfdetr-nano-1101.onnx` and
  `rfdetr-seg-nano-576.onnx` on TensorRT 11.2.1.2; on `data/dog.jpg` detection and segmentation
  (CPU path) find dog, bicycle, car and motorbike.
- FAIL, fixed (plan step 12): `--gpu-preprocess` aborted with `dlopen libnvimgcodec.so failed!`.
  With nvImageCodec staged, `--gpu-preprocess`, `--gpu-postprocess` and both together give the
  same four classes as the CPU path (score deltas ≤ 0.011), using the checked-in 576 `.dali`
  pipelines unchanged. After rebuilding, the `ffmpeg` and `opencv` × `dali`/`on` images pass the
  same run with no workaround. `fetch_dali.sh` into a fresh directory stages the flattened layout.
- UNRUN: the gpu-verify parity tolerances, compute-sanitizer and benchmarks over a video; the
  432 pipelines.
