# TensorRT 11 support (10.x retained)

TensorRT 11 is out (11.0 through 11.3 at the time of writing). This change makes the TensorRT
backend build and run on 11.x without dropping 10.x, which stays the pinned `TENSORRT_VERSION`.
It touches `src/backends/tensorrt_backend.cpp`, a path CI cannot execute, hence this spec.

## What TensorRT 11 changed that matters here

Checked against the public headers at NVIDIA/TensorRT tag `v11.3` and NVIDIA's 10.x→11.x
migration guide:

| Change | Effect on this project |
|--------|------------------------|
| Weak typing removed; `BuilderFlag::kFP16` (and every per-precision flag) deleted | `config->setFlag(BuilderFlag::kFP16)` no longer compiles — the only compile break |
| Networks are strongly typed; `kSTRONGLY_TYPED` deprecated and ignored | An FP32 ONNX builds an FP32 engine; FP16 needs a converted ONNX |
| `trtexec --fp16/--bf16/--fp8/--int4/--best` removed | `export_trt.sh` and the `trtexec` recipes in `docs/export.md` fail on an 11.x image |
| IPluginV2, static libraries, cuDNN plugins removed | Not used here |
| `getNbIOTensors`, `getTensorShape`, `setTensorAddress`, `enqueueV3`, `setMemoryPoolLimit`, 2-arg `deserializeCudaEngine`, ONNX parser | Unchanged; already on the 10.x API |

Upstream rfdetr 1.10.1 has the same FP16 problem in its native TensorRT export
(roboflow/rf-detr#1453): `fp16=True` silently yields an FP32 engine on 11.

## Requirements

1. The backend compiles under `-DWERROR=ON` against both TensorRT 10.x and 11.x headers, with
   every API difference behind `NV_TENSORRT_MAJOR`.
2. 10.x behaviour is unchanged: the FP16 builder flag is still set on an `.onnx` build.
3. On 11.x the build logs that engine precision follows the ONNX model.
4. The backend refuses an engine whose I/O tensors are not float32 (the buffers are float32-only),
   so a mis-converted FP16 ONNX fails loudly instead of corrupting memory.
5. CI compile-checks the 11.x path on every PR, from a pin in `versions.env`.
6. `export_trt.sh` works on containers with either `trtexec`.
7. Docs explain the 11.x FP16 route (ModelOpt AutoCast with `--keep_io_types`).

## Pin bump to the 26.08 stack (added 2026-09-24)

Originally out of scope; brought in on request. The pinned stack moves to what
`nvcr.io/nvidia/tensorrt:26.08-py3` ships, with the previous stack kept compiling.

8. `versions.env` pins `TENSORRT_VERSION=11.2.1.2`, `CUDA_VERSION=13.3` (the CUDA series NVIDIA
   builds 11.2.1.2 for), `NGC_CONTAINER_TAG=26.08` and `DALI_VERSION=2.2.0` (the DALI in
   `tritonserver:26.08-py3`); `dockerfile.trt` follows through its `ARG` defaults.
9. Backward compatibility: TensorRT 10.x and DALI 1.x remain supported. `versions.env` gains
   `TENSORRT_LEGACY_VERSION=10.13.3.9` and `DALI_LEGACY_VERSION=1.51.2`, and `gpu-compile.yml`
   compiles the full GPU pipeline against them (`TRT_HEADERS=legacy`) on every PR.
10. The TensorRT download works for both majors: `cmake/deps/packages/TensorRT.cmake` derives the
    10.x `….tar.gz` or 11.x `TensorRT-Enterprise-…-Release-external.tar.zst` archive name from
    `TENSORRT_VERSION`.
11. No prose outside the README version tables states a pinned value; `docs/` and `specs/` name the
    `versions.env` variable, commands read it through `scripts/versions.sh`, and
    `check_version_sync.sh` verifies the README tables.

## Out of scope

- An in-repo FP16 conversion script (would add an unverified `nvidia-modelopt` pin).
- Dropping TensorRT 10.x or DALI 1.x.
