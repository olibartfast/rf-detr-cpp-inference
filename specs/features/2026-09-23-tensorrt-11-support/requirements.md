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

## Out of scope

- Bumping `TENSORRT_VERSION`, `NGC_CONTAINER_TAG` or `dockerfile.trt` to 11.x. That is a separate
  pin bump that needs the gpu-verify gate on real hardware, including FP16-converted parity.
- An in-repo FP16 conversion script (would add an unverified `nvidia-modelopt` pin).
