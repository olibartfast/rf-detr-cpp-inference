# Plan

1. Reproduce: compile `tensorrt_backend.cpp` against NVIDIA/TensorRT `v11.3` headers — fails on
   `BuilderFlag::kFP16` (no GPU).
2. Backend: set `kFP16` on `NV_TENSORRT_MAJOR == 10` only; log the strongly-typed build on `>= 11`;
   reject non-float32 I/O tensors on `>= 10` (no GPU).
3. CI: `TENSORRT_COMPAT_VERSION` in `versions.env`; `TRT_HEADERS=compat` in
   `scripts/ci/stage_gpu_headers.sh` stages its headers from the OSS tag `v<major>.<minor>` with
   a sparse, blobless clone; a `gpu-compile.yml` matrix entry builds the full pipeline against them
   with its own header cache key (no GPU).
4. `export_trt.sh`: pass `--fp16` only when `trtexec --help` lists it (no GPU).
5. Docs: `docs/export.md` "TensorRT 11 and FP16", `docs/advanced-usage.md`, `docs/building.md`,
   README versions table, `specs/tech-stack.md`, `AGENTS.md`, `CHANGELOG.md` (no GPU).
6. Validate per `validation.md`; the runtime half is UNRUN until a gpu-verify pass on 11.x.
