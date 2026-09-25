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

Pin bump (requirements 8–11, added 2026-09-24):

7. `TensorRT.cmake`: pick the archive name by `TENSORRT_VERSION` major; check both URL forms
   resolve and configure through the download path with the cached 11.2.1.2 archive.
8. `versions.env` + `dockerfile.trt` `ARG` defaults to the 26.08 stack; `*_LEGACY_VERSION` pins.
9. `stage_gpu_headers.sh`: `TRT_HEADERS=legacy` (TensorRT from apt, `+cuda<v>` suffix looked up in
   the repo index; DALI legacy wheel); DALI wheel name looked up from the `cuda130` index since the
   manylinux tag differs between 1.x and 2.x. `gpu-compile.yml`: legacy matrix entry.
10. Prose: replace pinned values with variable names; extend `check_version_sync.sh` to the README
    tables; update `AGENTS.md`, `specs/tech-stack.md`, `specs/mission.md`, the release and
    rfdetr-alignment skills.
11. Docker gate on `dockerfile.trt` (26.08 images) and gpu-verify on the local RTX 3060 Laptop,
    including the checked-in `.dali` pipelines under DALI 2.2.0.
12. DALI 2.x `dlopen()`s nvImageCodec and its codec libraries (nvJPEG, nvCOMP, nvJPEG2000, nvTIFF)
    from sibling wheel directories that ldd cannot see. `dockerfile.trt`'s `dali-fetch` stage and
    `scripts/fetch_dali.sh` flatten them next to `libdali.so`, keeping `DALI_ROOT` a single
    directory; DALI 1.x (no `nvimgcodec/`) is staged as before.

One version only (requirements 12–15, added 2026-09-26):

13. Remove the three extra pins, the `TRT_HEADERS` staging modes, the compat/legacy matrix entries
    and their `check_version_sync.sh` expectations.
14. Resolve every `NV_TENSORRT_MAJOR` conditional in `tensorrt_backend.{hpp,cpp}` to the 11.x
    branch; add the `< 11` `#error`; `TensorRT.cmake` keeps the 11.x archive name and errors below 11.
15. `export_trt.sh` drops the `--fp16` probe; `fetch_dali.sh` and `dockerfile.trt` drop the
    DALI 1.x guard; Docker gate on all eight `dockerfile.trt` combinations.
16. Prose: README, `docs/{advanced-usage,building,export}.md`, `AGENTS.md`, `specs/tech-stack.md`,
    `CHANGELOG.md`.
