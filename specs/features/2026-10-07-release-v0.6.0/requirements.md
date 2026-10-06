# Release v0.6.0

Minor release from `develop`. Three changes break existing builds, so before 1.0 this is a minor
bump, not a patch:

- TensorRT 8.x–10.x and DALI 1.x support removed; the TensorRT backend requires TensorRT 11.
- `-DUSE_GPU_PIPELINE=ON` builds CUDA preprocessing (nvJPEG + kernel), not DALI. DALI needs
  `-DUSE_DALI=ON`.
- `dockerfile.trt` `GPU_PIPELINE=cuda` is retired and fails the build; its replacement is `post`.

Contents since v0.5.1: Phase 6 (CUDA preprocessing, DALI as the alternative, PR #18), the
TensorRT 11 / NGC 26.08 stack (PR #16), the rfdetr 1.11.2 export alignment (PR #19), ONNX Runtime
1.28.0 and the stale download-root fix (PR #21), and agent tooling (PR #20, no CHANGELOG entry).

## Scope

- Pre-release verification of the items the merged work left unrun, except `--display`: a
  greyscale (and CMYK) JPEG through the CUDA preprocessor, and
  `model.export(format="tensorrt", fp16=True)` end to end into the C++ TensorRT backend.
- Set the project version to 0.6.0 in `CMakeLists.txt`, `vcpkg.json`, the README badge and
  `specs/tech-stack.md`.
- `CHANGELOG.md`: a migration note for the three breaking changes at the top of `[v0.6.0]`, and a
  Known issues section that covers the nvJPEG decoder gap as well as DALI's.
- Record v0.6.0 in the `specs/roadmap.md` Status section; cut with git-flow, publish on approval.

Out: `--display` playback (no display on the verification machine, stated as unverified), and the
unfinished `feature/performance-memory-investigation` branch.

## Decisions

- FP16-engine tolerance against the FP32 engine, which no earlier spec defined: same classes and
  count, score Δ ≤ `0.01`, box centre ≤ 1 px, mask IoU ≥ `0.98`. The ONNX-cast runs of the
  1.11.2 alignment measured score Δ ≤ 0.004.
- The greyscale and CMYK checks use committed JPEG fixtures and compute the CPU reference in the
  test from the same file, so no new golden `.bin` is committed.
