# Validation — release v0.7.1

## Automated — no GPU

- [x] `./scripts/check_version_sync.sh` — every pin restatement agrees with `versions.env`.
- [x] `./scripts/check_dockerfile_parity.sh` — 5 shared blocks agree across all three Dockerfiles.
- [x] `find src tests -name '*.cpp' -o -name '*.hpp' | xargs clang-format-18 --dry-run --Werror`.
- [x] `clang-tidy-18 -p build` over `src/` (excluding `tensorrt_backend.cpp`, as `lint.yml` does).
- [x] `cppcheck --enable=all --check-level=exhaustive` over `src/`.
- [x] Default ONNX Runtime build and `ctest --test-dir build --output-on-failure`.
- [x] `develop` CI green on the parent commit `a6a3472` before the release branch was cut.
- [x] Upstream `rfdetr` at 1.12.0, equal to `RFDETR_VERSION` — no alignment pass owed.

## Automated — with a device

Recorded in full, with hardware and driver versions, in
[`../2026-11-08-gpu-verify-recording/validation.md`](../2026-11-08-gpu-verify-recording/validation.md).

- [x] Full TensorRT + CUDA preprocessing + CUDA postprocessing build.
- [x] Full TensorRT + DALI preprocessing + CUDA postprocessing build.
- [x] Four combinations (CPU/CPU, GPU-pre/CPU-post, CPU-pre/GPU-post, GPU/GPU) run.
- [x] Four-combination parity suite on both GPU builds, measured against the tolerances
      `integration_test_gpu_parity.cpp` declares.
- [x] `compute-sanitizer` 1000-frame run — no memory findings across all frames.
- [x] Benchmark suite on the GPU build.

## Manual

- [x] README version table and build options re-read against `versions.env`, `CMakeLists.txt`,
      `CMakePresets.json` and `dockerfile.*`.
- [x] Roadmap status corrected to the release that actually exists.
- [x] Release notes published with the tag.

## Definition of done

- [x] `[Unreleased]` moved to `[v0.7.1] - 2026-11-08`, a fresh `[Unreleased]` opened.
- [x] Version statements agree at `0.7.1` in all four locations.
- [x] `release/v0.7.1` merged to `master`, tagged, merged back to `develop`, branch deleted.
- [x] `master`, `develop` and the tag pushed after maintainer approval.

## Note on tolerance divergence

The four-combination parity test passes against tolerances declared in
`tests/integration/integration_test_gpu_parity.cpp`: `{0.06, 1% of the longest side in px, 0.95}`
for any GPU preprocessor, and `{1e-3, 1 px, 0.999}` when both sides share the CPU tensor. The
measured worst case on `data/dog.jpg` is scores 0.0187, centre 2.42 px, mask IoU 0.9913 — inside the
test's bound, outside the tighter numbers the GPU-preprocess path cannot meet for the reason
documented at the `preprocess_tolerance` declaration. Recording the numbers here keeps the bound
that was actually enforced visible next to the result.

## Publication

- Tag `v0.7.1` -> `01d57f0`; `master` merged as `01d57f0`, `develop` merged back as `9163d48`.
- CI on `master`, `develop` and the `v0.7.1` tag: Build & Test, GPU Backend Compile and
  C++ Lint & Build all succeeded before the GitHub release was created.
- [v0.7.1 published](https://github.com/olibartfast/rf-detr-cpp-inference/releases/tag/v0.7.1).
- Fresh clone of the tag (`git clone --branch v0.7.1 --depth 1`) configured and built under the
  default ONNX Runtime backend, and all 11 CTest entries pass.
