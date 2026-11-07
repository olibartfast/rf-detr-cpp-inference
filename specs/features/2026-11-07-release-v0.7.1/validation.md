# Validation

## Automated — no GPU
- [x] Alignment PR #27: Build & Test, C++ Lint & Build, GPU Backend Compile all green.
- [x] Version statements agree; ./scripts/check_version_sync.sh and Dockerfile parity pass.
- [ ] Mandatory format 18, clang-tidy 18, exhaustive cppcheck before every commit and push.
- [x] Strict default build and ctest --test-dir build --output-on-failure.
- [ ] Release PR and master checks all green.
- [ ] Tag lint all green before GitHub release publication.
- [ ] Fresh clone of tag builds with default backend and unit tests pass.

## Device and alternate-runtime limits
UNRUN: TensorRT/CUDA/DALI runtime, GPU parity and compute-sanitizer (no NVIDIA device/toolkit); regenerated ExecuTorch programs on native pinned runtime (no install prefix). New ONNX Nano and LiteRT detection/segmentation/pose real-model checks passed in alignment validation. No new GPU/C++ runtime code is introduced.

## Manual
- [x] README/build options/backend constraints and known issues reviewed.
- [ ] Tagged release notes disclose unverified runtimes and retained Phase 7 display gap.

## Definition of done
- [ ] Version bump merged to master; annotated v0.7.1 tag and GitHub release published after required CI.
- [ ] Master merged back to develop; feature/release branches deleted; local prior histories preserved.

## Evidence
Maintainer approved release after green CI and selected v0.7.1. Results are added when observed; unchecked outcomes are not claimed complete.

Alignment merged as df0c85a after all 16 checks succeeded on PR #27. CI runs: [lint](https://github.com/olibartfast/rf-detr-cpp-inference/actions/runs/42400574451), [build/test](https://github.com/olibartfast/rf-detr-cpp-inference/actions/runs/42400574456), [GPU compile](https://github.com/olibartfast/rf-detr-cpp-inference/actions/runs/42400574617). The GPU run initially failed on an HTTP/2 protocol error downloading DALI headers; rerunning the failed job passed without a code change.

Release pre-commit format/clang-tidy 18.1.8, exhaustive cppcheck, pin sync, Dockerfile parity, strict build and all 11 CTest entries pass locally. Logs: /tmp/rfdetr-release-precommit-*.log and /tmp/rfdetr-v071-{configure,build,tests}.log.
