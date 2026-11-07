# Validation

## Automated — no GPU
- [x] Alignment PR #27: Build & Test, C++ Lint & Build, GPU Backend Compile all green.
- [x] Version statements agree; ./scripts/check_version_sync.sh and Dockerfile parity pass.
- [x] Mandatory format 18, clang-tidy 18, exhaustive cppcheck before every commit and push.
- [x] Strict default build and ctest --test-dir build --output-on-failure.
- [x] Release PR and master checks all green.
- [x] Tag lint all green before GitHub release publication.
- [x] Fresh clone of tag builds with default backend and unit tests pass.

## Device and alternate-runtime limits
UNRUN: TensorRT/CUDA/DALI runtime, GPU parity and compute-sanitizer (no NVIDIA device/toolkit); regenerated ExecuTorch programs on native pinned runtime (no install prefix). New ONNX Nano and LiteRT detection/segmentation/pose real-model checks passed in alignment validation. No new GPU/C++ runtime code is introduced.

## Manual
- [x] README/build options/backend constraints and known issues reviewed.
- [x] Tagged release notes disclose unverified runtimes and retained Phase 7 display gap.

## Definition of done
- [x] Version bump merged to master; annotated v0.7.1 tag and GitHub release published after required CI.
- [ ] Push this master-to-develop merge and verify develop CI; local branch synchronization follows that check. Remote feature/release branches have been deleted.

## Evidence
Maintainer approved release after green CI and selected v0.7.1. Results are added when observed; unchecked outcomes are not claimed complete.

Alignment merged as df0c85a after all 16 checks succeeded on PR #27. CI runs: [lint](https://github.com/olibartfast/rf-detr-cpp-inference/actions/runs/42400574451), [build/test](https://github.com/olibartfast/rf-detr-cpp-inference/actions/runs/42400574456), [GPU compile](https://github.com/olibartfast/rf-detr-cpp-inference/actions/runs/42400574617). The GPU run initially failed on an HTTP/2 protocol error downloading DALI headers; rerunning the failed job passed without a code change.

Release pre-commit format/clang-tidy 18.1.8, exhaustive cppcheck, pin sync, Dockerfile parity, strict build and all 11 CTest entries pass locally. Logs: /tmp/rfdetr-release-precommit-*.log and /tmp/rfdetr-v071-{configure,build,tests}.log.

## Publication observed

- Release PR #28: all 16 checks passed. Release commit b7e5d34, merged master commit 548092693c6cdcecbeb007e60ed2e85fc12d0201.
- Master CI: [build/test 42401377947](https://github.com/olibartfast/rf-detr-cpp-inference/actions/runs/42401377947), [GPU compile 42401377939](https://github.com/olibartfast/rf-detr-cpp-inference/actions/runs/42401377939), [lint 42401377991](https://github.com/olibartfast/rf-detr-cpp-inference/actions/runs/42401377991) all succeeded before tagging.
- Annotated tag v0.7.1 points to that master commit. All five tag lint jobs passed in [42403680722](https://github.com/olibartfast/rf-detr-cpp-inference/actions/runs/42403680722) before GitHub release publication.
- [v0.7.1 published](https://github.com/olibartfast/rf-detr-cpp-inference/releases/tag/v0.7.1), with explicit alternate-runtime limitations and the known LiteRT validation-environment workaround.
- Fresh clone `/tmp/rfdetr-v071-fresh` checked out the tag, configured/built under WERROR with the default ONNX backend and already-downloaded matching dependency prefixes, and passed all 11 CTest entries plus detection integration with the new model. Logs `/tmp/rfdetr-v071-fresh-{build,tests,model}.log`.
- Remote feature/rfdetr-1.12.0 and release/v0.7.1 branches were deleted after publication. The current merge commit records the completed publication and brings master back to develop; its push/CI are the remaining external steps when this snapshot is committed.
