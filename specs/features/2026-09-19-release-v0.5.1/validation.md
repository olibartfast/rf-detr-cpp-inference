# Validation: Release v0.5.1

Validated on 2026-09-19 on branch `release/v0.5.1`, from `develop` at b45ebf4.

- PASS: `develop` fast-forwarded to `origin/develop` (b45ebf4) before any work.
- PASS: `./scripts/check_version_sync.sh` — all restated pins agree with `versions.env`.
- PASS: default Release configure and build (`cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release`,
  `cmake --build build --parallel`).
- PASS: `ctest --test-dir build --output-on-failure -R UnitTests`.
- PASS: `RFDETR_TEST_MODEL=data/models/rfdetr-nano-1101.onnx ctest --test-dir build --output-on-failure
  -R IntegrationTests` — real ONNX Runtime model, no skips for the detection path.
- PASS: `clang-format-18 --dry-run --Werror` over `src/` and `tests/`.
- PASS: `python3 -m unittest tests/python/test_export_scripts.py` — 7 tests.
- PASS: `git diff --check`.
- PASS: `CMakeLists.txt` `project()` declares `VERSION 0.5.1`; `vcpkg.json` and the README badge agree.

## Scope

- No source, backend, build-option, or dependency pin changed. The v0.5.0 GPU parity gate remains
  valid; no GPU/TensorRT/ExecuTorch behaviour is re-verified or claimed by this release.
- No Docker-coupled file changed, so the pre-commit Docker gate does not apply.

## Remaining acceptance

- Pending: merge to `master`, tag `v0.5.1`, back-merge to `develop`, and push — only after user
  approval. The tag is not yet published.
