# Validation: Release v0.6.1

| Check | Result |
|---|---|
| Version statements agree (CMake, vcpkg, README badge, tech-stack) | PASS: 0.6.1; `CMAKE_PROJECT_VERSION` 0.6.1 in a fresh configure |
| `./scripts/check_version_sync.sh` | PASS |
| Default ONNX Runtime build + unit tests | PASS: 103/103 |
| CI on the released content (`d59f571`, PR #26) | PASS: all 15 jobs |
| GPU paths | PASS, recorded in `specs/features/2026-10-08-colab-gpu-gate/validation.md` (T4, L4) |

## Not verified

- `--display` on a real screen and on the DALI build (open Phase 7 item).
- ExecuTorch not rebuilt; `src/backends/executorch_backend.cpp` unchanged since Phase 7's V-15.
