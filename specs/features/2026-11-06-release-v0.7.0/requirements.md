# Release v0.7.0 requirements

## Scope

The user requested a release bump, readable CHANGELOG, version reconciliation, commit and push.
The maintainer accepted the recommended v0.7.0 with “go”. Publication is authorized by the explicit
commit/push/release request; do not ask for the same authorization again.

| In | Paths/actions |
|---|---|
| Ship the verified LiteRT CPU backend and export helpers | Existing feature/litert-backend diff |
| Reconcile release version | CMakeLists.txt, vcpkg.json, README.md, specs/tech-stack.md |
| Human-readable release notes and known limits | CHANGELOG.md, README.md, release notes |
| Clear mandatory lint gates | src/display.cpp, src/rfdetr_inference.{hpp,cpp}, src/backends/executorch_backend.{hpp,cpp}, affected call sites |
| Verify affected Docker builds and tests | dockerfile.onnxrt, dockerfile.executorch, dockerfile.trt; no Docker redesign |
| Git-flow release and GitHub publication | Feature into develop, release branch, master tag, merge back, authorized push |

Out: dependency alignment to a newer RF-DETR release, GPU/NPU acceleration for LiteRT,
resolving the historical real-screen/DALI display verification, uploading to YouTube,
committing model binaries/video artifacts.

## Decisions

- Preserve exactly one backend and shared pre/postprocessing from [mission](../../mission.md).
- Keep dependency pins unchanged apart from the new LiteRT/runtime-converter pins already verified.
- Correct the false static-function suggestion for file preprocessing by sharing a helper that receives the
  actual preprocessing configuration from both overloads. Make genuinely stateless drawing/saving operations static;
  existing ordinary instance call syntax remains valid. The private ExecuTorch validator becomes static.
- Keep OpenCV window teardown best-effort and nonthrowing, with a line-scoped justified clang-tidy suppression.
- Continue carrying the existing GPU display verification issue; it was already accepted for v0.6.1
  and is not a new release gate for the independent LiteRT phase. Record exactly what is unverified.
- The user rejected pose predictions in the showcase. Publish a detection/segmentation-only edit;
  backend pose support remains experimental and its raw outputs must not be presented as an accuracy claim.

## Context

Current baseline is v0.6.1 on origin/develop. Fetch completed; local branch is not behind.
RFDETR_VERSION is verified against its official upstream release; no upstream alignment is included.
LiteRT backend/video validation lives in [the feature spec](../2026-11-06-litert-backend/validation.md).
AGENTS.md release and pre-commit/pre-push checklists govern this release.
