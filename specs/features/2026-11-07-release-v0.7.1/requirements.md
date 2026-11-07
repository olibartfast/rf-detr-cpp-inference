# Release v0.7.1

## Scope
Release the RF-DETR 1.12.0 alignment after its CI passes. The maintainer selected v0.7.1 and authorized the release on 2026-11-07.

| In | Paths |
|---|---|
| Reconcile project version | CMakeLists.txt, vcpkg.json, README.md, specs/tech-stack.md |
| Same-commit constitution propagation | AGENTS.md, open alignment spec |
| Release notes and status | CHANGELOG.md, specs/roadmap.md |
| Release procedure and evidence | this directory |

Out: runtime feature work, new dependencies, revalidation on unavailable GPU/ExecuTorch hardware, and closing Phase 7 display work.

## Decisions
- Patch release selected explicitly by maintainer; no new native runtime feature or native tensor contract.
- Gate on all alignment CI checks, then all release checks. Tag only after release/master checks are green; publish GitHub release only after tag lint is green.
- Preserve known issues and record exact UNRUN limitations. Phase 7 display remains open under the earlier release decision; it is not a gate for this alignment-only patch.
- Follow git-flow from remote develop, preserving the divergent local develop/master histories in named archive branches if local synchronization is needed.
- No runtime pins or Docker build options change; Python exporter requirements are not installed by the backend Dockerfiles.

## Context
Baseline v0.7.0, alignment PR #27. Official upstream release 1.12.0 exists (published 2026-11-06). See ../2026-11-07-rfdetr-1.12.0/validation.md for real-model evidence. Mission architectural commitments are unchanged.
