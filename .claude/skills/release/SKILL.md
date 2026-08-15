---
name: release
description: Cuts a git-flow release — Spec Sync documentation checks, [Unreleased] to [vX.Y.Z] in CHANGELOG.md, version-statement reconciliation across CMakeLists.txt/vcpkg.json/README, then release branch, tag, and merge back. Trigger when the user says "cut a release", "release vX.Y.Z", "prepare the release", or invokes /release.
---

# Release

Written for any coding agent. A release is one of the cases that requires a spec directory under
`specs/features/` — write it first via [`feature-spec`](../feature-spec/SKILL.md) if the release
carries more than a version bump.

## Step 0 — Read before acting

Mandatory, in this order: `AGENTS.md`, `README.md`, `CHANGELOG.md`, `specs/roadmap.md`. If the
release includes an upstream `rfdetr` alignment, verify that release against
<https://github.com/roboflow/rf-detr> — never assume an upstream version is a local Git tag. See
[`rfdetr-alignment`](../rfdetr-alignment/SKILL.md).

## Step 1 — Confirm the gate

`specs/roadmap.md` says which phases the release is gated on. Do not cut a release with an unticked
gating phase unless the user explicitly decides to — and if they do, record that decision in the
CHANGELOG.

For anything touching TensorRT, ExecuTorch, DALI, or CUDA: **CI has never built or run it.** Run
[`gpu-verify`](../gpu-verify/SKILL.md) and the manual backend checks first, or state in the release
notes exactly what went out unverified.

## Step 2 — Reconcile the version statements

These disagree today and must be made to agree as part of the release:

| Location | Current state |
|----------|---------------|
| `CMakeLists.txt` `project()` | declares **no** version |
| `vcpkg.json` | `0.1.0` |
| `README.md` badge | `0.4.0` |

Pick the release version, set all three, and note the reconciliation in the CHANGELOG. Also check
the known pin duplications in `specs/tech-stack.md`: TensorRT is pinned in
`cmake/deps/packages/TensorRT.cmake:9` **and** hardcoded in `Dockerfile`; the Triton container tag
appears in `scripts/fetch_dali.sh`, `scripts/generate_dali_pipelines.sh`, and `export_trt.sh`.

## Step 3 — Spec Sync checklist

From `AGENTS.md`. Every box is mandatory:

- [ ] `README.md` updated in the same change as any code, build-option, backend-version, Docker, or
      export-package move
- [ ] README dependency and version statements verified against `CMakeLists.txt`,
      `CMakePresets.json`, `deploy/requirements.txt`, `Dockerfile*`, `docs/export.md`
- [ ] README lists current C++ library/runtime versions, CMake options, backend constraints, and
      the pip packages used for export tooling
- [ ] `specs/tech-stack.md` matches the files that own each pin
- [ ] `specs/mission.md` upstream-version line is current
- [ ] Any completed roadmap phase is ticked `[x]` and its heading marked `(Complete)`
- [ ] If the release intentionally needs no README change, the reason is written in `CHANGELOG.md`

## Step 4 — CHANGELOG

- Move `[Unreleased]` to `[vX.Y.Z]` with the date; open a fresh empty `[Unreleased]`.
- Review the **Known Issues** table: close what this release fixes, and leave what it does not with
  its reason intact.
- Keep the house style — prose explaining *why*, plus per-file change tables. This project does not
  generate its changelog from `git log`; a bullet per commit would lose the reasoning.

## Step 5 — Cut it

Git-flow. Confirm with the user before pushing or tagging — these are outward-facing and hard to
undo.

```bash
git checkout develop && git pull
git checkout -b release/vX.Y.Z
# version bumps + CHANGELOG commit here
git checkout master && git merge --no-ff release/vX.Y.Z
git tag -a vX.Y.Z -m "vX.Y.Z"
git checkout develop && git merge --no-ff master
git branch -d release/vX.Y.Z
```

Push `master`, `develop`, and the tag only once the user has approved.

## Step 6 — After

- [ ] Verify the tag builds clean from a fresh clone with the default backend
- [ ] `specs/roadmap.md` Status section reflects the new baseline
- [ ] Anything deferred out of this release is in the roadmap Deferred table **with its reason**
