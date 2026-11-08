# Requirements — release v0.7.1

Implements the [release checklist](../../AGENTS.md#checklist-release). Supersedes
[`2026-11-07-release-v0.7.1`](../2026-11-07-release-v0.7.1/), whose tag and GitHub release were
withdrawn.

## Scope

| In | Path |
|----|------|
| Version statement reconciliation (already at `0.7.1`) | `CMakeLists.txt`, `vcpkg.json`, `README.md`, [`specs/tech-stack.md`](../../tech-stack.md) |
| `[Unreleased]` → `[v0.7.1]` | `CHANGELOG.md` |
| Release baseline and status correction | [`specs/roadmap.md`](../../roadmap.md) |
| Release branch, master merge, annotated tag | `release/v0.7.1` |

**Out:** any change to `src/`, `cmake/`, `deploy/` or the pins. The release commit carries
documentation and version state only.

## Decisions

| Decision | Reason |
|----------|--------|
| Version **v0.7.1** | Maintainer's decision. The withdrawn attempt used the same number, and every version statement already reads `0.7.1`, so re-cutting needs no bump across the four locations. |
| Re-cut instead of bumping to `0.7.2` | The withdrawn tag never reached a published artefact that consumers could depend on; GitHub carried no `v0.7.1` release between the withdrawal and this cut. |
| GPU pipeline verified before tagging | The v0.7.1 attempt was withdrawn. Verification now runs on real hardware rather than being carried forward from CI, which cannot execute these paths ([`specs/tech-stack.md`](../../tech-stack.md)). |
| No `Known issues` section | Maintainer's decision for this release line; unverified paths are recorded here in `validation.md` instead of the changelog. |

## Context

- `develop` was in sync with `origin/develop`, working tree clean, CI green on `a6a3472` before the
  branch was cut.
- Upstream `rfdetr` is at 1.12.0, identical to `RFDETR_VERSION`, so no alignment pass is owed.
- `./scripts/check_version_sync.sh` and `./scripts/check_dockerfile_parity.sh` both pass on the
  release branch.
- The GPU evidence comes from [`2026-11-08-gpu-verify-recording`](../2026-11-08-gpu-verify-recording/)
  and the gate artefacts it names.
