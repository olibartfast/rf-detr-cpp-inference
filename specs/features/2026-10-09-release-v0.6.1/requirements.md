# Requirements: Release v0.6.1

Cut v0.6.1 from `develop` (`d59f571`) per the [release checklist](../../../AGENTS.md#checklist-release).

## Scope

In: version statements (`CMakeLists.txt`, `vcpkg.json`, README badge, `specs/tech-stack.md`),
`CHANGELOG.md`, `specs/roadmap.md` status, tag `v0.6.1`. Out: any source change.

Contents: Phase 7 (PR #24), the Colab GPU gate (PR #25), R-8's measured profile (PR #26).

## Decisions

- **Patch bump, chosen by the maintainer.** No runtime behaviour change. This release replaces
  `v0.7.0`, which was tagged as a minor bump without the maintainer's agreement and withdrawn.
- **CHANGELOG rewritten** to Keep a Changelog form, one line per change, for every release.
  Reasoning and validation records stay in `specs/features/`.
- **Released with Phase 7's `--display` item open**, by the maintainer's decision: headless
  evidence only for the CUDA build; nothing for the DALI build.
