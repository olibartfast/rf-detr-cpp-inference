# Release v0.5.1

Patch release from `develop` at b45ebf4. It ships the documentation restructuring that
already landed on develop and changes no behaviour, build option, or pin.

- Ship the documentation restructure: `README.md` becomes a quick start, the exhaustive
  reference moves to the new `docs/advanced-usage.md`, `docs/usage.md` is trimmed to the
  operational reference, and the maintainer procedures move to `specs/`. Recorded in
  `CHANGELOG.md` under `[Unreleased]`; this release names it.
- Reconcile the three project-version statements — `CMakeLists.txt` `project()`,
  `vcpkg.json`, and the README badge — to **0.5.1**, as v0.5.0's release did for 0.5.0.
- Move `[Unreleased]` to `[v0.5.1]` with the release date; open a fresh empty `[Unreleased]`
  and update the compare links.
- Update the `specs/roadmap.md` Status section to record v0.5.1 as the last tag.
- Cut with git-flow; publish only on approval.

No source, backend, build-option, or dependency pin changes. The v0.5.0 GPU parity gate
therefore remains valid and no re-verification is claimed.
