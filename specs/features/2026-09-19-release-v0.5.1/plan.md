# Plan

1. Fetch and fast-forward `develop` to `origin/develop` (b45ebf4) before touching anything.
2. Create `release/v0.5.1` from `develop`.
3. Bump the project version to 0.5.1 in `CMakeLists.txt`, `vcpkg.json`, and the README badge.
4. Cut `CHANGELOG.md`: move `[Unreleased]` to `[v0.5.1]` with the date, open a fresh
   `[Unreleased]`, and repoint the compare links.
5. Record v0.5.1 in the `specs/roadmap.md` Status section.
6. Verify: `./scripts/check_version_sync.sh`, a clean default Release build, unit and
   integration tests, format check, and `git diff --check`.
7. Merge `release/v0.5.1` into `master` with `--no-ff`, tag `v0.5.1`, back-merge `master`
   into `develop`, and delete the release branch.
8. Push `master`, `develop`, and the tag only after user approval.
