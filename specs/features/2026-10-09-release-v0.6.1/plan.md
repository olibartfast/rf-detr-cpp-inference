# Plan: Release v0.6.1

No GPU needed; GPU evidence predates the release (T4 and L4, Colab).

1. Delete tag `v0.7.0` and its GitHub release; reset `master` and `develop` to their pre-v0.7.0 heads.
2. Branch `release/v0.6.1` from `develop` at `d59f571`.
3. Version statements to 0.6.1; `./scripts/check_version_sync.sh`.
4. Rewrite `CHANGELOG.md`; record the changelog style and the bump-level rule in `AGENTS.md`.
5. Default build and unit tests.
6. Merge to `master` (`--no-ff`), tag `v0.6.1`, merge back to `develop`, push, publish the GitHub release.
