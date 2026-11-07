# Plan

1. No GPU: commit/push alignment after mandatory local gates; require all PR checks green and merge to develop.
2. No GPU: branch release/v0.7.1 from current remote develop; reconcile all four project versions and propagate tech-stack metadata to AGENTS/open spec; move Unreleased to dated release; update roadmap baseline and known issues.
3. No GPU: run pins, format, clang-tidy, cppcheck, strict build/unit tests; commit/push and open release PR into master.
4. No GPU: merge only after all checks pass, verify master checks; tag the green master commit, then require tag lint green before publishing release notes.
5. No GPU: fresh-clone tag build/unit tests; merge master back to develop, preserve local divergent refs, record published outcomes, and delete merged feature/release branches.

No CMake target registration or GPU implementation changes are needed.
