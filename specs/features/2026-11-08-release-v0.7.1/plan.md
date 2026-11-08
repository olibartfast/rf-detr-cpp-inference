# Plan — release v0.7.1

Git-flow, branched from `develop`. No GPU is required for steps 1–4; step 5 is the hardware run and
is recorded in [`../2026-11-08-gpu-verify-recording/validation.md`](../2026-11-08-gpu-verify-recording/validation.md).

## 1. Version statements

1. `CMakeLists.txt` — `project(rfdetr_inference VERSION 0.7.1 ...)`. Already correct.
2. `vcpkg.json` — `"version-string": "0.7.1"`. Already correct.
3. `README.md` — badge and release-tag link at `v0.7.1`. Already correct.
4. `specs/tech-stack.md` — project version sentence. Already correct.
5. `./scripts/check_version_sync.sh` — verify every restatement still agrees.

## 2. Changelog

6. `CHANGELOG.md` — rename `[Unreleased]` to `[v0.7.1] - 2026-11-08`; open an empty
   `[Unreleased]` above it.

## 3. Spec sync

7. `specs/roadmap.md` — status bullet corrected to the real baseline: v0.7.1 dated 2026-11-08, with
   the GPU verification stated.
8. `specs/features/2026-11-07-release-v0.7.1/validation.md` — mark the withdrawn publication as
   withdrawn so the record stops claiming a live tag.

## 4. Pre-commit gate

9. `find src tests -name '*.cpp' -o -name '*.hpp' | xargs clang-format-18 --dry-run --Werror`
10. `cmake -S . -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON` then
    `find src -name '*.cpp' ! -name 'tensorrt_backend.cpp' | xargs clang-tidy-18 -p build`
11. `cppcheck --enable=all --check-level=exhaustive --std=c++20 --suppress=missingIncludeSystem --suppress=unmatchedSuppression --suppress=unusedFunction --error-exitcode=1 -I src src/`
12. `./scripts/check_version_sync.sh` and `./scripts/check_dockerfile_parity.sh`
13. Default build plus `ctest --test-dir build --output-on-failure` — the release commit touches no
    code, so this is a confirmation, not a regression hunt.

## 5. Hardware gate

14. Run the residual gate checks on a rented GPU and record the numbers in the recording spec.
    Tagging waits for this.

## 6. Cut

15. `git checkout master && git merge --no-ff release/v0.7.1`
16. `git tag -a v0.7.1 -m "v0.7.1"`
17. `git checkout develop && git merge --no-ff master`
18. `git branch -d release/v0.7.1`
19. Push `master`, `develop`, and the tag after the maintainer approves.
