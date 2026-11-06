# Release plan

All local work needs no GPU. Hardware-specific verification may be UNRUN only with the exact reason.

1. Finish the detection/segmentation showcase, YouTube title/description and final media validation.
2. Resolve the existing cppcheck/clang-tidy findings without changing inference results.
3. Reconcile the four project version statements and create a short, user-facing v0.7.0 changelog.
4. Build every locally runnable affected Docker/backend/media combination; run CPU tests and checks.
5. Run format, clang-tidy and cppcheck immediately before committing the feature/release changes.
6. Merge feature into develop, create release/v0.7.0, finish release metadata, merge into master and tag.
7. Verify the tagged source from a fresh checkout with the default backend, then merge master back to develop.
8. Re-run mandatory lint gates before each push; push master, develop and v0.7.0 without force.
9. Verify remote refs/CI; publish a GitHub release with concise notes and explicit unverified paths.
