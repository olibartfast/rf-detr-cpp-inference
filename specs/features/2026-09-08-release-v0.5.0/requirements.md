# Release v0.5.0

Recover the latest OpenCode session `ses_f7f65c738ffebDEmj0hT3V6yYF` and prepare
v0.5.0 from current develop (938dd6f), preserving the older dirty preparation
worktree at /tmp/opencode/rfdetr-v050-run1.

- Include the already-landed rfdetr 1.10.1 alignment and Phases 1-3 work.
- Repair the recovered ONNX catalog blocker: unsupported automatic-download
  targets must still permit disabled ONNX and provided-prefix/package-manager use.
- Keep the tech-stack, README and AGENTS dependency-provider descriptions aligned;
  document the existing six presets accurately.
- Correct the GPU design reference to the existing global top-k contract.
- Reconcile CMake, vcpkg and README to 0.5.0 and cut the changelog at release time.
- Preserve the incomplete Phase 4 sanitizer and engine benchmark checks as UNRUN.
  A current release-gate decision is pending; the latest session stopped on these
  gates, while an older preparation session recorded a waiver.
- Release from develop using Git-flow; validate before publishing master/develop/tag.

Third-party pins remain in versions.env. No GPU implementation changes are intended.
