# Keypoint schema support

Implements the keypoint item of roadmap Phase 1. User confirmed proceeding with the recommended scope: explicit CLI configuration, official Preview weights, backward compatibility.

Add `--keypoint-counts` with comma-separated non-negative counts and at least one active class. Keep the legacy `{0,17}` default. Background selection remains independently controlled by `--background-class-id`; determine the official model's layout from a real export before documenting its invocation. Reject configured counts exceeding the per-class tensor stride.

Scope: src/main.cpp, keypoint configuration validation in src/rfdetr_inference.cpp, tests/unit/test_rfdetr_inference.cpp, README, usage/export documentation and concise CHANGELOG. No GPU postprocess change, release tag, or dependency change.
