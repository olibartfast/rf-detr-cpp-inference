# Validation: Keypoint schema support

Validated on 2026-09-08, branch feature/phase1-known-issues.

- [x] Official export output signature inspected before implementation.
  - `deploy/export_keypoint.py` with rfdetr 1.10.1 produces `dets [1,100,4]`,
    `labels [1,100,2]`, `keypoints [1,100,34,8]` — the **background-first `[0, 17]`**
    layout (2 classes, 34 = 2×17 channels), which the default `{0, 17}` decodes
    directly. The active-first `[17]` schema only arises from a model configured
    with `num_keypoints_per_class=[17]`; the official checkpoint resizes to
    `[0, 17]` on load. The roadmap's "default export cannot decode" premise was
    therefore wrong; the flag is still needed for active-first finetunes.
- [x] Existing legacy keypoint tests pass unchanged (all prior KeypointPostprocessTest cases).
- [x] Active-first schema, invalid counts and stride checks tested.
  - `ActiveFirstSchemaDecodes` (counts `{17}`, `background_class_id=none`),
    `RejectsNegativeKeypointCount`, `RejectsAllBackgroundKeypointCounts`,
    `RejectsKeypointCountExceedingStride`.
- [x] CLI accepts valid lists and rejects malformed/missing arguments.
  - Manual run: `--keypoint-counts 0,17` decodes; `--keypoint-counts "1,x"` and
    `--keypoint-counts 0,-1` are rejected with the intended messages.
- [x] Default build and unit tests pass; formatting and git diff checks pass.
  - `cmake --build build --parallel`; `ctest -R UnitTests` (52 tests + 4 new);
    `python3 -m unittest tests/python/test_export_scripts.py` (7 tests).
- [x] Real keypoint inference completes using the documented flags.
  - `./build/inference_app /tmp/kp-export/rfdetr-keypoint.onnx data/dog.jpg labels.txt --keypoint`
    decodes without error (0 detections — no person in the image). The active-first
    `--keypoint-counts 17 --background-class-id none` invocation is covered by unit tests,
    not a real finetune export.
- [x] README, usage/export docs and CHANGELOG synchronized.

GPU and alternate backends are not inferred from ONNX Runtime validation. Roadmap completion is
marked only after merge and verification.
