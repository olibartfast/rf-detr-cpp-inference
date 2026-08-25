# Plan: RF-DETR 1.9.3 / 1.9.4 Alignment

Four groups. Group 1 is the contract and must land before 2 and 3, which are independent of each
other. Group 4 is documentation and pins, required in the same commit by the Spec Sync rule in
`AGENTS.md`.

No new build targets, so nothing needs `CMakeLists.txt` registration: every file edited is already
listed (`rfdetr_inference_lib` sources at `CMakeLists.txt`, `unit_tests` at the same file's test
section). No new dependency — `specs/tech-stack.md` is unchanged except for the export pin row.

## Group 1 — The shared selection contract (no GPU)

1. `src/processing_utils.hpp` — declare `resolve_background_slot()`, `foreground_class_count()`,
   `slot_for_foreground_column()`, `build_foreground_scores()`, `select_topk_multiclass()`. Document
   each as a mirror of its upstream counterpart (`export/_class_layout.py`, `export/_topk.py`).
2. `src/processing_utils.cpp` — implement them. `resolve_background_slot()` throws
   `std::invalid_argument` on an out-of-range slot (upstream raises `ValueError`); NumPy negative
   indexing supported. `select_topk_multiclass()` uses `std::partial_sort` over a NaN→`+inf`
   ranking key with the comparator `key desc, index asc`.
3. `src/rfdetr_inference.hpp` — add `Config::background_class_id` (`std::optional<int>`, default
   `0`) and a reusable `score_grid_` member so the video path does not reallocate per frame.

## Group 2 — CPU decode paths (no GPU)

4. `src/rfdetr_inference.cpp` — anonymous-namespace `validate_config()`, called from both
   constructors, rejecting a negative `max_detections`.
5. `src/rfdetr_inference.cpp::postprocess_outputs()` — replace the per-query argmax with
   `build_foreground_scores()` + `select_topk_multiclass()`; threshold with `score > threshold`;
   decode `query = flat / C_fg`, `class_id = flat % C_fg`.
6. `src/rfdetr_inference.cpp::postprocess_segmentation_outputs()` — same selection, replacing the
   local score/index vectors and the score-only `partial_sort` comparator.
7. `src/rfdetr_inference.cpp::postprocess_keypoint_outputs()` — same selection; recover the exported
   slot with `slot_for_foreground_column()` for the `kp_map` lookup, which is slot-indexed.
8. `src/main.cpp` — `--background-class-id <n|none>`, usage text, and the two-level "unset vs.
   explicit none" parse.

## Group 3 — CUDA mirror (**needs a GPU to verify**, compiles without one)

9. `src/gpu/rfdetr_postprocess.hpp` — `SegPostprocessParams::background_slot`, documented as
   "fill from `resolve_background_slot()`".
10. `src/gpu/rfdetr_postprocess.cu` — `DeviceParams` gains `background_slot` and `num_foreground`;
    `decode_scores` compacts the foreground columns before the sort; `select_and_decode` decodes
    against `num_foreground` and drops NaN via `!(score > threshold)`; `Impl` sizes its scratch by
    the foreground count and rejects an out-of-range slot.
11. `src/rfdetr_inference.cpp::postprocess_segmentation_outputs_gpu()` — fill `background_slot`
    from the same helper the CPU path uses.

## Group 4 — Tests, pins, documentation (no GPU)

12. `tests/unit/test_rfdetr_inference.cpp` — a `Config`-taking `make_inference()` overload, then one
    test per new rule: multi-label, descending order, tie order, cap-before-threshold, NaN dropped,
    `background_class_id` none / negative / out-of-range, negative `max_detections`, and the
    segmentation path sharing the selection.
13. `deploy/requirements.txt`, `deploy/export_executorch.py`,
    `tests/integration/integration_test_rfdetr_inference.cpp` — pin and message text to 1.9.4.
14. `specs/tech-stack.md` (export-tooling row), `specs/mission.md` ("currently **1.9.4**").
15. `README.md` — `--background-class-id` in both option tables, and a "How detections are selected"
    section covering multi-label ranking, cap-then-threshold order, and the tie rule.
16. `docs/export.md` — what 1.9.2–1.9.4 do and do not change for an exported model.
17. `CHANGELOG.md` — `[Unreleased]` entry in house style, after the 1.9.2 entry.
18. `AGENTS.md` — the fetch-before-work rule (see `requirements.md` → Context).
