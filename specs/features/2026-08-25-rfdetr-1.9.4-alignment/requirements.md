# Requirements: RF-DETR 1.9.3 / 1.9.4 Alignment

The standing obligation from [roadmap.md](../../roadmap.md) → Deferred: every upstream `rfdetr`
release triggers an alignment pass. Follows the
[rfdetr-alignment](../../../.claude/skills/rfdetr-alignment/SKILL.md) checklist. The 1.9.2 pass is
already recorded in [CHANGELOG.md](../../../CHANGELOG.md); this covers
[1.9.3](https://github.com/roboflow/rf-detr/releases/tag/1.9.3) and
[1.9.4](https://github.com/roboflow/rf-detr/releases/tag/1.9.4).

Unlike 1.9.2, these two **do** change the decode contract, so this is not a pins-and-docs pass.

## Step 1 — Classification (skill Step 1)

| Question | Answer |
|----------|--------|
| Exported model **inputs** change? | **No.** `export/_resize.py` is byte-identical between 1.9.1 and 1.9.4. `src/media.cpp` and `data/dali/` do not move. |
| Exported **outputs** change? | **No** in shape/order/count — but their **decode** does. Tensor names, ranks, and positions are unchanged, so `validate_output_order()` in every backend is untouched. Every postprocess path changes, `src/gpu/rfdetr_postprocess.cu` included. |
| Required **runtime operators** change? | **No.** `export/_executorch/` is byte-identical 1.9.1→1.9.4; ExecuTorch stays pinned at v1.4.0 and the `EXECUTORCH_BUILD_KERNELS_OPTIMIZED` requirement is unchanged. |
| Public Python APIs used by `deploy/` change? | **No.** The new `background_class_id` / `rank4_output` arguments sit on private `_run_inference` helpers this project does not call. |
| Training/dataset-only change? | **Partly** — most of both releases is. The exceptions are the two decode changes below. |

### The two changes that reach C++

1. **Multi-label selection** ([1.9.3 PR #1320](https://github.com/roboflow/rf-detr/pull/1320)).
   RF-DETR scores classes with independent sigmoids, not a softmax, so one query can clear the
   threshold on several classes at once. Upstream's ONNX/TFLite reference decoders took a per-query
   `argmax` and silently dropped the rest; 1.9.3 replaced that with a flattened *(Q, C)* ranking
   (new `rfdetr/export/_topk.py`), and changed `PostProcess._select_topk` from `torch.topk` to a
   stable `argsort` so ties resolve as descending score, then ascending flattened index.
   **This project had the same argmax bug** in `postprocess_outputs()` and
   `postprocess_keypoint_outputs()`.
2. **Explicit background logit slot** ([1.9.4 PR #1397](https://github.com/roboflow/rf-detr/pull/1397)).
   The background slot is checkpoint-dependent and cannot be inferred from tensor width, so it
   became an argument (new `rfdetr/export/_class_layout.py`). **This project hard-coded slot 0.**

## Scope

### In

| Deliverable | Path |
|-------------|------|
| Shared selection + class-layout helpers | `src/processing_utils.{hpp,cpp}` |
| Detection, segmentation, keypoint decode | `src/rfdetr_inference.cpp` |
| `Config::background_class_id` | `src/rfdetr_inference.hpp` |
| CUDA mirror of the same selection | `src/gpu/rfdetr_postprocess.{hpp,cu}` |
| `--background-class-id` flag | `src/main.cpp` |
| Unit coverage for every new rule | `tests/unit/test_rfdetr_inference.cpp` |
| Pins | `deploy/requirements.txt`, `deploy/export_executorch.py`, `specs/tech-stack.md`, `specs/mission.md` |
| Documentation | `README.md`, `docs/export.md`, `CHANGELOG.md` |

### Out

- **No re-export of the checked-in TensorRT engine.** `exports/model.engine` stays as is; the graph
  did not change, only its host-side decode.
- **No DALI pipeline regeneration.** Preprocessing is untouched, so `data/dali/*.dali` are still valid
  ([gpu-pipeline.md](../../gpu-pipeline.md) rules 1–2 are not engaged).
- **No `rank4_output` equivalent.** Upstream needed it because its TFLite helper guessed a mask from
  any lone rank-4 output. This project selects outputs positionally under an explicit
  `--segmentation` / `--keypoint` mode and never guesses, so there is nothing to port.
- **No change to the 1.9.2 dataset/label-space work.** Python-side, covered by its own entry.

## Decisions

- **`background_class_id` defaults to `0`, not upstream's `-1`.** Upstream's own release notes state
  that `-1` mis-decodes the official pretrained COCO weights, because a real foreground category
  (id 90, `toothbrush`) occupies the final slot. This project's convention — logit 0 background,
  logit *n* = COCO category *n*, `data/coco-labels-91.txt` indexed by COCO id — is upstream's
  `background_class_id=0` case. Defaulting to `0` keeps every decoded label exactly where it was.
- **The background column is excluded *before* ranking, not filtered after.** Upstream's
  `_exclude_background_class` runs ahead of `_select_topk_multiclass`. Filtering afterwards (what the
  segmentation path used to do) lets background candidates consume slots of the `num_select` cap.
- **After exclusion, foreground column *c* is label index *c*.** Upstream keeps original exported
  ids and maps them to names through a sparse dict; this project's label file is already id-indexed,
  so the positional mapping is equivalent and preserves the existing `class_id` values.
- **NaN ranks first, then fails the threshold.** Matches torch's descending-sort behaviour, which
  `_topk.py` documents and reproduces with `np.where(np.isnan(...), np.inf, ...)`. It also makes the
  comparator a strict weak ordering — the previous `a > b` on raw scores was undefined behaviour in
  `std::partial_sort` the moment any score was NaN.
- **One implementation, four call sites.** Per [mission.md](../../mission.md) architectural
  commitments, the CPU and CUDA postprocessors are two implementations of one contract, so the rule
  lives in `processing_utils.cpp` and the kernel mirrors it rather than re-deriving it.

## Context

- **This spec was written after the implementation**, contrary to the skill's ordering. The work
  began from a local `develop` that was 4 commits behind `origin/develop` and therefore did not yet
  contain `specs/` or the skills. Recorded here rather than quietly backdated. The fetch-first rule
  added to `AGENTS.md` in this same change is the fix for the cause.
- Existing patterns followed: `src/processing_utils.cpp:22-33` (small, `noexcept`, pure decode
  helpers), `src/gpu/rfdetr_postprocess.cu:58-80` (one thread per (query, class) pair, `DeviceParams`
  passed by value).
- The CUDA path already satisfied the tie rule by accident: CUB's radix sort is stable, so equal
  scores kept the ascending flattened index order `decode_scores` wrote them in. That is now written
  down beside the sort so it is not lost to a future switch to an unstable sort.
- Upstream sources for both mirrors, read from the 1.9.4 sdist: `rfdetr/export/_topk.py`,
  `rfdetr/export/_class_layout.py`, `rfdetr/models/postprocess.py::PostProcess._select_topk`.
- Open question carried forward: `Config::keypoint_counts` is indexed by exported logit slot while
  `class_id` is now a label index. They are correctly distinguished in
  `postprocess_keypoint_outputs()` via `slot_for_foreground_column()`, but a non-background-first
  keypoint checkpoint would also need `keypoint_counts` re-based. No such checkpoint ships today.
