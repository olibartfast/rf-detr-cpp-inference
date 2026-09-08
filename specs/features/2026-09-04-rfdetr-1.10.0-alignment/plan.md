# Plan: RF-DETR 1.10.0 Alignment

1. Verify the official release and tag diff; record the classification before implementation.
2. Update only the RF-DETR pin and its required restatement; keep all runtime pins fixed.
3. Add one shared returned-path validator. Make all four exporters accept and forward
   `--output_name`, default to stable repository stems, and print the returned artifact path.
4. Retain keypoint's compatibility copy, sourced from the returned artifact rather than a guess.
5. Add hermetic exporter tests with fake RF-DETR classes and no downloads.
6. Synchronize mission, tech stack, active roadmap wording, README, affected docs, and CHANGELOG.
7. Run version sync, exporter tests, diff checks, clean 1.10 export, default build/tests, and one
   C++ inference. If final staging also includes Docker changes, run their separate build gate.

## Stop conditions

- Stop before changing C++/CUDA/DALI code if exported inputs, outputs, or operators differ.
- Stop if `model.export()` does not return a usable path; do not add heuristic suffix discovery.
- Preserve all pre-existing dirty Docker work and stage only explicitly reviewed paths.
