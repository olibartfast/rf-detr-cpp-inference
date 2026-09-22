# Requirements — Performance and Memory Investigation

Spec: `specs/features/2026-09-22-performance-memory-investigation/`

Branch: `feature/performance-memory-investigation`

## Goal

Profile the RF-DETR inference application on this host with a fixed, reproducible workload,
identify measured CPU bottlenecks, quantify memory behavior, and rank optimization work from
recorded evidence. The guide is supporting material; metrics are the deliverable.

## In Scope

- [R-1] Explain reproducible `perf stat`, `perf record`, and `perf report` workflows for a
  representative RF-DETR image or video workload.
- [R-2] Define the relevant metrics and their limits: elapsed time, task-clock, IPC, branch
  miss rate, cache events, context switches, migrations, faults, sampling attribution, and
  counter multiplexing.
- [R-3] Explain native peak RSS, live heap, allocation churn, retained capacity,
  fragmentation, leaks, mapped pages, and GPU memory as distinct measurements.
- [R-4] Provide native RSS, Massif, and allocation-profiler procedures, with the correct
  build type and sanitizer constraints.
- [R-5] Prioritize source-backed performance and memory hypotheses, with a controlled
  experiment, expected evidence, competing explanation, and correctness risk for each.
- [R-6] Record tool availability and every unrun check honestly; estimates must be labeled
  as estimates.
- [R-7] Execute the application with the checked-in ONNX model, image, and labels, plus a
  generated fixed-length video derived from the checked-in image.
- [R-8] Record repeated wall time, throughput, task-clock, cycles, instructions, IPC,
  branches, branch misses, faults, context switches, and migrations for the fixed workload.
- [R-9] Record a sampled call graph that identifies the highest-cost application symbols.
- [R-10] Record native maximum RSS and a heap profile with peak allocation owners.
- [R-11] Publish the exact environment, commands, raw artifact locations, summarized
  metrics, interpretation, and remaining uncertainty in `results.md`.

## Out of Scope

- Production-code optimization, benchmark harness implementation, GPU rental, or changes to
  inference results.
- Speedup or memory-reduction claims without a separately measured candidate implementation.
- Changes to full-frame mask semantics, top-k selection, threshold behavior, backend
  interfaces, or ring-buffer ownership.
- Updating the product roadmap or `CHANGELOG.md`; this phase changes maintainer guidance and
  feature specifications only.

## Decisions

- [D-1] Store every deliverable in this feature directory. The repository's `AGENTS.md`
  requires feature packets under `specs/features/`, overriding the external skill's generic
  `specs/YYYY-MM-DD-.../` example.
- [D-2] Treat measured profiling as the completion gate. Documentation alone cannot complete
  the feature.
- [D-3] Use `output_detection/inference_model.onnx`, `data/dog.jpg`, and
  `data/coco-labels-91.txt`. Generate a deterministic video from `data/dog.jpg` outside the
  repository so the four-stage video pipeline executes without adding a binary fixture.
- [D-4] Separate startup-inclusive process measurements from steady-state measurements.
  Repeated process launches do not remove model loading or first-frame costs.
- [D-5] Delegate measurement-protocol review and profiling execution as separate bounded
  tasks. Native agents inherit the session model, so delegation provides isolation and
  reviewability rather than a claimed cost saving.

## Constraints

- Preserve the architectural commitments in `specs/mission.md` and the eight-rule model
  contract in `specs/gpu-pipeline.md`.
- Use an optimized, symbolized build for CPU profiling and a plain unsanitized Debug build
  for Valgrind.
- No new dependency or version change.
- Existing edits in the main `develop` checkout are user-owned and must remain untouched.
- Agent path restrictions are advisory in this harness and must be checked with the final
  diff; the isolated worktree is the enforced filesystem boundary.

## Context

- Finding: the application uses a four-stage bounded video pipeline; see
  `src/video_pipeline.hpp` and `src/video_pipeline.cpp`.
- Finding: segmentation uses full-frame masks and CPU/GPU numerical parity is an explicit
  project invariant; see `specs/mission.md` and `specs/gpu-pipeline.md`.
- Finding: Google Benchmark is optional through `-DBENCHMARKS=ON`; existing coverage must
  be inspected before stating what it measures.
- Finding: the checked-in model, image, and labels provide an executable detection workload.
- Assumption [A-1]: a generated repeated-frame video is suitable for measuring pipeline
  mechanics and stable throughput, but not scene-complexity sensitivity. Results must say so.

## Definition of Done

- [ ] R-1 through R-11 map to executed validation evidence.
- [ ] `results.md` contains the required measured metrics and raw artifact locations.
- [ ] `profiling-guide.md` contains no fabricated result or unsupported optimization claim.
- [ ] Only this feature directory changes on the feature branch.
