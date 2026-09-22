# Requirements — Performance and Memory Investigation

Spec: `specs/features/2026-09-22-performance-memory-investigation/`

Branch: `feature/performance-memory-investigation`

## Goal

Give maintainers a repository-specific procedure for locating CPU performance bottlenecks,
interpreting Linux `perf` evidence, diagnosing memory spikes, and selecting optimization
experiments without claiming improvements that have not been measured.

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

## Out of Scope

- Production-code optimization, benchmark harness implementation, dependency installation,
  GPU rental, or changes to inference results.
- Claims about a measured bottleneck, speedup, memory reduction, or leak without a supplied
  representative model/input and recorded before/after evidence.
- Changes to full-frame mask semantics, top-k selection, threshold behavior, backend
  interfaces, or ring-buffer ownership.
- Updating the product roadmap or `CHANGELOG.md`; this phase changes maintainer guidance and
  feature specifications only.

## Decisions

- [D-1] Store every deliverable in this feature directory. The repository's `AGENTS.md`
  requires feature packets under `specs/features/`, overriding the external skill's generic
  `specs/YYYY-MM-DD-.../` example.
- [D-2] Treat this as an investigation/documentation phase. The user's verbs are “explain”
  and “propose”; implementation follows only after measurement identifies a target.
- [D-3] Use placeholders for model, video, and labels. No representative workload was
  supplied, so an arbitrary local file must not become an implied benchmark standard.
- [D-4] Separate startup-inclusive process measurements from steady-state measurements.
  Repeated process launches do not remove model loading or first-frame costs.
- [D-5] Delegate source review and guide drafting as separate bounded tasks. This provides
  fresh-context reviewability; because the native agents inherit the session model, it is
  not presented as a cost-saving model-tier comparison.

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
- Assumption [A-1]: the immediate deliverable is a reusable maintainer guide rather than
  measurements on this host. Basis: no model/video/target budget was supplied.

## Definition of Done

- [x] R-1 through R-6 map to executed validation evidence.
- [x] `profiling-guide.md` contains no fabricated runtime result or unsupported optimization
  claim.
- [x] Only this feature directory changes on the feature branch.
