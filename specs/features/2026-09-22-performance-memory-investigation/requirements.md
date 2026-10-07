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
- [R-12] Measure steady-state per-stage CPU cost with Google Benchmark, isolating the stages
  the process-level measurements cannot separate: full-frame CPU preprocessing at the
  workload resolution, detection decode (sigmoid over the logit grid, top-k selection,
  box conversion), and annotation rendering.
- [R-13] Report each microbenchmark with its real-time and CPU-time means, the iteration
  count, and the residual variance Google Benchmark reports, so a later candidate
  implementation can be compared against a recorded baseline rather than a recollection.
- [R-14] State explicitly whether the measured detection-decode cost supports or contradicts
  the roadmap's recorded reason for deferring GPU detection postprocessing ("300x91 sigmoids
  and a threshold - not a bottleneck"). Do not silently change that deferral.
- [R-15] Provide a first-class build path for profiling instead of hand-passed flags: a
  `PROFILING` CMake option, default `OFF`, that applies `-fno-omit-frame-pointer` (and the
  debug information perf needs to symbolize) to every compiled target, not only the
  application's own translation units. Frame-pointer preservation must reach the static
  library targets too, or call graphs terminate at the library boundary.
- [R-16] Parameterize each microbenchmark over its problem size, and measure at least one
  stage against a competing implementation of the same stage. A single absolute number is not
  a performance result; the comparison is the result. Defeat dead-store and dead-code
  elimination explicitly at every benchmark's observation point.

## Out of Scope

- Production-code optimization, GPU rental, or changes to inference results.
- Benchmark harness implementation is in scope only as additions under `tests/benchmark/`
  and their registration in `CMakeLists.txt`. No `src/` file changes; the measured code is
  the shipped code, called through its existing public headers.
- Speedup or memory-reduction claims without a separately measured candidate implementation.
- Changes to full-frame mask semantics, top-k selection, threshold behavior, backend
  interfaces, or ring-buffer ownership.
- Changing any `src/` file, altering inference results, or marking a roadmap phase complete.
- Note: this exclusion originally covered `CHANGELOG.md` as well, on the premise that the
  phase touched only maintainer guidance. Adding the `PROFILING` option (R-15) and the
  benchmark file (R-12) makes it a build-facing project change, so `CHANGELOG.md` and the
  README/`docs/advanced-usage.md` build-option tables are now in scope and required by
  `AGENTS.md` Spec Sync. Roadmap phase ticks remain out of scope.

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
- [D-6] The workload model is `data/models/rfdetr-nano-1101.onnx` (detection, input
  `1x3x640x640`, outputs `dets [1,300,4]` and `labels [1,300,91]`). D-3's
  `output_detection/inference_model.onnx` does not exist, and `*.onnx` is gitignored, so the
  premise that a model is checked in was wrong. Resolution auto-detects from the model, so
  `--resolution` is not passed. This supersedes D-3's model path; the image and labels are
  unchanged and are genuinely checked in.
- [D-7] `kernel.perf_event_paranoid` was `4`, which denies unprivileged counter access, and
  `sudo` is not available non-interactively to the agent. The user lowered the sysctl
  explicitly for this session rather than the profile substituting simulated counters. Both
  the denied probe and the granted setting are recorded; the sysctl is the user's to restore.
- [D-8] Valgrind runs use a 60-frame video, not the 600-frame fixture. Massif with
  `--stacks=yes --detailed-freq=1` over 600 frames costs hours for peak-heap information that
  model load plus a few steady-state frames already establishes. Recorded as a deviation with
  its reason; the 600-frame fixture remains the timing and `perf` workload.
- [D-9] Steady-state stage cost is measured with the repository's existing optional Google
  Benchmark target (`-DBENCHMARKS=ON`, `find_dependency_unified(GoogleBenchmark)`,
  `Deps::GoogleBenchmark`), extended with a CPU-path file, rather than a new harness or a
  bare `find_package(benchmark CONFIG REQUIRED)`. The dependency is already declared in
  `cmake/deps/packages/GoogleBenchmark.cmake` and pinned in `versions.env`, so this adds no
  dependency and no version change.
- [D-10] The canonical profiling build is `-DCMAKE_BUILD_TYPE=Release -DPROFILING=ON`:
  production optimization level, frame pointers preserved, symbols present. The earlier
  `RelWithDebInfo` build with hand-passed `CMAKE_CXX_FLAGS_RELWITHDEBINFO` is kept and
  reported as the second arm of a controlled comparison rather than discarded, since it
  differs from the canonical build in exactly one respect that matters (`-O2` versus `-O3`).
  Reporting both is what makes the optimization level a measured variable instead of an
  assumption.
- [D-11] `PROFILING` is deliberately orthogonal to `CMAKE_BUILD_TYPE` and defaults to `OFF`,
  so no default or CI build changes behaviour. It is not folded into the sanitizer options,
  which are mutually exclusive of each other for reasons that do not apply here.
- [R-16 note] The competing implementation for the decode comparison lives in the benchmark
  file, not in `src/`. Nothing shipped is replaced by it; it exists to give the shipped
  decode a baseline to be relative to.

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

- [ ] R-1 through R-16 map to executed validation evidence.
- [ ] `results.md` contains the required measured metrics and raw artifact locations.
- [ ] `profiling-guide.md` contains no fabricated result or unsupported optimization claim.
- [ ] Changes outside this feature directory are limited to `CMakeLists.txt`, a new file
      under `tests/benchmark/`, `CHANGELOG.md`, `README.md`, `docs/advanced-usage.md`, and
      `specs/tech-stack.md` — each traceable to R-12 or R-15.
- [ ] `./scripts/scoreboard.sh` passes with the new option and benchmark file present.
