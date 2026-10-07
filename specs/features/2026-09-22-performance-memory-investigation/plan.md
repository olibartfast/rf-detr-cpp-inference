# Plan — Performance and Memory Investigation

## Group 1 — Establish the contract

- [T-1] Read the project constitution, profiling commands, GPU contract, and workflow rules.
- [T-2] Define scope, decisions, traceability, and validation before drafting the guide.
- Exit: `requirements.md`, `plan.md`, and `validation.md` exist on the feature branch and
  every in-scope requirement has a validation row.

## Group 2 — Measurement protocol

- [T-3] Delegate a read-only review of workload validity, counter groupings, sampling,
  memory tools, warm-up, repetitions, and artifact capture.
- [T-4] Fix one protocol before execution so results are not selected after observation.
- Exit: reviewer accepts or rejects the protocol without modifying the worktree.

## Group 3 — Execute profiling

- [T-5] Build `RelWithDebInfo`, generate the fixed video under `/tmp`, verify the workload,
  and capture repeated native timing and peak RSS.
- [T-6] Capture `perf stat` counter groups and a `perf record` call graph against the fixed
  workload with the required privileges.
- [T-7] Capture a Massif heap profile and peak allocation owners using a plain Debug build.
- [T-8] Write `results.md` with exact commands, summarized metrics, raw artifact paths,
  bottleneck classification, competing explanations, and optimization priorities.
- Exit: all metrics required by R-8 through R-10 are present and traceable to raw output.

## Group 4 — Planner review and evidence

- [T-9] Check the worker diff against R-1 through R-11 and the writable-path boundary.
- [T-10] Validate links, shell syntax, formulas, estimates, and source citations; run
  `git diff --check`.
- [T-11] Record all pass, fail, and unrun evidence in `validation.md` and the attempt outcome
  in `attempt-log.md`.
- Exit: the feature stays incomplete unless measured metrics are present.

## Group 5 — Profiling build path and steady-state benchmarks

- [T-12] Add the `PROFILING` CMake option (default `OFF`) applying `-fno-omit-frame-pointer`
  plus debug information to every target, so call graphs survive the library boundary, and
  document it where `AGENTS.md` Spec Sync requires build options to appear.
- [T-13] Add `tests/benchmark/bench_cpu_pipeline.cpp`, registered on the existing `benchmarks`
  target, covering full-frame CPU preprocessing, detection decode, and annotation rendering.
  Parameterize over problem size; give the decode stage a competing implementation to be
  measured against; defeat dead-code elimination at every observation point.
- [T-14] Run the benchmark target on the quiet machine and capture the raw JSON plus console
  output as checksummed artifacts.
- Exit: the scoreboard passes with both changes present, and every benchmark reports a real
  and CPU time with its iteration count.

## Group 6 — Controlled build-configuration comparison

- [T-15] Rebuild the canonical `Release` + `PROFILING=ON` tree and repeat the seven-run timing
  set, one variable changed against the `RelWithDebInfo` arm.
- [T-16] Report both arms with their dispersion, and state whether the difference exceeds the
  measured run-to-run noise. Do not report a difference inside the noise as a speedup.
- Exit: both arms recorded with their coefficients of variation.

## Delegation Boundary

The reviewer has no writable paths. The measurement workers write only run-owned files under
`/tmp/rfdetr-profile-results/` plus the gitignored `build-perf/` and `build-valg/` trees. The
build-and-benchmark worker additionally writes `CMakeLists.txt`,
`tests/benchmark/bench_cpu_pipeline.cpp`, and the documentation files R-15 requires. The
results worker writes only
`specs/features/2026-09-22-performance-memory-investigation/results.md`.

`requirements.md`, `plan.md`, `validation.md`, `protocol.md`, and `attempt-log.md` are
planner-owned acceptance inputs: no worker may edit the file that scores it. Path limits are
advisory in this harness, so the planner enforces them with `git status --short` and
`git diff --name-only` against the packet's writable list, and rejects an out-of-scope path
exactly as if a sandbox had blocked it. A user-owned modification to
`.claude/skills/release/SKILL.md` is present in the working tree and must survive untouched.

Measurement packets are serialized on purpose: two workers running concurrently would each
be measuring the other's load. Only one measurement packet is in flight at a time, and no
packet that builds or benchmarks overlaps one that times.

## Scoreboard

- Command: `./scripts/scoreboard.sh`
- Adjudicates: existing formatting, configuration, build, and test contract remains intact.
- Run per worker attempt: exactly once, as the final action, by any packet that changes an
  in-repo file. Packets that write only `/tmp` artifacts run it not at all — there is nothing
  in the repository for it to score, and a redundant Release rebuild would perturb the
  measurement the packet exists to take.
- Limitation: the scoreboard protects regression behavior but cannot replace R-7 through
  R-11's measured profiling evidence. It also does not compile the `benchmarks` target, which
  is opt-in through `-DBENCHMARKS=ON`; the benchmark packet must configure and build that
  target itself in addition to running the scoreboard.
