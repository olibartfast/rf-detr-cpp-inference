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

## Delegation Boundary

The reviewer has no writable paths. The profiling worker may write only
`specs/features/2026-09-22-performance-memory-investigation/results.md` and run-owned files
under `/tmp/rfdetr-profile-results/`. The specification and validation files are
planner-owned acceptance inputs. Codex workspace write access is coarse, so the planner
enforces this boundary by inspecting the final diff.

## Scoreboard

- Command: `./scripts/scoreboard.sh`
- Adjudicates: existing formatting, configuration, build, and test contract remains intact.
- Run per worker attempt: exactly once, as the final action.
- Limitation: the scoreboard protects regression behavior but cannot replace R-7 through
  R-11's measured profiling evidence.
