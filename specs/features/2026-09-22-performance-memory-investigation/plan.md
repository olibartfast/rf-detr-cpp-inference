# Plan — Performance and Memory Investigation

## Group 1 — Establish the contract

- [T-1] Read the project constitution, profiling commands, GPU contract, and workflow rules.
- [T-2] Define scope, decisions, traceability, and validation before drafting the guide.
- Exit: `requirements.md`, `plan.md`, and `validation.md` exist on the feature branch and
  every in-scope requirement has a validation row.

## Group 2 — Source investigation

- [T-3] Delegate a read-only review of inference output ownership, mask processing, video
  slot lifetimes, media drawing, and existing benchmarks.
- [T-4] Require each hypothesis to include exact source evidence, a competing explanation,
  and a decisive measurement.
- Exit: reviewer returns a bounded findings report without modifying the worktree.

## Group 3 — Maintainer guide

- [T-5] Delegate `profiling-guide.md` as the worker's only writable path.
- [T-6] Explain `perf`, memory profilers, controlled workload design, source hypotheses,
  optimization experiments, and correctness constraints.
- Exit: worker runs `./scripts/scoreboard.sh` exactly once as its final action and reports
  the result without repair.

## Group 4 — Planner review and evidence

- [T-7] Check the worker diff against R-1 through R-6 and the writable-path boundary.
- [T-8] Validate links, shell syntax, formulas, estimates, and source citations; run
  `git diff --check`.
- [T-9] Record all pass, fail, and unrun evidence in `validation.md` and the attempt outcome
  in `attempt-log.md`.
- Exit: the feature packet accurately reflects the delivered guide and validation state.

## Delegation Boundary

The reviewer has no writable paths. The documentation worker may create only
`specs/features/2026-09-22-performance-memory-investigation/profiling-guide.md`.
The specification and validation files are planner-owned acceptance inputs. Codex workspace
write access is coarse, so the planner enforces this boundary by inspecting the final diff.

## Scoreboard

- Command: `./scripts/scoreboard.sh`
- Adjudicates: existing formatting, configuration, build, and test contract remains intact.
- Run per worker attempt: exactly once, as the final action.
- Limitation: a passing scoreboard does not establish guide accuracy or a performance
  improvement; R-1 through R-6 also require the manual evidence checks in `validation.md`.
