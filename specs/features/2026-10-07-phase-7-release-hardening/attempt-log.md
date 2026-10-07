# Attempt log — Phase 7

One row per delegated attempt. The roles are the Claude Code mapping in
`specs/delegation-workflow.md`, and the implementer has `maxTurns: 12`. Token counts are the
subagent totals the harness reported.

| # | Packet | Worker setup | Outcome | Tokens | Tool calls | Interventions |
|---|--------|--------------|---------|--------|------------|---------------|
| 1 | P1 (CLI + main split, whole) | `isolation: worktree` | Turn cap hit after reading only; no edits | 88.9k | 20 | Discarded |
| 2 | P2 (orchestrator + keypoint, whole) | `isolation: worktree` | Worktree cut from `master` (`74d008f`), not the phase branch; stopped by driver | — | — | Stopped |
| 3 | P3 (FFmpeg) | `isolation: worktree` | Turn cap hit after reading only; no edits | 97.4k | 20 | Discarded |
| 4 | P1a (parser unit only) | Driver-made worktree, pre-built, baseline edge cases pasted into packet | Files written at turn cap; finished after one resume; SCOREBOARD PASS | 117.4k | 22 | 1 resume |
| 5 | P2a (tidy, keypoint split deferred) | Same | Committed `8d401b1` on second window; two `-Werror` follow-ups left uncommitted, committed by driver as `93843da`; SCOREBOARD PASS (driver-run) | 134.4k | 58 | 2 resumes; driver commit |
| 6 | P3 (FFmpeg, re-dispatch) | Same | Edits complete at second cap, uncommitted; driver judged (format, build, tests, golden, tidy) and committed `6871b81` unchanged; SCOREBOARD PASS (driver-run) | 106.6k | 36 | 1 resume; driver commit |

## Lessons carried into the next packets

- **The worktree must be cut from the phase branch.** `isolation: worktree` started from the
  default branch here. The driver now creates worktrees under `/tmp/rfdetr-phase7/wt-*` from
  `feature/phase-7-release-hardening` and builds them before dispatch.
- **12 turns covers about 2 files of edits plus 1 batched check.** Packets must name the
  edits precisely enough that no reading turn is needed beyond the files being edited. Facts
  the worker would otherwise gather, such as the baseline edge cases, go into the packet.
- **The harness, not the worker, was wrong once.** `golden_cli.sh` did not normalise `argv[0]`,
  so the usage text differed by binary path. P1a found and reported it; fixed in `e7e5c9b`.
  This is a planner defect and is recorded as one.
- No driver edits to `src/` or `tests/`. The driver commits were of worker-written content,
  unchanged, after the driver had run the judge checks.
