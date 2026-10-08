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
| 7 | P1a review fix (cppcheck `useStlAlgorithm`, 8 unpinned behaviours) | Resumed P1a agent with corrected packet | Edits complete at cap, uncommitted; driver judged and committed unchanged | 141.7k | 12 | 1 resume; driver commit |
| 8 | P3 review fix (`const` on mutating methods) | Resumed P3 agent | Committed `fccddbb`; SCOREBOARD PASS | 125.7k | 22 | — |
| 9 | P2b (keypoint split) | Driver worktree from merged branch | `02fbceb`; reported cppcheck exit 0 — **false**: inline `cppcheck-suppress` is inert without `--inline-suppr`. Root cause: the driver's packet mandated two members nothing reads | 127.4k | 47 | Driver reject |
| 10 | P2b fix (drop unread members) | Resumed P2b agent | `ad734b2`; driver-verified cppcheck 0; reviewer ACCEPT | 146.7k | 28 | — |
| 11 | P1b (main split) | Driver worktree from merged branch | `2e3026b`, first pass, within budget; reviewer ACCEPT | 76.6k | 6 | — |
| 12 | P5 (PROFILING + benchmarks) | Same | `fcd8993` committed at cap; driver-verified; reviewer ACCEPT | 80.5k | 13 | — |
| 13 | P4 (enforce + cleanups) | Same, ≤ 6-turn budget stated | `dc8dcd3`, first pass; corrected the driver's channel description against the code; reviewer ACCEPT | 73.4k | 18 | — |
| 14 | P4b (review notes) | Same | `1f0fa7b`, first pass; driver-verified | 69.4k | 12 | — |

Reviewer runs: 3 (P1a/P2a/P3: 82.8k; P1b/P2b: 60.1k; P4/P5: 54.1k). Verdicts: P2a, P1b,
P2b (after the driver reject), P4 and P5 ACCEPT on first review; P1a and P3 REJECT, then fixed.

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
- **The scoreboard is not the lint gate.** `scripts/scoreboard.sh` runs format, build and tests
  only. Two packets passed it with cppcheck failing. Every judge step now runs cppcheck and
  clang-tidy separately, and `AGENTS.md` says so.
- **Worker self-reports are claims, not evidence.** One "cppcheck exit 0" report was false.
  The driver re-runs every acceptance check before merging.
- **Packets that name members commit to them.** P2b's dead members came from the packet, not
  the worker. Name the helpers and leave struct contents to the worker, or name only what
  the caller reads.
- **Driver error, recorded:** the driver rebuilt `build-baseline` from merged code. The
  baseline now lives in a detached worktree at `cc0d4f9`, and its rebuild reproduced all 45
  hashes before anything else was compared against it.
- **First-pass success came once packets were small.** Packets of at most ~2 files, with the
  check command pasted in, mostly passed first time (P1b, P4, P4b). Packets that bundled
  design work with edits ran out of turns.

