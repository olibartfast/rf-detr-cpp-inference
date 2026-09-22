# Validation — Performance and Memory Investigation

Validation was defined before `profiling-guide.md` was implemented.

## Requirements-to-Evidence Matrix

| ID | Requirement or boundary | Exact check | Result | Evidence |
|---|---|---|---|---|
| V-1 / R-1 | Reproducible `perf` workflow | Review commands against application CLI; parse every Bash block with `bash -n` | PASS | Bash blocks parse; CLI order and flags match `src/main.cpp` |
| V-2 / R-2 | Correct metric interpretation | Recompute formulas; review task-clock, IPC, miss rates, multiplexing, Self/Children, and off-CPU limits | PASS | Formulas and interpretation limits reviewed against perf manuals |
| V-3 / R-3 | Memory concepts remain distinct | Review RSS/live heap/churn/retention/leak/mappings/VRAM descriptions for conflation | PASS | Guide separates all seven concepts and appropriate tools |
| V-4 / R-4 | Profiling procedures respect build constraints | Check commands against `AGENTS.md`, `CMakeLists.txt`, and Valgrind/sanitizer exclusions | PASS | RelWithDebInfo for perf; plain Debug for Massif; no sanitizer flags combined |
| V-5 / R-5 | Hypotheses are source-backed | Resolve every relative source link; inspect cited code; require competing explanation and controlled experiment | PASS | All relative links resolve; five hypotheses contain measurement and risk |
| V-6 / R-6 | Evidence honesty | Search for unsupported “measured”, “proved”, “improved”, and unlabeled estimates; compare with tool inventory | PASS | Runtime profiling explicitly UNRUN; 1.55 GiB example labeled hypothetical |
| V-7 / scope | No implementation or unrelated edits | `git status --short` and `git diff --check` | PASS | Only this feature directory is untracked; diff check clean |
| V-8 / regression | Repository gate | `./scripts/scoreboard.sh` once by documentation worker | PASS | `SCOREBOARD: PASS`; configure/build succeeded and 10/10 tests passed |

## Baseline Environment

Record before implementation:

- Start revision: `e9c081566ba98a91b35543f1250b58cba4958a03`.
- CPU and kernel: AMD Ryzen 7 5700U, 16 logical CPUs, Linux `7.0.0-31-generic`.
- Available: `/usr/bin/perf`, `/usr/bin/time`, and generic `/usr/bin/clang-format`.
  Unavailable from `PATH`: Valgrind, Heaptrack, and `clang-format-18`.
- `perf_event_paranoid`: `4`; `perf stat -e cycles,instructions -- true` failed with
  “No supported events found” because performance-monitoring access is restricted.
- Representative workload: unavailable unless supplied; runtime profiling remains `UNRUN`.

## Evidence Log

| Check | Result | Date | Notes |
|---|---|---|---|
| Contract written before guide | PASS | 2026-09-22 | Requirements, plan, and validation created first |
| Source review | PASS | 2026-09-22 | Five hypotheses tied to exact code, competing causes, decisive measurements, expected metric movement, and risks |
| Worker scoreboard | PASS | 2026-09-22 | Run exactly once; configure/build passed and 10/10 tests passed; Valgrind target disabled because Valgrind unavailable |
| Planner content review | PASS | 2026-09-22 | Commands, formulas, source hypotheses, qualifications, estimates, and official links reviewed |
| Scope diff | PASS | 2026-09-22 | Only `specs/features/2026-09-22-performance-memory-investigation/` changed |

## Definition of Done

- [x] Every matrix row has an executed result and concrete evidence.
- [x] Failures and unavailable checks remain visible as `FAIL` or `UNRUN`.
- [x] Guide and spec describe actual delivered scope.
- [x] No roadmap phase or release is marked complete by this investigation.
