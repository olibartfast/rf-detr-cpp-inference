# Attempt Log — Performance and Memory Investigation

| Attempt | Role/configuration | Start SHA | Scoreboard | Outcome | Interventions | Metrics | Notes |
|---|---|---|---|---|---|---|---|
| 1 | Read-only reviewer; inherited session model | `e9c0815` | Not applicable | PASS: five source-backed hypotheses and benchmark gaps reported | 0 | Native per-agent telemetry unavailable | No writes or workloads |
| 2 | Documentation worker; inherited session model | `e9c0815` | `./scripts/scoreboard.sh` | PASS: configure/build and 10/10 tests | 0 | Native per-agent telemetry unavailable | Created only `profiling-guide.md`; Valgrind target disabled because tool unavailable |
| 3 | Read-only protocol reviewer; inherited session model | `585e34b` | Not applicable | REJECT: plan lacked fixed executable protocol | 0 | Native per-agent telemetry unavailable | Corrections adopted before measurement |
| 4 | Implementer packet 1 (env record, `build-perf`, `build-valg`, fixtures, timing); inherited session model | `46fedbc` | Not run (packet writes no in-repo file) | PARTIAL: items 1-2 of 7 done (environment record, both builds); stopped at the 12-turn step ceiling before fixtures and timing | 0 | 68.6k subagent tokens, 14 tool calls, 151 s | Wrote only `/tmp/rfdetr-profile-results/` and the two gitignored build trees; `git status --short` confirmed no tracked file touched. Honestly recorded that the host was **not** idle (load 3.72, k3d cluster + AnyDesk) instead of reporting clean numbers |
| 5 | Implementer packet 2 (`PROFILING` option, CPU benchmarks, doc propagation); inherited session model | `46fedbc` | Not reported (agent stopped before its single scoreboard run) | PARTIAL + REJECT: `CMakeLists.txt` accepted; `bench_cpu_pipeline.cpp` rejected on four items; documentation propagation never started; stopped at the 12-turn ceiling | 1 (planner rejection) | 115.6k subagent tokens, 24 tool calls, 388 s | Rejections: (1) the shipped-vs-alternative top-k arms used different seeds and only one arm had a NaN, so the A/B measured different inputs; (2) `BM_BuildForegroundScores` used `background_slot = num_classes - 1` while the shipped default is `0`, exercising a branch production never takes; (3) `BM_DrawDetections` paused/resumed timing every iteration, overhead comparable to the measured work at `Arg(1)`, for a restore that is unnecessary because drawing cost is content-independent; (4) `<span>` used but not included. Planner did **not** repair any of them |
| 6 | Implementer packet 2b (four rejections + documentation propagation); inherited session model | `46fedbc` | Not reported (stopped before its single scoreboard run) | PARTIAL: all four rejections correctly fixed and verified by the planner; `README.md` and `docs/advanced-usage.md` done; `specs/tech-stack.md`, `AGENTS.md`, `CHANGELOG.md` not started; stopped at the 12-turn ceiling | 0 | 75.4k subagent tokens, 14 tool calls, 114 s | Re-dispatched as a corrected packet rather than fixed in place, so implementation cost stayed with the worker role. Planner verified each rejection on disk rather than trusting a report — no report was delivered |
| 7 | Planner, completing the three remaining documentation files | `46fedbc` | Deferred to packet 8 | PASS | — | Not applicable | `specs/tech-stack.md`, `AGENTS.md` and `CHANGELOG.md` are planner-writable under `specs/delegation-workflow.md`, so this is not self-repair of worker code: the worker's `CMakeLists.txt` and benchmark file were already accepted. Delegating planner-owned prose after three ceiling stops would have spent turns to no purpose. Also corrected every stale `CMakeLists.txt` line reference in the `tech-stack.md` option table |
| 8 | Implementer packet 3 (pre-existing `-Werror` break in `bench_gpu_pipeline.cpp`) | `46fedbc` | Pending | Pending | — | Pending | Found by the planner's own verification build, not by any worker: `encode_jpeg` is defined outside the `#ifdef USE_DALI` guard that holds its only call site, so `-DBENCHMARKS=ON -DWERROR=ON` fails on every non-DALI build. Predates this feature |

The harness does not expose per-agent active-context or metered-cost telemetry here; those
fields remain unavailable rather than estimated. Subagent token totals, tool-call counts and
wall-clock **are** reported by the harness on completion and are recorded above as given.
Same-model delegation is used for isolation and reviewability, not presented as a cost
reduction — every agent here inherits the session model, so no tier saving is claimed.

## Observation on the step ceiling

Two consecutive packets stopped at the ~12-turn worker ceiling before finishing, both after
spending most of their turns on compiler and configure round-trips. The ceiling is doing its
job — neither worker thrashed, and both left the tree in a reviewable state — but it means a
packet whose verification includes a full CMake configure plus build plus a scoreboard run has
roughly half its budget consumed by feedback loops. The mitigation applied from packet 2b
onward is to state the batching requirement in the packet itself rather than to raise the
ceiling: fewer, larger tool calls, and no exploratory reading outside the read-only list.
Splitting a packet further is the other lever, and is preferable to a worker that reports
nothing.
