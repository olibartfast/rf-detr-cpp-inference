# Validation — Performance and Memory Investigation

Validation was defined before `profiling-guide.md` was implemented.

The feature was reopened after review established that documentation had been mistaken for
the deliverable. It remains incomplete until the metrics rows below pass.

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
| V-9 / R-7 | Actual application workload | Run checked-in model/image/labels and generated video; record frame count and output | Pending | Pending |
| V-10 / R-8 | Repeated timing and counters | Fixed `perf stat` groups with repeated runs and raw logs | Pending | Pending |
| V-11 / R-9 | Hotspot call graph | `perf record` plus noninteractive report with symbols | Pending | Pending |
| V-12 / R-10 | Native and heap peaks | `/usr/bin/time -v`, Massif, and `ms_print` peak tree | Pending | Pending |
| V-13 / R-11 | Results record | Review `results.md` against raw artifacts and exact commands | Pending | Pending |
| V-14 / R-15 | `PROFILING` build option | Configure with and without `-DPROFILING=ON`; confirm default `OFF` changes nothing, and that `-fno-omit-frame-pointer` reaches a library target's compile line, not only `inference_app` | Pending | Pending |
| V-15 / R-12, R-13 | Steady-state benchmarks | Build `-DBENCHMARKS=ON`; run `benchmarks`; every added case reports real time, CPU time and iterations | Pending | Pending |
| V-16 / R-16 | Benchmark rigor | Each added case is size-parameterized; the decode case has a competing implementation; every observation point uses `DoNotOptimize`/`ClobberMemory` | Pending | Pending |
| V-17 / R-14 | Deferral claim tested | `results.md` states whether measured decode cost supports or contradicts the roadmap's "not a bottleneck" reason | Pending | Pending |
| V-18 / R-15 scope | Option documented | `PROFILING` appears in `README.md`, `docs/advanced-usage.md`, and `specs/tech-stack.md`; `CHANGELOG.md` has the entry | Pending | Pending |
| V-19 / D-10 | Controlled comparison | Both build arms reported with dispersion; no difference inside the noise reported as a speedup | Pending | Pending |
| V-21 / regression | Pre-existing benchmark build break | Build `-DBENCHMARKS=ON -DWERROR=ON` without DALI | FAIL (pre-existing), fix dispatched | `bench_gpu_pipeline.cpp:44` defines `encode_jpeg` outside the `#ifdef USE_DALI` guard holding its only call site (`:95`), so `-Wunused-function` fires under `-Werror`. Broken since that file landed; CI misses it because `BENCHMARKS` defaults `OFF` and the GPU job builds the library only |
| V-22 / R-15 | Flag reaches a library TU | `ninja -C build-bench -t commands rfdetr_inference_lib` with `-DPROFILING=ON` | PASS | Compile line carries `-fno-omit-frame-pointer -g -O3`, confirming `add_compile_options` reaches `rfdetr_inference_lib` and not merely `inference_app` |
| V-20 / scope | Writable-path boundary | `git status --short` and `git diff --name-only` show only the paths R-12/R-15 permit, and `.claude/skills/release/SKILL.md` keeps the user's modification | Pending | Pending |

## Baseline Environment

Record before implementation:

- Start revision: `e9c081566ba98a91b35543f1250b58cba4958a03`.
- CPU and kernel: ~~AMD Ryzen 7 5700U, 16 logical CPUs~~, Linux `7.0.0-31-generic`.
  **This CPU claim was false and is retracted.** `lscpu` on the execution pass reports an
  11th Gen Intel Core i5-11400H, 6 cores / 12 threads, L3 12 MiB. No evidence for the Ryzen
  figure exists in the repository or on the host; it was asserted, not measured. Correct
  values are in `/tmp/rfdetr-profile-results/env/environment.md`, captured from `lscpu`.
- Available: `/usr/bin/perf`, `/usr/bin/time`, and generic `/usr/bin/clang-format`.
  Unavailable from `PATH`: Valgrind, Heaptrack, and `clang-format-18`.
- `perf_event_paranoid`: `4`; `perf stat -e cycles,instructions -- true` failed with
  “No supported events found” because performance-monitoring access is restricted.
- Representative workload: unavailable unless supplied; runtime profiling remains `UNRUN`.

### Re-verified 2026-09-22 (execution pass)

The inventory above was recorded when the feature was documentation-only. It is stale in three
respects; the corrected values, not the stale ones, govern the execution pass:

- Now available: `/usr/bin/valgrind`, `/usr/bin/ms_print`, `/usr/bin/clang-format-18`,
  `/usr/bin/ffmpeg`, `/usr/bin/ffprobe`. Massif and Memcheck are therefore executable rather
  than `UNRUN`. Still unavailable: `heaptrack` — the allocation-profiler procedure under R-4 is
  served by Massif's detailed peak tree plus Memcheck, and heaptrack stays `UNRUN`.
- `perf_event_paranoid` was still `4` and `sudo` is unavailable non-interactively to the agent,
  so the unprivileged probe failed again and is recorded. The user lowered the sysctl
  explicitly for this session (D-7); the value in force during capture is recorded in
  `results.md` with the probe that failed before it.
- A representative workload now exists: `data/models/rfdetr-nano-1101.onnx` with `data/dog.jpg`
  and `data/coco-labels-91.txt` (D-6). `output_detection/inference_model.onnx` named in D-3 does
  not exist and `*.onnx` is gitignored, so no model is checked in — the earlier premise was
  wrong, and runtime profiling is no longer blocked on it.
- Host: 11th Gen Intel Core i5-11400H, 6 cores / 12 threads, 38 GiB RAM, governor
  `powersave`, kernel `7.0.0-31-generic`, g++ 13.3.0, cmake 3.28.3, valgrind 3.22.0. Captured
  from `lscpu`/`free`/`--version` into `environment.md`, superseding the retracted Ryzen claim
  above.
- Measurement host was **not idle** at first capture: load average 3.72, with a k3d cluster
  (`k3d-neuriplo-server-0`, k3s plus its in-container containerd at roughly 60% of a core
  combined), AnyDesk, and the agent session itself competing for CPU. Timing taken under that
  load is contaminated. The mitigation is recorded with the measurements: the cluster and
  AnyDesk are stopped for the measured runs, and every measured run is additionally pinned to
  isolated cores with `taskset`. Any number captured before the host was quiet is labelled as
  such and is not the figure of record.

## Evidence Log

| Check | Result | Date | Notes |
|---|---|---|---|
| Contract written before guide | PASS | 2026-09-22 | Requirements, plan, and validation created first |
| Source review | PASS | 2026-09-22 | Five hypotheses tied to exact code, competing causes, decisive measurements, expected metric movement, and risks |
| Fixed measurement protocol | PASS | 2026-09-22 | Initial loose plan rejected; reviewer corrections adopted in `protocol.md` before execution |
| Worker scoreboard | PASS | 2026-09-22 | Run exactly once; configure/build passed and 10/10 tests passed; Valgrind target disabled because Valgrind unavailable |
| Planner content review | PASS | 2026-09-22 | Commands, formulas, source hypotheses, qualifications, estimates, and official links reviewed |
| Scope diff | PASS | 2026-09-22 | Only `specs/features/2026-09-22-performance-memory-investigation/` changed |
| Actual workload metrics | Pending | — | Feature reopened; documentation is insufficient |

## Definition of Done

- [ ] Every matrix row has an executed result and concrete evidence.
- [ ] Required application metrics are present in `results.md`.
- [ ] Failures and unavailable checks remain visible as `FAIL` or `UNRUN`.
- [ ] Guide and spec describe actual delivered scope.
- [x] No roadmap phase or release is marked complete by this investigation.
