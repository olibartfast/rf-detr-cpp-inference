# Fixed Profiling Protocol

This protocol was fixed before measurement. Raw artifacts live under
`/tmp/rfdetr-profile-results/`; the committed summary lives in `results.md`.

## Workload and environment

- Model: `output_detection/inference_model.onnx`
- Image: `data/dog.jpg`
- Labels: `data/coco-labels-91.txt`
- Video: `/tmp/rfdetr-dog-600f.mkv`, generated from the image as 600 lossless FFV1
  frames at 30 FPS, 768x576, intra-only, metadata stripped.
- Backend: default CPU ONNX Runtime; do not pass `--segmentation`.
- Record revision, kernel, CPU, CMake/perf/Valgrind versions, input SHA-256 hashes,
  video properties, compiler flags, and `perf_event_paranoid` before profiling.

Generate the video:

```bash
ffmpeg -hide_banner -loglevel error -y \
  -loop 1 -framerate 30 -i data/dog.jpg \
  -frames:v 600 -an -c:v ffv1 -level 3 -g 1 -pix_fmt bgr0 \
  -map_metadata -1 -fflags +bitexact -flags:v +bitexact \
  /tmp/rfdetr-dog-600f.mkv
```

Acceptance: `ffprobe` reports FFV1, 768x576, 30/1, and 600 decoded frames.

## Setting Up Applications for Profiling

Optimized build with symbols and frame pointers:

```bash
cmake -S . -B build-perf -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_CXX_FLAGS_RELWITHDEBINFO="-O2 -g -DNDEBUG -fno-omit-frame-pointer"
cmake --build build-perf --parallel
```

Plain Debug build for Valgrind:

```bash
cmake -S . -B build-valg -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug -DSANITIZERS=OFF \
  -DSTRICT_UBSAN=OFF -DTHREAD_SANITIZER=OFF -DBENCHMARKS=OFF
cmake --build build-valg --parallel
```

Run one excluded warm-up each for image and video. The image measures startup-inclusive
latency. The full 600-frame process measures pipeline throughput with startup amortized;
it is not a true internal steady-state window.

## Perf the Program and Profiling Your Code

Run `/usr/bin/time -v` seven times for both image and video. Preserve every stdout,
stderr, and time record. All runs must exit successfully; every video run must report
600 processed frames. Report minimum, maximum, median, arithmetic mean, sample standard
deviation, and coefficient of variation for elapsed time and maximum RSS. A CV above 5%
is marked unstable and triggers a second seven-run set without discarding the first.

Record the failed unprivileged probe caused by `perf_event_paranoid=4`. Do not change the
sysctl. With explicit privileged execution, run these groups separately seven times on
the video workload:

```text
task-clock,context-switches,cpu-migrations,page-faults,minor-faults,major-faults
{cycles,instructions}
{branches,branch-misses}
{cache-references,cache-misses}
```

No counter may report `<not counted>` or `<not supported>`. Record enabled/running ratios;
below 90% is multiplexed and requires splitting that group and rerunning it.

Formulas:

```text
FPS = 600 / elapsed_seconds
average utilized cores = task_clock_seconds / elapsed_seconds
IPC = instructions / cycles
branch miss percent = 100 * branch_misses / branches
generic cache miss percent = 100 * cache_misses / cache_references
faults/context-switches/migrations per frame = event / 600
CV percent = 100 * sample_standard_deviation / arithmetic_mean
```

Capture `perf record -e cycles:u -F 199 --call-graph dwarf` for the video, followed by
noninteractive Self and Children reports. If `cycles:u` is unavailable, record the failure
and retry with `cpu-clock:u`. Require at least 1,000 samples, report lost samples and
unresolved-symbol share, and list the top ten application and library symbols. More than
10% unresolved symbols requires fixing attribution and repeating the capture.

## Identifying Memory Bottlenecks

Native peak RSS comes from the seven `/usr/bin/time -v` video runs. Report median and
maximum in KiB and MiB.

Run Massif directly on the Debug application, not the repository convenience target:

```bash
valgrind --tool=massif --stacks=yes --time-unit=B \
  --detailed-freq=1 --max-snapshots=200 \
  --massif-out-file=/tmp/rfdetr-profile-results/memory/massif.out \
  ./build-valg/inference_app output_detection/inference_model.onnx \
  /tmp/rfdetr-dog-600f.mkv data/coco-labels-91.txt \
  --output /tmp/rfdetr-profile-results/outputs/massif-video.mp4
```

`ms_print` must parse the output. Record peak `mem_heap_B`, `mem_heap_extra_B`, and
`mem_stacks_B`, and allocation owners covering at least 90% of the detailed peak.

Run Memcheck directly on the image workload with full leak kinds, origins, definite and
indirect errors enabled, and `--error-exitcode=99`. Record definitely lost, indirectly
lost, possibly lost, still reachable, suppressed bytes, and error count. Findings do not
invalidate the profile. Explain that live heap and RSS measure different things; VRAM is
out of scope for the CPU ONNX Runtime workload.

## Identifying Slow Computation and Analysis

`results.md` must report all environment details, exact commands, raw artifact paths,
distributions, formulas, sampled Self/Children symbols, RSS, Massif peak owners, Memcheck
classifications, and a ranked bottleneck table. Each bottleneck needs measured evidence,
a competing explanation, the next controlled experiment, and correctness risk.

The repeated-frame video measures reproducible pipeline mechanics. It has invariant
detections, unusually predictable control flow and allocations, high temporal codec
compressibility, and no scene transitions. Do not generalize its rankings to diverse real
video without another measured workload.

The feature remains incomplete until every required metric has a concrete value traceable
to checksummed raw artifacts. `UNRUN` is honest but does not satisfy the completion gate.

---

# Protocol Addendum — Execution Pass (2026-09-22)

The protocol above was fixed before measurement and is not rewritten. This addendum records
every deviation from it, with the reason, and fixes the additional protocol the expanded scope
needs. Where the two disagree, the addendum governs, and the disagreement is visible rather
than edited away.

## Recorded deviations

| # | Protocol said | Actually done | Reason |
|---|---|---|---|
| 1 | Model `output_detection/inference_model.onnx` | `data/models/rfdetr-nano-1101.onnx` (detection, `1x3x640x640`, `dets [1,300,4]`, `labels [1,300,91]`) | The named path does not exist and `*.onnx` is gitignored, so no model is checked in. The premise behind D-3 was wrong. Image and labels are unchanged and are checked in. |
| 2 | `--resolution` unspecified | Not passed at all | `src/main.cpp` auto-detects resolution from the model when the flag is omitted; passing it would risk contradicting the model. |
| 3 | Massif on the 600-frame video | Massif on a 60-frame video (`/tmp/rfdetr-dog-60f.mkv`) | `--stacks=yes --detailed-freq=1` over 600 frames costs hours for peak-heap information that model load plus a few steady-state frames already establishes. The 600-frame fixture remains the timing and `perf` workload. |
| 4 | Valgrind and heaptrack unavailable, Massif/Memcheck `UNRUN` | Valgrind 3.22.0 and `ms_print` are present and were run | The tool inventory in `validation.md` was stale. `heaptrack` is genuinely still absent and stays `UNRUN`; Massif's detailed peak tree plus Memcheck serve R-4's allocation-profiler intent. |
| 5 | `perf_event_paranoid=4`, counters denied | Sysctl lowered to `1` by the user for this session (D-7) | The agent has no non-interactive `sudo`. The denied probe at `4` and the value in force during capture are both recorded. Restoring it is the user's. |
| 6 | Machine assumed usable as-is | Host was **not** idle at first capture: load average 3.72, a k3d cluster (`k3d-neuriplo-server-0`: k3s plus in-container containerd, roughly 60% of a core combined), AnyDesk, and the agent session | Background load inflates elapsed time and pollutes every counter. Mitigation is below; numbers captured before the host was quiet are labelled and are not figures of record. |
| 7 | One build configuration | Two, compared under control (D-10) | `-fno-omit-frame-pointer` was first applied through `CMAKE_CXX_FLAGS_RELWITHDEBINFO`, which reaches only the application's own translation units. The `PROFILING` option (R-15) applies it to every target, including `rfdetr_inference_lib`, without which call graphs terminate at the library boundary. |
| 8 | Scoreboard once per worker attempt | Only packets that change an in-repo file run it | A packet writing solely `/tmp` artifacts gives the scoreboard nothing to score, and its Release rebuild would perturb the measurement the packet exists to take. |

## Quiet-host and pinning rules

These apply to every measured run — timing, `perf`, Massif, and benchmarks — and must be
identical across arms, or the arms are not comparable:

- The k3d cluster is stopped (`k3d cluster stop neuriplo`) and AnyDesk is stopped before
  capture. Record `uptime` load average immediately before and after each set.
- Every measured process is pinned with `taskset -c 0-5` — CPUs 0-5 are six distinct physical
  cores on this host (`thread_siblings_list` pairs them with 6-11), so the mask is one
  thread per core, chosen from `/sys/devices/system/cpu/cpu*/topology/thread_siblings_list` so
  no two pinned CPUs are hyperthread siblings. Record the exact mask and the sibling map.
- Pinning reduces available parallelism, so absolute throughput is lower than an unpinned run
  on the same host. That is accepted deliberately: a consistent, stated core budget makes the
  arms comparable, which absolute peak throughput would not. Never compare a pinned number
  with an unpinned one.
- The agent session itself consumes CPU. It stays idle during capture, and the run's own
  `%CPU` from `/usr/bin/time -v` is reported so contention is visible.

## Steady-state microbenchmarks

Process-level timing cannot separate model load, decode, and rendering; Google Benchmark
measures the stages in isolation (R-12, R-13, R-16).

- Target: the repository's existing opt-in `benchmarks` target
  (`-DBENCHMARKS=ON`, `find_dependency_unified(GoogleBenchmark)`, `Deps::GoogleBenchmark`).
  No new harness, no bare `find_package(benchmark CONFIG REQUIRED)`, no new dependency.
- Build: `-DCMAKE_BUILD_TYPE=Release -DBENCHMARKS=ON -DWERROR=ON`, pinned with `taskset`,
  `--benchmark_repetitions=5 --benchmark_report_aggregates_only=false`, JSON output kept as a
  raw artifact.
- Every case is parameterized over problem size through `state.range`, and every observation
  point uses `benchmark::DoNotOptimize` / `benchmark::ClobberMemory`. An unparameterized
  single number is not a performance result.
- `BM_SelectTopkShipped` versus `BM_SelectTopkBoundedHeap` is the one required A/B: the shipped
  `select_topk_multiclass` allocates and `std::iota`s an index vector the full size of the
  score grid (27,300 entries at 300x91) and `std::partial_sort`s it, so the alternative avoids
  that vector while reproducing the ordering rule exactly — descending score, NaN ranked ahead
  of every finite score, ties by ascending flattened index. The alternative asserts equality
  with the shipped result, including on a NaN input, before it is timed. A faster function that
  returns something different is not a comparison, and the competing implementation lives in
  the benchmark file, never in `src/`.
- Report real time, CPU time, iterations and the aggregate dispersion Google Benchmark emits.
  A difference inside that dispersion is not a speedup and must not be reported as one.

## What this addendum does not change

The completion gate is unchanged: the feature stays incomplete until every required metric has
a concrete value traceable to a checksummed raw artifact. `UNRUN` remains honest and remains
insufficient. Nothing here authorizes an optimization to `src/`; the measured code is the
shipped code.

## Second addendum — re-scoping forced by measured cost (2026-09-22)

A calibration probe (30-frame video, pinned, `build-perf`) measured **104.71 s elapsed, user
99.42 s, sys 0.93 s, max RSS 837,068 KB** — about 3.5 s per frame. Three consequences, each a
recorded deviation:

- **[D-12] The 600-frame fixture is infeasible here.** One run costs roughly 35 minutes, so the
  protocol's seven repetitions across two build arms would cost eight hours or more. The
  repeated timing sets therefore use a 30-frame video (n=7 per arm) plus a 60-frame video
  (n=5, canonical arm only). Two lengths are strictly better than one long run for the purpose:
  the slope of elapsed time against frame count separates per-frame cost from fixed startup
  cost, which a single 600-frame run can only assume. The 600-frame fixture is recorded as
  measured-infeasible, not skipped silently, and `600 / elapsed` as an FPS formula no longer
  applies — FPS is computed from the actual frame count of each fixture.
- **[D-13] Valgrind workloads shrink again.** Massif and Memcheck run 20-50x slower than native,
  so even the 60-frame video would cost hours under Massif. Massif runs on the image workload
  (model load plus one frame, where the heap peak actually forms) and on a 5-frame video for
  pipeline steady-state behaviour. This supersedes D-8's 60-frame choice.
- **[D-14] The single-thread finding gets a measured candidate.**
  `src/backends/onnx_runtime_backend.cpp:25` calls `SetIntraOpNumThreads(1)`, so ONNX Runtime
  executes with one intra-op thread. The calibration's `user/elapsed` ratio of 0.95 confirms
  roughly one core is used out of the six the run was pinned to. Requirements forbid changing
  `src/` and forbid claiming a speedup without a separately measured candidate implementation,
  so the counterfactual is measured in a **git worktree** — an isolated checkout where the
  thread count is varied, built and timed — leaving the main checkout untouched. The committed
  tree keeps the shipped value; only the measurement crosses into the counterfactual.

### Recorded degradation of the delegation loop

The measurement sets run for tens of minutes. A worker capped at ~12 turns cannot supervise a
40-minute job — three packets already stopped at that ceiling on ordinary build round-trips.
The explicit degradation, per the "degrade explicitly rather than assuming the guardrail is
there" rule: the **planner drives the measurement scripts**, which are mechanical and carry no
design authority, while authoring `results.md` — the part that requires judgement about what
the numbers mean — stays delegated. This is a departure from "measurement workers write the
raw artifacts" and is recorded rather than quietly adopted.

## Third addendum — execution on Google Colab (2026-10-09)

Fixed before this pass measured anything. The September pass never produced `results.md`; its
raw artifacts under `/tmp` and its model no longer exist, so this pass measures afresh. It runs
`run_profile.sh` (this directory), which executes the protocol above and both addenda unattended,
and `summarize_profile.py`, which computes every summary statistic from the raw files.

| # | Previously | This pass | Reason |
|---|---|---|---|
| D-15 | i5-11400H workstation | Colab CPU runtime, High-RAM | The workstation's `perf_event_paranoid` is 4 and Valgrind is no longer installed; both need the user's `sudo`. Colab provides root, Valgrind and user-space `perf`. Its numbers are not comparable with any figure from the workstation. |
| D-16 | `cycles:u` call graph | `cycles:u`, falling back to `cpu-clock:u` | Colab's VM exposes no hardware PMU (`<not supported> instructions`, probe 2026-10-09). The fallback is the protocol's own. |
| D-17 | Hardware counter groups on the same host | `{cycles,instructions}`, `{branches,branch-misses}` and `{cache-references,cache-misses}` are attempted, recorded `UNRUN` on Colab, and run separately on a host with a PMU | Not available in the VM. Software counters run on Colab. |
| D-18 | `rfdetr-nano-1101.onnx` (rfdetr 1.10.1) | rf-detr nano detection at 640, exported on the runtime with the pinned `RFDETR_VERSION` | The earlier model is gone. Same family, size and input shape; a different export. |
| D-19 | `taskset -c 0-5` | One CPU per physical core, read from `thread_siblings_list` at run time | The core count depends on the runtime Colab allocates. The mask and sibling map are recorded. |
| D-20 | k3d and AnyDesk stopped by hand | Nothing to stop | A fresh Colab runtime runs only the notebook kernel. Load before and after each timing set is recorded. Colab VMs share physical hosts, so noise is still possible; the CV gate is the guard. |
| D-21 | Arm C (D-14) built ad hoc | Arm C is an automatic worktree with `SetIntraOpNumThreads(<pinned cores>)`, timed on the 30-frame video | The same counterfactual, now scripted; the diff is kept as `env/threads-counterfactual.diff`. |
| D-22 | Massif on the 5-frame video (D-13) | Stopped at a 3-hour cap: 3 h 02 min elapsed, 3 h 01 min CPU at 99.7 %, RSS 1.05 GB, no output written (`memory/massif-5f.stopped`). Recorded measured-infeasible, as D-12 did for the 600-frame fixture | Massif with `--stacks=yes` on the image alone took 41 min. The 5-frame run is UNRUN, so steady-state heap growth across frames is unmeasured. The peak itself forms at model load and is established by the image run. The cap was set during the run, not before it; that ordering is stated here rather than hidden. |
