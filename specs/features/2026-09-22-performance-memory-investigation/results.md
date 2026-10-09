# Results — Performance and Memory Investigation

Measured 2026-10-09 by `run_profile.sh` under [protocol.md](protocol.md) and its three addenda.
Every figure below comes from `summarize_profile.py` run over the raw artifacts, or from the raw
files directly; nothing is estimated.

**Raw artifacts were not retained.** They lived on the Colab runtime
(`/tmp/rfdetr-profile-results/`, tarball `rfdetr-profile-results.tar.gz`, 238,288 KiB,
SHA-256 `22ba2a7f980ee308…`; `SHA256SUMS` lists 855 files, SHA-256 `173c10b768b99c11…`) and were
deleted with the runtime at the user's choice not to download them. The hardware-counter run's
artifacts are on the workstation under `/tmp/rfdetr-profile-results-local/` (50 files,
`SHA256SUMS`). The figures here are transcribed from `summary.md` (SHA-256 `4b4db58d6fa9cb9e…`);
the checksums are given so a reader can tell what was measured, not so the files can be re-read.

## Environment

| | Colab (everything except hardware counters) | Workstation (hardware counters only) |
|---|---|---|
| CPU | Intel Xeon @ 2.20 GHz, 4 cores / 8 threads | AMD Ryzen 7 5700U, 8 cores / 16 threads |
| Pin mask | `0,1,2,3` (one per physical core) | `0,2,4,6,8,10,12,14` (one per physical core) |
| Memory | 51 GB | 38 GB |
| Kernel | 6.6.122+ | 7.0.0-31-generic |
| `perf_event_paranoid` | 2 | 4 → 1 for the run (user, D-7) |
| PMU | none (hardware events `<not supported>`) | yes |
| Load before / after timing | 0.28 / 1.27 | 1.06 / 1.97 (AnyDesk running, not stopped) |
| Revision | `28210bf` (driver), `7e90bdf` (arm C re-run) | `28210bf` |
| Model | rf-detr nano, detection, 640, rfdetr 1.11.2, SHA-256 `d1c5a40218faa31a…` | same export, different torch: `bac8622fae9825ab…` |
| 30-frame video | `8a1b3ef2e96a134b…` | `bc48deb4162613d4…` |

The September record retracted an "AMD Ryzen 7 5700U" description in favour of an
i5-11400H. `lscpu` on this workstation reports the Ryzen 7 5700U on 2026-10-09. This record
states what `lscpu` measured today and does not resolve the conflict further.

The two FFV1 videos come from the same `protocol.md` command and differ in bytes because the
ffmpeg builds differ, despite `-fflags +bitexact`. Both decode to 30 frames of 768x576 at 30/1.

## Throughput and timing — R-8, V-9, V-10, V-19

`/usr/bin/time -v`, pinned, one excluded warm-up per set. Arm A: `RelWithDebInfo`, original
protocol flags. Arm B: `Release` + `PROFILING=ON`, the canonical arm. Arm C: arm B's build of a
worktree where `SetIntraOpNumThreads(1)` becomes `(4)` (D-14, D-21). Every video run printed
`Processed N frames` with the expected N.

| Set | n | Elapsed median | min / max | CV | Max RSS median | CPU | FPS |
|---|---|---|---|---|---|---|---|
| A image | 7 | 2.690 s | 2.620 / 2.720 | 1.41 % | 326.1 MiB | 99 % | 0.372 |
| B image | 7 | 2.890 s | 2.730 / 3.040 | 3.89 % | 324.2 MiB | 99 % | 0.346 |
| A 30f | 7 | 61.360 s | 60.290 / 64.480 | 2.23 % | 675.0 MiB | 102 % | 0.489 |
| B 30f | 7 | 61.070 s | 60.380 / 62.350 | 1.10 % | 674.6 MiB | 102 % | 0.491 |
| **C 30f** | 7 | **21.030 s** | 20.240 / 21.370 | 2.05 % | 672.9 MiB | **382 %** | **1.427** |
| B 60f | 5 | 120.000 s | 119.790 / 120.200 | 0.14 % | 673.0 MiB | 102 % | 0.500 |

No set exceeded the 5 % CV gate, so no second set was needed.

- **Per-frame cost and startup (arm B, D-12 slope):** (120.000 − 61.070) / 30 = **1.964 s/frame**;
  fixed cost 61.070 − 30 × 1.964 = **2.140 s**.
- **Arm A vs arm B (V-19):** 61.360 s vs 61.070 s on 30 frames, a 0.5 % difference against CVs
  of 2.23 % and 1.10 %. **Inside the noise; no difference is claimed.** The image sets differ by
  7 % (2.690 vs 2.890 s) with CVs of 1.41 % and 3.89 %. That is a startup-dominated, 3-second
  workload, so it is reported but not ranked.
- **Arm C vs arm B (D-14):** 21.030 s vs 61.070 s, **2.90×** faster, with CPU rising from 102 % to
  382 % of the four pinned cores and RSS unchanged. The difference is about 60 times the larger of
  the two standard deviations.

Invalid set, kept and excluded: the first arm C build compiled the shipped source by mistake,
because `run_profile.sh` configured every build from `$REPO`. Its 7 runs (`timing/invalid-C-built-from-repo/`) measured
60.780 s median at 102 % CPU, the same as arm B. That is consistent with the cause, which
`7e90bdf` fixed. Before that, the first pass lost every timing run to a missing
`/usr/bin/time` (exit 127); it was re-run with the tool installed, and those failures appear in
`status.txt`.

## Counters — R-8, V-10

Software group: Colab, arm B, 30-frame video, 7 runs, pinned. Every enabled/running ratio was
100 %, so nothing was multiplexed.

| Event | Median | CV | Per frame |
|---|---|---|---|
| task-clock | 61,596.9 ms | 0.25 % | — |
| context-switches | 1,017 | 6.66 % | 33.9 |
| cpu-migrations | 108 | 12.26 % | 3.6 |
| page-faults (all minor) | 183,942 | 0.55 % | 6,131 |
| major-faults | 0 | — | 0 |

Average utilised cores = 61.60 s / 61.07 s = **1.01**. Context switches and migrations exceed 5 %
CV. They are small counts and are reported as measured.

Hardware groups: workstation, same build configuration, local export of the same model, 30-frame
video, 7 runs each, pinned. Every enabled/running ratio was 100 %.

| Metric | Value | Inputs (median) | CV |
|---|---|---|---|
| IPC | **2.236** | 401.6 G instructions / 179.6 G cycles | 0.02 % / 1.04 % |
| Branch miss | **1.013 %** | 96.08 M / 9.483 G branches | 2.23 % / 0.14 % |
| Generic cache miss | **6.111 %** | 1.392 G / 22.78 G references | 0.28 % / 0.29 % |

On Colab the three hardware groups are `UNRUN`: the VM has no PMU (D-16, D-17). The workstation
ran only these groups and is a different host, so its counters describe the same code on a
different CPU. They do not decompose the Colab timings.

## Call graph — R-9, V-11

`perf record -e cycles:u` was unavailable on Colab. Following the protocol, it fell back to
`cpu-clock:u` (reported by perf as `task-clock:uH`) at 199 Hz with DWARF call graphs: **12K
samples, 0 lost.**

| DSO | Self share |
|---|---|
| `libonnxruntime.so.1.28.0` | **95.97 %** |
| `libavcodec.so.60` (FFV1 decode) | 1.98 % |
| `libc.so.6` | 1.18 % |
| `inference_app` | 0.51 % |
| `libx264.so.164` (output encode) | 0.15 % |
| `libstdc++`, `libswscale`, `ld-linux` | 0.17 % combined |

**Unresolved-symbol share is 98.96 %, above the protocol's 10 % gate.** The cause is that the
prebuilt ONNX Runtime 1.28.0 library ships without symbols, so its 96 % appears as raw addresses.
Fixing attribution below the library would mean building ONNX Runtime from source with symbols,
which the requirements rule out (no new dependency or build path). The gate is therefore
recorded as **FAIL at symbol level**. The DSO-level attribution is resolved and is the figure
of record: inference inside ONNX Runtime is the only significant consumer, and this project's
own code is 0.51 %.

## Memory — R-10, V-12

| Measurement | Value |
|---|---|
| Native max RSS, image (`time -v`, B, n=7) | median 324.2 MiB, max 333.2 MiB (341,152 KiB) |
| Native max RSS, 30-frame video (B, n=7) | median 674.6 MiB, max 681.6 MiB (697,964 KiB) |
| Native max RSS, 60-frame video (B, n=5) | median 673.0 MiB, max 675.4 MiB (691,636 KiB) |
| Massif peak, image (Debug, `--stacks=yes`) | heap 389.2 MiB + allocator overhead 1.7 MiB + stacks 0.01 MiB = **390.9 MiB**, snapshot 80 |
| Massif, 5-frame video | **UNRUN**: stopped at a 3-hour cap (D-22) |

Massif peak owners, top level, together covering 97.2 % of the peak:

| Share | Bytes | Owner |
|---|---|---|
| 65.50 % | 268,435,456 | `libonnxruntime` (unresolved) — exactly 256 MiB |
| 24.09 % | 98,735,412 | `libonnxruntime` (unresolved) |
| 2.80 % | 11,484,143 | 3,440 sites below Massif's 1 % threshold |
| 2.54 % | 10,404,056 | `libonnxruntime` |
| 2.05 % | 8,388,608 | `libonnxruntime` — exactly 8 MiB |
| 1.38 % | 5,665,610 | `libonnxruntime` |
| 1.20 % | 4,915,200 | application `std::vector<float>` (`new_allocator.h:151`) |

RSS and live heap measure different things. RSS counts resident pages, including the mapped model
file, libraries and freed-but-retained memory. Live heap counts bytes allocated at an instant. The
image's RSS (324 MiB) is *below* its Massif peak (391 MiB): the two runs differ (Release vs Debug),
and Massif counts allocations that may never be touched and so never become resident. Video
roughly doubles RSS (675 MiB), and that does not grow between 30 and 60 frames (674.6 vs
673.0 MiB). That is the only steady-state evidence available with D-22's Massif run missing, and
it shows no growth. VRAM is out of scope for this CPU ONNX Runtime workload.

Memcheck, image, Debug: **0 errors**; definitely 0, indirectly 0 and possibly 0 bytes lost;
still reachable 48,528 bytes in 257 blocks; in use at exit 50,544 bytes in 278 blocks;
suppressed 0; exit 0. `heaptrack` is not installed and stays UNRUN, as in September.

## Microbenchmarks — R-12, R-13, R-16, V-15, V-16

`build-bench`: Release, `-DBENCHMARKS=ON -DWERROR=ON` (it builds, closing V-21's earlier
failure). Colab, pinned, 5 repetitions, mean shown, CV of the 5.

| Benchmark | Real mean | Real CV | CPU mean | Iterations |
|---|---|---|---|---|
| BM_PreprocessBgrImage/432 · 576 · 640 | 4.60 · 8.19 · 10.14 ms | 1.27 · 1.42 · 0.80 % | 4.58 · 8.14 · 10.09 ms | 151 · 85 · 70 |
| BM_BuildForegroundScores/100/91 · 300/91 | 59.9 · 178.2 µs | 2.47 · 1.90 % | 59.6 · 177.2 µs | 12,028 · 3,956 |
| BM_SelectTopkShipped/100/91 · 300/91 | 150.7 · 254.6 µs | 0.82 · 0.57 % | 149.9 · 253.3 µs | 4,717 · 2,768 |
| BM_SelectTopkBoundedHeap/100/91 · 300/91 | 160.9 · 272.2 µs | 2.23 · 0.87 % | 160.1 · 271.3 µs | 4,453 · 2,538 |
| BM_DrawDetections/1 · 10 · 100 | 0.556 · 5.49 · 55.3 µs | 1.21 · 3.10 · 0.99 % | 0.553 · 5.46 · 55.0 µs | 1,292,880 · 131,364 · 12,722 |

Every added case is size-parameterised and has a competing implementation where the protocol
requires one (V-16). The same run covers the pre-existing benchmarks, for example
`BM_CpuPreprocess/432` at 4.69 ms and `BM_CpuSegPostprocess` at 2.40 s per call.

**Top-k A/B:** the shipped `select_topk_multiclass` is **faster** than the bounded heap: 6.3 % at
100×91 and 6.5 % at 300×91, against CVs of at most 2.23 %. The difference is outside the
dispersion. The shipped implementation's per-call index vector costs less than the heap's
bookkeeping at these sizes, so the alternative is rejected and nothing changes in `src/`.

## Deferral claim — R-14, V-17

The roadmap defers GPU postprocessing for detection as "not a bottleneck". Measured, the detection
decode is foreground scoring (178.2 µs) plus top-k (254.6 µs) at the shipped 300×91, so
**0.43 ms per frame**. Against the measured 1,964 ms per frame that is **0.022 %**. Arm C has
no 60-frame set, so its per-frame cost is only an estimate: (21.03 s − 2.14 s startup) / 30 ≈
0.63 s, which puts the decode at about 0.07 %. Either way, **the measurement supports the deferral.**

## Bottlenecks, ranked

| # | Bottleneck | Measured evidence | Competing explanation | Next controlled experiment | Correctness risk |
|---|---|---|---|---|---|
| 1 | **ONNX Runtime runs single-threaded** (`SetIntraOpNumThreads(1)`, `src/backends/onnx_runtime_backend.cpp:25`) | 96 % of samples in ONNX Runtime; 1.01 cores utilised of 4 pinned; arm C at 4 threads is 2.90× faster at 382 % CPU | Arm C may be faster partly through per-run variance or a Colab host effect | Repeat arm C on the workstation at 1, 2, 4 and 8 threads to get a scaling curve, then choose the default (the thread count of the pin, `hardware_concurrency`, or a CLI flag) | Low for results, since intra-op threading does not change the arithmetic beyond float reduction order. Higher for embedding: a library that assumes one core changes its CPU footprint, so the default needs a decision, not just a patch |
| 2 | **Peak heap is dominated by fixed ONNX Runtime blocks** (256 MiB exactly, then 98.7 MiB) | Massif: 65.5 % in one 268,435,456-byte allocation | It may be the model's weights or a preallocated buffer that the model genuinely needs, not an arena default | Re-run Massif with ONNX Runtime's arena disabled (`DisableCpuMemArena`) and compare peak and runtime | Low for output; moderate for throughput if allocation churn returns |
| 3 | **Video RSS is about 2× image RSS** (675 vs 324 MiB) | `time -v`, n=7 each | The video pipeline's ring buffer (8 slots of decoded frames plus tensors) and the encoder, not a leak: RSS is flat from 30 to 60 frames | Vary `ring_buffer_size` and read RSS | Low |
| 4 | Video decode and encode | `libavcodec` 1.98 %, `libx264` 0.15 % | — | None warranted at this share | — |

On generalisation: the repeated-frame video gives invariant detections, predictable branches and
allocations, and very compressible input. The ranking holds for this pipeline's mechanics. It is
not a claim about diverse real video, and the 1.01 % branch-miss rate in particular is flattered
by it.

## What was not measured

- Massif on the 5-frame video: stopped at the 3-hour cap (D-22).
- Symbol-level attribution inside ONNX Runtime: the library is stripped (see the call-graph section).
- Hardware counters on Colab: no PMU in the VM (D-16, D-17). They were measured on the workstation instead.
- `heaptrack`: not installed.
- The raw artifacts themselves: not retained (see the top of this file).
