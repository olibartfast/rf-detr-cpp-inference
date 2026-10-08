# CPU Performance and Memory Profiling Guide

This guide is for maintainers investigating CPU bottlenecks or memory spikes. It defines a repeatable measurement process and source-backed experiments; it does not claim that any candidate below is a measured bottleneck. Record the command, commit, backend, model, input, configuration, output checksum, and complete profiler output for every comparison. Never report an optimization result without measurements from the same machine and fixed workload.

## Local profiling environment

The development host inspected while this guide was written has an AMD Ryzen 7 5700U, 16 logical CPUs, and a 7.0.0-31-generic kernel. `perf`, `/usr/bin/time`, and generic `clang-format` are present; `clang-format-18`, Valgrind, and Heaptrack are unavailable. `/proc/sys/kernel/perf_event_paranoid` is `4`, and even this minimal probe fails because no supported performance events are available:

```bash
perf stat -e cycles,instructions -- true
```

Consequently, hardware counters, a runtime inference workload, Massif, and all proposed runtime measurements below are **UNRUN**. The examples are procedures, not results. Install or permission changes are outside this investigation.

## Establish a controlled workload

Build optimized code while retaining symbols for useful stack attribution:

```bash
cmake -S . -B build-perf -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build-perf --parallel
```

Use explicit paths and keep every input and option fixed:

```bash
MODEL=/absolute/path/to/model.onnx
VIDEO=/absolute/path/to/input.mp4
LABELS=/absolute/path/to/labels.txt
OUTPUT=/tmp/rfdetr-profile-output.mp4
APP=(./build-perf/inference_app "$MODEL" "$VIDEO" "$LABELS" --segmentation --output "$OUTPUT")
```

First run `"${APP[@]}"` once to expose startup, model loading, dynamic linking, and cold filesystem effects. Decide before measuring whether the question is startup-inclusive end-to-end latency or steady-state frame processing. The CLI measures an entire process, so steady-state work is better isolated with the existing benchmarks or additional stage timing around a long fixed video, excluding an explicitly recorded warm-up interval. Do not silently mix the two definitions.

Repeat each baseline and candidate enough times to see run-to-run variation, keep CPU frequency and competing load comparable, and verify that outputs still match the expected boxes, class IDs, scores, masks, frame count, and media properties. A faster run that changes or omits output is invalid.

## Identify CPU performance bottlenecks with `perf`

The Linux kernel's perf documentation includes the source manuals for [`perf stat`](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/tree/tools/perf/Documentation/perf-stat.txt) and [`perf record`](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/tree/tools/perf/Documentation/perf-record.txt); rendered [`perf stat`](https://man7.org/linux/man-pages/man1/perf-stat.1.html) and [`perf record`](https://man7.org/linux/man-pages/man1/perf-record.1.html) pages are easier to navigate. They document event counting, repeated runs, sampling, and call-graph collection. Start by checking which software and hardware events this kernel exposes:

```bash
perf list
```

Measure stable groups separately to reduce counter pressure:

```bash
perf stat -r 7 \
  -e task-clock,context-switches,cpu-migrations,page-faults \
  -- "${APP[@]}"

perf stat -r 7 \
  -e cycles,instructions,branches,branch-misses \
  -- "${APP[@]}"
```

`-r 7` runs the entire command seven times and reports variation. The first group uses scheduler and process events; the second requests hardware counters. Exact availability and names are CPU and kernel dependent. If `perf` reports unsupported events, do not substitute zeros. If it reports counters as a percentage of time enabled, events were multiplexed because more counters were requested than the PMU could schedule simultaneously. Split the list into smaller runs; heavily multiplexed counts are estimates and make small differences unreliable.

Interpret the counters together:

- **Elapsed time** is wall-clock duration for the requested scope. For video, also record processed frames and calculate frames per second. It is the primary end-to-end metric.
- **Task-clock** is accumulated scheduled CPU time. `task-clock / elapsed time` gives approximate average utilized cores when both use the same time unit. For example, 8 CPU-seconds over 4 wall-seconds means about two cores on average; it does not mean every stage used two cores continuously.
- **IPC** is `instructions / cycles`. A low IPC can result from cache misses, branch misprediction, dependency chains, or waits within sampled code. A high IPC can still accompany too much total work. Compare it only across the same workload and CPU.
- **Branch miss rate** is `branch-misses / branches`. A high rate matters only when branch execution is also substantial and samples attribute it to relevant hot code.
- **Cache events** such as `cache-misses` are generic mappings chosen by the platform. They are not a portable definition of last-level misses or memory bandwidth. Prefer CPU-specific events from `perf list`, document their definitions, and avoid comparing raw cache counts between different machines.
- **Context switches** can reflect normal pipeline blocking, oversubscription, or lock contention. **CPU migrations** may harm cache locality, but a small count alone proves nothing. Correlate both with lower throughput and stage/queue behavior.
- **Page faults** include usually inexpensive minor faults from first-touch allocation and potentially expensive major faults that require I/O. `perf stat`'s aggregate does not by itself distinguish a leak from normal allocation.

Collect samples after counters show the regression or expensive scope:

```bash
perf record -o /tmp/rfdetr-perf.data \
  -F 199 --call-graph dwarf -- "${APP[@]}"
perf report -i /tmp/rfdetr-perf.data
```

In `perf report`, **Self** is samples attributed directly to a symbol. **Children** includes samples in callees reached through its recorded call chains. A high-Children wrapper can simply call the true hot function; expand it before optimizing. Sampling is statistical, so short functions and short runs can be missed, and DWARF unwinding adds overhead. Repeat profiles and look for stable attribution.

Default CPU sampling describes code executing on a CPU. It does not explain off-CPU time while a stage sleeps on a queue, waits for I/O, or waits for GPU synchronization. Compare task-clock with elapsed time, add explicit per-stage and queue-wait timing, and use scheduler tracing when unexplained wall time dominates. For TensorRT, use a GPU timeline and device-memory profiler as well; CPU samples cannot identify kernel time or VRAM use.

## Analyze memory spikes

Start with peak process accounting on the optimized workload:

```bash
/usr/bin/time -v "${APP[@]}"
```

“Maximum resident set size” is peak **RSS**: resident process pages, including live allocations, allocator-retained pages, stacks, libraries, and file-backed mappings. It is neither live heap nor proportional ownership. While a run is active, inspect current totals and mapping attribution:

```bash
PID=12345
sed -n '1,80p' "/proc/$PID/status"
sed -n '1,120p' "/proc/$PID/smaps_rollup"
```

`VmRSS` is current RSS. `RssAnon`, `RssFile`, and `RssShmem` help separate anonymous memory from mapped files; `Pss` proportionally divides shared mappings. Sample at fixed intervals or around known pipeline stages, since one snapshot can miss a spike.

Use a separate plain, unsanitized Debug build for Valgrind. Sanitizers conflict with this measurement, and the optimized `perf` build answers a different question:

```bash
cmake -S . -B build-valg -G Ninja -DCMAKE_BUILD_TYPE=Debug
cmake --build build-valg --parallel

valgrind --tool=massif --stacks=yes --time-unit=ms \
  --massif-out-file=/tmp/rfdetr-massif.out \
  ./build-valg/inference_app "$MODEL" "$VIDEO" "$LABELS" \
  --segmentation --output /tmp/rfdetr-massif-output.mp4
ms_print /tmp/rfdetr-massif.out
```

The official [Massif manual](https://valgrind.org/docs/manual/ms-manual.html) explains snapshots, allocation trees, stack accounting, and `--pages-as-heap=yes`. Default Massif primarily answers which allocation stacks own live heap at peaks; it does not equal RSS. A second `--pages-as-heap=yes` run can expose page-level mappings, but changes the accounting model and should be labeled separately.

Distinguish these patterns before changing code:

- **Live heap** is memory still reachable from active allocations. A Massif peak allocation tree identifies its owners.
- **Allocation churn** is a high rate of allocate/free operations while live heap stays bounded. Heaptrack, when available, is useful for allocation counts, temporary allocations, and peak contributors. It is unavailable on the inspected host.
- **Allocator retention or fragmentation** occurs when objects are freed but the allocator keeps arenas/pages, so RSS does not fall. A reduced Massif live heap with unchanged RSS can be expected and does not disprove shorter object lifetime.
- **Leaks** are allocations no longer reachable. Run `valgrind --leak-check=full --show-leak-kinds=all ...` or the repository's `memcheck` target; distinguish definitely lost memory from intentional still-reachable caches.
- **Mappings and PSS** cover shared libraries, file-backed model data, and shared pages. Use `smaps_rollup` or full `smaps`; heap-only tools do not account for all mappings.
- **VRAM** is outside RSS and Massif. For TensorRT/DALI/CUDA, measure device allocations and their timeline with NVIDIA tooling, and correlate host/device peaks by stage.

## Source-backed hypotheses and controlled experiments

Treat each item as a candidate. Change one factor at a time, verify identical output, and retain the baseline data even when the hypothesis is rejected.

### Per-result full-frame mask resize

The CPU segmentation postprocessor calls `resize_threshold_mask` for every retained result in [`src/rfdetr_inference.cpp`](../../../src/rfdetr_inference.cpp), and that function allocates an `orig_w * orig_h` byte mask and performs half-pixel bilinear sampling over every output pixel in [`src/media.cpp`](../../../src/media.cpp). A competing cause is backend inference, score ranking, decoding, or video codec work.

Measure fixed synthetic output shapes and hit counts with `BM_CpuSegPostprocess`, then sweep only output dimensions and retained result count. Confirm with `perf report` attribution to `resize_threshold_mask`, instructions, elapsed time, and allocation profiles. Expected evidence is roughly result-count × pixel-count growth, with the resize gaining Self samples and mask allocations dominating live/transient heap. Experiment with safe loop/vectorization improvements or parallelism only after attribution. Preserve full-frame masks, shared global top-k, strict `score > threshold`, raw-logit mask threshold, the existing half-pixel resize rule, and CPU/GPU parity. Cropping masks or changing selection semantics is outside this optimization.

### Full-frame mask drawing

`draw_segmentation_masks` scans every image pixel for every full-size mask before drawing its box in [`src/media.cpp`](../../../src/media.cpp). A competing cause is resizing those masks or video encoding after drawing.

Profile the same precomputed masks with drawing enabled and with an instrumentation-only no-draw comparison, while retaining equivalent output verification outside the timed region. Record samples in `draw_segmentation_masks`/`blend_pixel`, elapsed time, instructions, and codec time separately. If drawing is responsible, these metrics should scale with masks × frame pixels. Experiment with cache-friendly traversal, safe SIMD, or parallel processing of disjoint pixels while preserving blend order where masks overlap. Do not change full-frame mask semantics or rendered output merely to reduce work.

### Confirmed extra mask copy

After creating a local full-frame `binary_mask`, the CPU path uses `masks.push_back(binary_mask)` in [`src/rfdetr_inference.cpp`](../../../src/rfdetr_inference.cpp), which deep-copies its byte vector. A competing cause is the required resize allocation itself, which is much larger work and remains after removing the copy.

Compare the current insertion with `masks.push_back(std::move(binary_mask))` under fixed synthetic results. Use allocation counts/bytes, peak live heap, instructions, and postprocess time. The expected change is removal of **one transient full-frame allocation and copy per retained result**; it does not remove the retained masks. Verify every mask byte and all result ordering. This is a narrow ownership transfer and must not be described as eliminating mask storage.

### Frame-slot result lifetime

Each `FrameSlot` owns result vectors and `clear_results()` in [`src/video_pipeline.hpp`](../../../src/video_pipeline.hpp). The inference stage clears a slot only immediately before its next inference, while the draw/write stage publishes it to `free_slots_` after its last writer/display consumer without clearing it in [`src/video_pipeline.cpp`](../../../src/video_pipeline.cpp). Thus retained mask objects can remain associated with idle free slots. A competing cause is allocator-retained capacity: clearing vectors destroys masks but may not return vector or allocator pages to the OS.

Measure RSS, Massif live heap, queue occupancy, throughput, and output across fixed dense video before and after clearing results **after the final writer/display consumer and before `free_slots_.push(slot_idx)`**. Expected evidence is a lower live-object peak proportional to free slots holding dense results; RSS may remain unchanged because of retained capacity or allocator behavior. Never clear before drawing, writing, or display finishes, and check shutdown paths. A smaller ring is a separate experiment: it may reduce the number of simultaneously retained frames/results, but must preserve pipeline overlap and DALI's prefetch-depth requirement from [`specs/gpu-pipeline.md`](../../gpu-pipeline.md).

A 1920 × 1080 byte mask is 2,073,600 bytes, about 1.98 MiB. For illustration only, 100 such masks in each of eight simultaneously populated slots would contain about 1.55 GiB of mask bytes, excluding frame buffers, vector capacity, tensors, codecs, runtimes, and VRAM. This is a hypothetical capacity calculation, not an observed peak or a statement that all slots hold 100 masks.

### Repeated output-cache allocation and copying

Every `run_inference` clears both caches, allocates one `std::vector<float>` per backend output, copies backend data into it, and pushes it into the cache in [`src/rfdetr_inference.cpp`](../../../src/rfdetr_inference.cpp). A competing cause is the backend's own internal tensors and execution cost.

Time repeated `run_inference` calls with fixed mock output shapes and separately profile an end-to-end real backend. Record allocations, copied bytes, peak live heap, cache-related samples, and elapsed time. Experiment with reusing existing vectors and capacity when output count and shapes permit. Reuse must still call the backend copy, preserve independent ownership after the backend returns, resize correctly when output count or any tensor shape changes, and remove stale outputs when the count shrinks. Expected evidence is fewer allocations and lower churn; retained capacity can keep RSS flat or higher, so RSS alone is not the acceptance metric.

## What the existing benchmark establishes

`BM_CpuSegPostprocess` in [`tests/benchmark/bench_gpu_pipeline.cpp`](../../../tests/benchmark/bench_gpu_pipeline.cpp) constructs dense synthetic backend outputs, runs inference once, and repeatedly invokes CPU segmentation postprocessing at 1920 × 1080. It covers selection/postprocess work, per-result mask resize, and the current mask insertion copy.

It does **not** cover repeated inference cache clear/allocation/copy, segmentation drawing, decoding or encoding, frame-slot result lifetime, queue occupancy/waiting, or end-to-end video throughput. Use it to isolate resize/copy experiments, then confirm any material change with the controlled video workload and memory tools above.

## Acceptance record

For each experiment, record baseline and candidate distributions rather than a single fastest run; profiler availability and multiplexing; startup-inclusive or steady-state scope; model/input hashes; result count and resolution; output correctness evidence; peak RSS; live heap peak; allocation count/bytes when available; elapsed time/FPS; and relevant `perf report` symbols. Report unsupported or unavailable checks as **UNRUN** with the exact reason. A source-level suspicion, hypothetical memory calculation, or profiler plan is never a measured speedup or confirmed bottleneck.
