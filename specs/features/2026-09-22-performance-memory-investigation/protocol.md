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
