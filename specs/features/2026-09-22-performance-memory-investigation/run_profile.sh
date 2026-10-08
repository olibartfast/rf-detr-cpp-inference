#!/usr/bin/env bash
# run_profile.sh — executes protocol.md (and its addenda) end to end, unattended.
#
# Written for the Colab CPU runtime (third addendum in protocol.md) but runs on any
# Linux host with perf, valgrind, ffmpeg and the repo's build dependencies:
#
#   ./specs/features/2026-09-22-performance-memory-investigation/run_profile.sh
#
# Raw artifacts go to $OUT (default /tmp/rfdetr-profile-results); summary.md and
# SHA256SUMS are written last. Deliberately not `set -e`: a failed step is recorded
# in status.txt and the remaining steps still run.
set -uo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
OUT="${OUT:-/tmp/rfdetr-profile-results}"
PERF="${PERF:-$(command -v perf || ls /usr/lib/linux-tools/*/perf 2>/dev/null | head -1)}"
TIMING_RUNS="${TIMING_RUNS:-7}"
LONG_RUNS="${LONG_RUNS:-5}"
LABELS="$REPO/data/coco-labels-91.txt"
IMAGE="$REPO/data/dog.jpg"

# STEPS selects sections; inputs and builds are idempotent and always run.
# A re-run of one section, e.g. STEPS=timing, appends to status.txt.
STEPS="${STEPS:-timing perf memory bench}"
want() { [[ " $STEPS " == *" $1 "* ]]; }

mkdir -p "$OUT"/{env,inputs,timing,perf,memory,bench,outputs,logs}
STATUS="$OUT/status.txt"
[[ "$STEPS" == "timing perf memory bench" ]] && : > "$STATUS"

# Every tool up front: a missing /usr/bin/time made a whole Colab timing pass
# fail with exit 127 on 2026-10-09.
for tool in /usr/bin/time taskset "$PERF" valgrind ms_print ffmpeg ffprobe ninja cmake python3; do
    command -v "$tool" > /dev/null || { echo "missing tool: $tool" >&2; exit 2; }
done
log()  { echo "[$(date -Is)] $*" | tee -a "$OUT/logs/run.log"; }
mark() { printf '%-6s %s\n' "$1" "$2" | tee -a "$STATUS"; }

# One thread per physical core, from the sibling map, so no two pinned CPUs share a core.
MASK="$(for f in /sys/devices/system/cpu/cpu[0-9]*/topology/thread_siblings_list; do
            cut -d, -f1 "$f" | cut -d- -f1; done | sort -un | paste -sd, -)"
PIN=(taskset -c "$MASK")

# --- Environment ---------------------------------------------------------------
log "environment"
{
    echo "date: $(date -Is)"
    echo "revision: $(git -C "$REPO" rev-parse HEAD)"
    echo "kernel: $(uname -r)"
    lscpu
    echo "--- memory"; free -m
    echo "--- sibling map"; cat /sys/devices/system/cpu/cpu[0-9]*/topology/thread_siblings_list | sort -u
    echo "pin mask: $MASK"
    echo "perf_event_paranoid: $(cat /proc/sys/kernel/perf_event_paranoid)"
    echo "--- tools"; cmake --version | head -1; g++ --version | head -1
    "$PERF" --version; valgrind --version; ffmpeg -version | head -1
    echo "--- load"; uptime
} > "$OUT/env/environment.md" 2>&1

# --- Inputs ----------------------------------------------------------------------
MODEL="$OUT/inputs/model.onnx"
if [[ ! -f "$MODEL" ]]; then
    log "export rf-detr nano (detection, 640) with the pinned rfdetr"
    (cd "$REPO" && python3 -m pip install -q -r deploy/requirements.txt \
        && python3 deploy/export_detection.py --model_type nano --output_dir "$OUT/inputs/export") \
        > "$OUT/logs/export.log" 2>&1
    cp "$(ls "$OUT"/inputs/export/*.onnx | head -1)" "$MODEL" \
        && mark PASS "model exported" || mark FAIL "model export — see logs/export.log"
fi

make_video() { # frames, path — protocol.md command, frame count varied (D-12)
    ffmpeg -hide_banner -loglevel error -y -loop 1 -framerate 30 -i "$IMAGE" \
        -frames:v "$1" -an -c:v ffv1 -level 3 -g 1 -pix_fmt bgr0 \
        -map_metadata -1 -fflags +bitexact -flags:v +bitexact "$2"
    local got
    got="$(ffprobe -v error -count_frames -select_streams v:0 \
        -show_entries stream=codec_name,width,height,r_frame_rate,nb_read_frames -of csv=p=0 "$2")"
    echo "$2: $got" >> "$OUT/inputs/videos.txt"
    [[ "$got" == "ffv1,768,576,30/1,$1" ]]
}
for n in 5 30 60; do
    make_video "$n" "$OUT/inputs/dog-${n}f.mkv" && mark PASS "video ${n}f" || mark FAIL "video ${n}f"
done
sha256sum "$MODEL" "$IMAGE" "$LABELS" "$OUT"/inputs/*.mkv > "$OUT/inputs/SHA256SUMS"

# --- Builds ----------------------------------------------------------------------
build() { # dir, cmake args...
    local dir="$1"; shift
    cmake -S "$REPO" -B "$dir" -G Ninja "$@" > "$OUT/logs/$(basename "$dir").log" 2>&1 \
        && cmake --build "$dir" --parallel >> "$OUT/logs/$(basename "$dir").log" 2>&1
}
log "builds"
# Arm A: protocol.md's original RelWithDebInfo flags. Arm B: the canonical Release +
# PROFILING=ON tree (plan T-15). One variable changed between them.
build "$REPO/build-perf" -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    "-DCMAKE_CXX_FLAGS_RELWITHDEBINFO=-O2 -g -DNDEBUG -fno-omit-frame-pointer" \
    && mark PASS "build arm A (RelWithDebInfo)" || mark FAIL "build arm A"
build "$REPO/build-prof" -DCMAKE_BUILD_TYPE=Release -DPROFILING=ON \
    && mark PASS "build arm B (Release + PROFILING)" || mark FAIL "build arm B"
build "$REPO/build-valg" -DCMAKE_BUILD_TYPE=Debug -DSANITIZERS=OFF \
    -DSTRICT_UBSAN=OFF -DTHREAD_SANITIZER=OFF -DBENCHMARKS=OFF \
    && mark PASS "build Debug for Valgrind" || mark FAIL "build Debug for Valgrind"
build "$REPO/build-bench" -DCMAKE_BUILD_TYPE=Release -DBENCHMARKS=ON -DWERROR=ON \
    && mark PASS "build benchmarks -DWERROR=ON (V-21)" || mark FAIL "build benchmarks -DWERROR=ON (V-21)"

# V-14 / V-22: PROFILING reaches a library TU; default OFF adds nothing.
cmake -S "$REPO" -B "$REPO/build-proftest-off" -G Ninja -DCMAKE_BUILD_TYPE=Release > /dev/null 2>&1
ninja -C "$REPO/build-prof" -t commands rfdetr_inference_lib | grep -m1 'processing_utils.cpp' > "$OUT/env/profiling-on.cmd"
ninja -C "$REPO/build-proftest-off" -t commands rfdetr_inference_lib | grep -m1 'processing_utils.cpp' > "$OUT/env/profiling-off.cmd"
if grep -q -- '-fno-omit-frame-pointer' "$OUT/env/profiling-on.cmd" \
   && ! grep -q -- '-fno-omit-frame-pointer' "$OUT/env/profiling-off.cmd"; then
    mark PASS "PROFILING=ON reaches rfdetr_inference_lib; OFF does not (V-14, V-22)"
else
    mark FAIL "PROFILING flag check — see env/profiling-*.cmd"
fi

# D-14 counterfactual: ONNX Runtime intra-op threads = pinned core count, in a
# worktree so the measured tree keeps the shipped value.
THREADS="$(tr ',' '\n' <<< "$MASK" | wc -l)"
WT="$OUT/worktree-threads"
git -C "$REPO" worktree add -f --detach "$WT" HEAD > /dev/null 2>&1
sed -i "s/SetIntraOpNumThreads(1)/SetIntraOpNumThreads(${THREADS})/" "$WT/src/backends/onnx_runtime_backend.cpp"
git -C "$WT" diff > "$OUT/env/threads-counterfactual.diff"
build "$WT/build-prof" -DCMAKE_BUILD_TYPE=Release -DPROFILING=ON \
    && mark PASS "build counterfactual (IntraOpNumThreads=${THREADS})" || mark FAIL "build counterfactual"

if want timing; then
# --- Timing ----------------------------------------------------------------------
timed() { # tag, app, input, expected-frames(0=image)
    local tag="$1" app="$2" input="$3" want="$4" i
    for ((i = 1; i <= TIMING_RUNS_NOW; i++)); do
        local base="$OUT/timing/${tag}-${i}"
        ( cd "$OUT/outputs" && /usr/bin/time -v -o "${base}.time" "${PIN[@]}" "$app" "$MODEL" "$input" "$LABELS" \
            --output "$OUT/outputs/${tag}.$([[ $want -eq 0 ]] && echo jpg || echo mp4)" ) \
            > "${base}.out" 2> "${base}.err"
        local rc=$?
        if [[ $rc -ne 0 ]] || { [[ $want -gt 0 ]] && ! grep -q "^Processed ${want} frames" "${base}.out"; }; then
            mark FAIL "${tag} run ${i}: exit ${rc} or wrong frame count"
        fi
    done
}
# protocol.md: a set whose elapsed-time CV exceeds 5 % gets a second set, and the
# first is kept. The summary reports both.
unstable() { # tag
    python3 - "$OUT/timing" "$1" <<'PY'
import glob, re, statistics, sys
v = []
for p in glob.glob(f"{sys.argv[1]}/{sys.argv[2]}-*.time"):
    t = re.search(r"Elapsed \(wall clock\) time.*?: ([\d:.]+)", open(p).read()).group(1)
    s = 0.0
    for part in t.split(":"):
        s = s * 60 + float(part)
    v.append(s)
sys.exit(0 if len(v) > 1 and 100 * statistics.stdev(v) / statistics.fmean(v) > 5 else 1)
PY
}
timed_set() { # tag, app, input, frames, runs
    TIMING_RUNS_NOW=$5 timed "$1" "$2" "$3" "$4"
    if unstable "$1"; then
        mark NOTE "$1: CV > 5 %, second set ${1}-set2 run (first kept)"
        TIMING_RUNS_NOW=$5 timed "${1}-set2" "$2" "$3" "$4"
    fi
}
uptime > "$OUT/timing/load-before.txt"
for arm in A:build-perf B:build-prof C:counterfactual; do
    tag="${arm%%:*}"; dir="${arm#*:}"
    app="$REPO/$dir/inference_app"; [[ "$tag" == C ]] && app="$WT/build-prof/inference_app"
    log "timing arm ${tag}"
    TIMING_RUNS_NOW=1 timed "warmup-${tag}-image" "$app" "$IMAGE" 0
    TIMING_RUNS_NOW=1 timed "warmup-${tag}-30f" "$app" "$OUT/inputs/dog-30f.mkv" 30
    [[ "$tag" != C ]] && timed_set "${tag}-image" "$app" "$IMAGE" 0 "$TIMING_RUNS"
    timed_set "${tag}-30f" "$app" "$OUT/inputs/dog-30f.mkv" 30 "$TIMING_RUNS"
done
log "timing arm B, 60f"
timed_set "B-60f" "$REPO/build-prof/inference_app" "$OUT/inputs/dog-60f.mkv" 60 "$LONG_RUNS"
uptime > "$OUT/timing/load-after.txt"
mark DONE "timing sets"

fi

if want perf; then
# --- perf stat -------------------------------------------------------------------
APP="$REPO/build-prof/inference_app"
VID="$OUT/inputs/dog-30f.mkv"
"$PERF" stat -e cycles,instructions -- true > "$OUT/perf/probe.txt" 2>&1
groups=(
    "sw:task-clock,context-switches,cpu-migrations,page-faults,minor-faults,major-faults"
    "ipc:{cycles,instructions}"
    "branch:{branches,branch-misses}"
    "cache:{cache-references,cache-misses}"
)
for g in "${groups[@]}"; do
    name="${g%%:*}"; events="${g#*:}"
    log "perf stat ${name}"
    for ((i = 1; i <= TIMING_RUNS; i++)); do
        ( cd "$OUT/outputs" && "${PIN[@]}" "$PERF" stat -x, -o "$OUT/perf/stat-${name}-${i}.csv" -e "$events" \
            "$APP" "$MODEL" "$VID" "$LABELS" --output "$OUT/outputs/perf.mp4" ) > /dev/null 2>&1
        # No PMU: perf may print <not supported>, or reject the whole {group} and
        # write no counter row at all (Colab, 2026-10-09). Both are UNRUN.
        if grep -q -e '<not supported>' -e '<not counted>' "$OUT/perf/stat-${name}-${i}.csv" \
           || ! grep -q '^[0-9]' "$OUT/perf/stat-${name}-${i}.csv"; then
            mark UNRUN "perf stat ${name}: counters not supported on this host — see perf/stat-${name}-1.csv"
            break
        fi
    done
done

log "perf record"
( cd "$OUT/outputs" && "${PIN[@]}" "$PERF" record -e cycles:u -F 199 --call-graph dwarf \
    -o "$OUT/perf/record.data" "$APP" "$MODEL" "$VID" "$LABELS" --output "$OUT/outputs/record.mp4" ) \
    > "$OUT/perf/record.log" 2>&1 \
|| { echo "cycles:u failed; retrying with cpu-clock:u (protocol fallback)" >> "$OUT/perf/record.log"
     ( cd "$OUT/outputs" && "${PIN[@]}" "$PERF" record -e cpu-clock:u -F 199 --call-graph dwarf \
        -o "$OUT/perf/record.data" "$APP" "$MODEL" "$VID" "$LABELS" --output "$OUT/outputs/record.mp4" ) \
        >> "$OUT/perf/record.log" 2>&1; }
"$PERF" report -i "$OUT/perf/record.data" --stdio --no-children --sort symbol,dso \
    > "$OUT/perf/report-self.txt" 2> /dev/null
"$PERF" report -i "$OUT/perf/record.data" --stdio --children --sort symbol,dso -g none \
    > "$OUT/perf/report-children.txt" 2> /dev/null
"$PERF" report -i "$OUT/perf/record.data" --stdio --no-children --sort dso \
    > "$OUT/perf/report-dso.txt" 2> /dev/null
[[ -s "$OUT/perf/report-self.txt" ]] && mark PASS "perf record + reports" || mark FAIL "perf record"

fi

if want memory; then
# --- Memory ----------------------------------------------------------------------
VAPP="$REPO/build-valg/inference_app"
log "massif image"
( cd "$OUT/outputs" && valgrind --tool=massif --stacks=yes --time-unit=B --detailed-freq=1 --max-snapshots=200 \
    --massif-out-file="$OUT/memory/massif-image.out" "$VAPP" "$MODEL" "$IMAGE" "$LABELS" \
    --output "$OUT/outputs/massif-image.jpg" ) > "$OUT/memory/massif-image.log" 2>&1
ms_print "$OUT/memory/massif-image.out" > "$OUT/memory/massif-image.txt" 2>&1 \
    && mark PASS "massif image + ms_print" || mark FAIL "massif image"
log "massif 5f video"
( cd "$OUT/outputs" && valgrind --tool=massif --stacks=yes --time-unit=B --detailed-freq=1 --max-snapshots=200 \
    --massif-out-file="$OUT/memory/massif-5f.out" "$VAPP" "$MODEL" "$OUT/inputs/dog-5f.mkv" "$LABELS" \
    --output "$OUT/outputs/massif-5f.mp4" ) > "$OUT/memory/massif-5f.log" 2>&1
ms_print "$OUT/memory/massif-5f.out" > "$OUT/memory/massif-5f.txt" 2>&1 \
    && mark PASS "massif 5f + ms_print" || mark FAIL "massif 5f"
log "memcheck image"
( cd "$OUT/outputs" && valgrind --tool=memcheck --leak-check=full --show-leak-kinds=all --track-origins=yes \
    --errors-for-leak-kinds=definite,indirect --error-exitcode=99 \
    "$VAPP" "$MODEL" "$IMAGE" "$LABELS" --output "$OUT/outputs/memcheck.jpg" ) > "$OUT/memory/memcheck.log" 2>&1
echo "exit: $?" >> "$OUT/memory/memcheck.log"
mark DONE "memcheck (findings do not invalidate the profile)"

fi

if want bench; then
# --- Benchmarks ------------------------------------------------------------------
log "benchmarks"
"${PIN[@]}" "$REPO/build-bench/benchmarks" --benchmark_repetitions=5 --benchmark_report_aggregates_only=false \
    --benchmark_out="$OUT/bench/benchmarks.json" --benchmark_out_format=json > "$OUT/bench/benchmarks.txt" 2>&1 \
    && mark PASS "benchmarks" || mark FAIL "benchmarks"

fi

# --- Summary ---------------------------------------------------------------------
python3 "$(dirname "${BASH_SOURCE[0]}")/summarize_profile.py" "$OUT" > "$OUT/summary.md" 2> "$OUT/logs/summary.err" \
    && mark PASS "summary.md" || mark FAIL "summary — see logs/summary.err"
(cd "$OUT" && find . -type f ! -name SHA256SUMS ! -name '*.data' -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
sha256sum "$OUT/perf/record.data" >> "$OUT/SHA256SUMS" 2> /dev/null
tar -C "$(dirname "$OUT")" -czf "${OUT}.tar.gz" --exclude='*.data' --exclude='worktree-threads' "$(basename "$OUT")"
log "done: $OUT/summary.md, ${OUT}.tar.gz"
