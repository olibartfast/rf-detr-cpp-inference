#!/usr/bin/env bash
# run_gate.sh — unattended driver for the gpu-verify checklist on a rented GPU box.
#
# Runs every part of .claude/skills/gpu-verify/SKILL.md that is executable today,
# records the parts that are not, arms a deadline watchdog so a hang cannot bill
# forever, and stops the instance when it is done.
#
#   ./scripts/run_gate.sh                                   # defaults below
#   MODEL=~/rfdetr.onnx VIDEO=~/long.mp4 ./scripts/run_gate.sh
#   DEADLINE_HOURS=1 CUDA_ARCH=89 SKIP_DEFAULT_PATH=1 ./scripts/run_gate.sh
#
# Run it under tmux so a dropped SSH session does not kill it — a rented box
# keeps billing whether or not your laptop is open:
#   tmux new -s gate './scripts/run_gate.sh; bash'
#
# Pull results to your own machine at any time (run this locally, not here):
#   rsync -av <instance>:~/gate-results/ ./gate-results/
#
# What it cannot do: the parity tolerances in the skill (preprocessed tensor
# 2e-2, scores 1e-3, box centres 1 px, mask IoU 0.999) have no fixtures to run
# against — tests/data/gpu_parity/ is roadmap Phase 2 and does not exist, and
# src/main.cpp has no output-path flag (Phase 1). Those checks are reported
# UNRUN. Per the skill: an unrun check is reported as unrun, never implied to
# have passed.
#
# Deliberately NOT `set -e`: a failing check is data, not a reason to abandon the
# remaining checks. Failures are recorded and the script continues.
set -uo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DALI_ROOT="${DALI_ROOT:-$HOME/dependencies/dali}"
RESULTS="${RESULTS:-$HOME/gate-results}"
LOG="${RESULTS}/gate.log"
SUMMARY="${RESULTS}/gate.summary"

# L4/L40S/RTX-Ada are 89, A4000/A10G/A10/A5000 are 86, A100 is 80, T4 is 75.
CUDA_ARCH="${CUDA_ARCH:-89}"

# Hard stop regardless of whether the gate finished. Re-armed on every run.
DEADLINE_HOURS="${DEADLINE_HOURS:-6}"
SELF_STOP="${SELF_STOP:-1}"

# Step 6 needs no GPU. On a rented box that is paid time for nothing — run it at
# home first and set this to 1 to keep the metered clock on GPU-only work.
SKIP_DEFAULT_PATH="${SKIP_DEFAULT_PATH:-0}"

MODEL="${MODEL:-}"                                   # .onnx or .engine — required for steps 2-5
VIDEO="${VIDEO:-}"                                   # >= 1000 frames — required for step 4
IMAGE="${IMAGE:-$REPO/data/dog.jpg}"
LABELS="${LABELS:-$REPO/data/coco-labels-91.txt}"

mkdir -p "$RESULTS"
: > "$SUMMARY"

note()   { echo "[$(date -Is)] $*" | tee -a "$LOG"; }
record() { printf '%-8s  %s\n' "$1" "$2" >> "$SUMMARY"; note "$1: $2"; }
pass()   { record PASS  "$1"; }
fail()   { record FAIL  "$1"; }
unrun()  { record UNRUN "$1"; }

# --- Watchdog -----------------------------------------------------------------
# Fires whether the gate finished, crashed, or hung. This is the layer that
# survives you forgetting about the box; the self-stop at the end only covers
# the clean path.
arm_watchdog() {
    if command -v brev >/dev/null 2>&1; then
        sudo systemd-run --on-active="${DEADLINE_HOURS}h" --unit=gate-watchdog \
            "$(command -v brev)" stop "$(hostname)" 2>/dev/null \
            && note "watchdog armed: brev stop in ${DEADLINE_HOURS}h" && return
    fi
    # Fallback: halt the guest. Verify once on the Brev console that a halted VM
    # registers as *stopped* (storage-only billing) and not billed-but-off.
    if sudo shutdown -h "+$((DEADLINE_HOURS * 60))" 2>/dev/null; then
        note "watchdog armed: shutdown in ${DEADLINE_HOURS}h (verify it bills as stopped)"
    else
        note "WARNING: could not arm a watchdog — stop this instance yourself"
    fi
}

# --- Environment --------------------------------------------------------------
capture_env() {
    note "=== environment ==="
    {
        nvidia-smi 2>&1 | head -12
        echo "--- nvcc ---";     nvcc --version 2>&1 | tail -2
        echo "--- TensorRT ---"; grep -rhoE 'TensorRT [0-9.]+|NV_TENSORRT_[A-Z]+ +[0-9]+' \
                                    /usr/include/x86_64-linux-gnu/NvInferVersion.h 2>/dev/null | head
        echo "--- DALI ---";     find "$DALI_ROOT" -maxdepth 1 2>&1 | head
        echo "--- repo ---";     git -C "$REPO" log --oneline -1 2>&1
    } > "${RESULTS}/environment.txt" 2>&1
    note "environment written to environment.txt (needed for the CHANGELOG entry, step 7)"
}

# --- Step 1: build the matrix -------------------------------------------------
step_build() {
    note "=== step 1: build matrix ==="

    if cmake -S "$REPO" -B "$REPO/build-gpu" -G Ninja \
             -DUSE_ONNX_RUNTIME=OFF -DUSE_TENSORRT=ON -DUSE_GPU_PIPELINE=ON \
             -DDALI_ROOT="$DALI_ROOT" -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
             -DCMAKE_BUILD_TYPE=Release -DWERROR=ON >> "$LOG" 2>&1 \
       && cmake --build "$REPO/build-gpu" --parallel >> "$LOG" 2>&1; then
        pass "full TensorRT + DALI + CUDA build"
    else
        # Everything downstream needs this binary, so this is the one fatal step.
        fail "full TensorRT + DALI + CUDA build"
        return 1
    fi

    # The halves must build independently — DALI needs no nvcc, CUDA postprocess
    # needs no DALI. A change that silently couples them is a regression.
    if cmake -S "$REPO" -B "$REPO/build-dali-only" -G Ninja \
             -DUSE_ONNX_RUNTIME=OFF -DUSE_TENSORRT=ON -DUSE_DALI=ON \
             -DDALI_ROOT="$DALI_ROOT" -DCMAKE_BUILD_TYPE=Release >> "$LOG" 2>&1 \
       && cmake --build "$REPO/build-dali-only" --parallel >> "$LOG" 2>&1; then
        pass "USE_DALI=ON alone"
    else
        fail "USE_DALI=ON alone"
    fi

    if cmake -S "$REPO" -B "$REPO/build-cudapost-only" -G Ninja \
             -DUSE_ONNX_RUNTIME=OFF -DUSE_TENSORRT=ON -DUSE_CUDA_POSTPROCESS=ON \
             -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
             -DCMAKE_BUILD_TYPE=Release >> "$LOG" 2>&1 \
       && cmake --build "$REPO/build-cudapost-only" --parallel >> "$LOG" 2>&1; then
        pass "USE_CUDA_POSTPROCESS=ON alone"
    else
        fail "USE_CUDA_POSTPROCESS=ON alone"
    fi

    # Guard check: these MUST fail at configure time. An architectural
    # commitment, so a successful configure here is the failure.
    for opt in USE_DALI USE_CUDA_POSTPROCESS; do
        if cmake -S "$REPO" -B "$REPO/build-guard-$opt" -G Ninja \
                 -DUSE_ONNX_RUNTIME=ON "-D${opt}=ON" >> "$LOG" 2>&1; then
            fail "guard: ${opt} + ONNX Runtime should FATAL_ERROR but configured cleanly"
        else
            pass "guard: ${opt} + ONNX Runtime rejected at configure time"
        fi
        rm -rf "$REPO/build-guard-$opt"
    done
}

# --- Step 2/3: the four combinations ------------------------------------------
step_combinations() {
    note "=== step 2/3: four combinations ==="
    local app="$REPO/build-gpu/inference_app"

    if [[ -z "$MODEL" || ! -f "$MODEL" ]]; then
        unrun "four combinations — no MODEL set (export .onnx and re-run)"
        return
    fi

    # Smoke only. Real parity needs golden fixtures and an output-path flag,
    # neither of which exists yet (roadmap Phases 1 and 2).
    run_combo() { # name, extra flags...
        local name="$1"; shift
        if "$app" "$MODEL" "$IMAGE" "$LABELS" "$@" > "${RESULTS}/combo-${name}.txt" 2>&1; then
            pass "combination ${name} ran (smoke only, NOT a parity check)"
        else
            fail "combination ${name} crashed — see combo-${name}.txt"
        fi
    }
    run_combo cpu-cpu
    run_combo gpupre-cpupost --gpu-preprocess
    run_combo cpupre-gpupost --gpu-postprocess --segmentation
    run_combo gpu-gpu        --gpu-preprocess --gpu-postprocess --segmentation

    unrun "tolerance checks (tensor 2e-2, scores 1e-3, centres 1px, mask IoU 0.999)
          — tests/data/gpu_parity/ does not exist; roadmap Phase 2"
    unrun "dense fixture >100 detections — not built; roadmap Phase 2"
}

# --- Step 4: memory and long-run safety ---------------------------------------
step_sanitizer() {
    note "=== step 4: compute-sanitizer 1000-frame run ==="
    if [[ -z "$MODEL" || -z "$VIDEO" || ! -f "$VIDEO" ]]; then
        unrun "compute-sanitizer long run — needs MODEL and a >=1000 frame VIDEO"
        return
    fi
    if ! command -v compute-sanitizer >/dev/null 2>&1; then
        unrun "compute-sanitizer long run — binary not on PATH (install CUDA toolkit)"
        return
    fi
    # Watch daliOutputRelease ordering here: releasing before the TensorRT
    # enqueue produces intermittent garbage, not a crash, so a short clean run
    # proves nothing. This is the check that catches it.
    compute-sanitizer --tool memcheck "$REPO/build-gpu/inference_app" \
        "$MODEL" "$VIDEO" "$LABELS" --segmentation --gpu-preprocess --gpu-postprocess \
        > "${RESULTS}/compute-sanitizer.txt" 2>&1
    if grep -qE '0 errors|ERROR SUMMARY: 0' "${RESULTS}/compute-sanitizer.txt"; then
        pass "compute-sanitizer: no findings on the long run"
    else
        fail "compute-sanitizer: findings present — see compute-sanitizer.txt"
    fi
}

# --- Step 5: benchmarks -------------------------------------------------------
step_benchmarks() {
    note "=== step 5: benchmarks ==="
    if cmake -S "$REPO" -B "$REPO/build-gpu-bench" -G Ninja \
             -DUSE_ONNX_RUNTIME=OFF -DUSE_TENSORRT=ON -DUSE_GPU_PIPELINE=ON \
             -DDALI_ROOT="$DALI_ROOT" -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
             -DCMAKE_BUILD_TYPE=Release -DBENCHMARKS=ON >> "$LOG" 2>&1 \
       && cmake --build "$REPO/build-gpu-bench" --parallel >> "$LOG" 2>&1 \
       && "$REPO/build-gpu-bench/benchmarks" > "${RESULTS}/benchmarks.txt" 2>&1; then
        pass "benchmarks ran — see benchmarks.txt"
    else
        fail "benchmarks build or run failed"
    fi

    unrun "per-stage four-combination benchmark — bench_gpu_pipeline.cpp does not
          exist; bench_preprocessing.cpp covers only sigmoid/cxcywh/normalize. Phase 2"
}

# --- Step 6: the default path is unchanged ------------------------------------
step_default_path() {
    note "=== step 6: default ONNX Runtime path ==="
    if [[ "$SKIP_DEFAULT_PATH" == "1" ]]; then
        unrun "step 6 skipped on this box — run it locally, it needs no GPU"
        return
    fi
    if cmake -S "$REPO" -B "$REPO/build-default" -G Ninja \
             -DCMAKE_BUILD_TYPE=Release >> "$LOG" 2>&1 \
       && cmake --build "$REPO/build-default" --parallel >> "$LOG" 2>&1 \
       && ctest --test-dir "$REPO/build-default" --output-on-failure -R UnitTests \
            > "${RESULTS}/unit-tests-default.txt" 2>&1; then
        pass "default ONNX Runtime build + UnitTests"
    else
        fail "default ONNX Runtime build or UnitTests — see unit-tests-default.txt"
    fi

    if ctest --test-dir "$REPO/build-gpu" --output-on-failure -R UnitTests \
           > "${RESULTS}/unit-tests-gpu.txt" 2>&1; then
        pass "UnitTests on the GPU build (test_gpu_postprocess runs, not skipped)"
    else
        fail "UnitTests on the GPU build — see unit-tests-gpu.txt"
    fi

    # Cannot be checked here by construction: this box has a device, so the
    # GTEST_SKIP() path never executes. Run it on a CPU-only machine.
    unrun "GPU tests report SKIPPED on a device-less machine — untestable on this
          box by construction; verify on a CPU-only host"
}

# --- Main ---------------------------------------------------------------------
main() {
    note "gate starting; results in ${RESULTS}"
    arm_watchdog
    capture_env

    step_build && {
        step_combinations
        step_sanitizer
        step_benchmarks
    }
    step_default_path

    note "=== summary ==="
    sort -k1,1 "$SUMMARY" | tee -a "$LOG"
    note "$(grep -c '^PASS'  "$SUMMARY") passed, \
$(grep -c '^FAIL'  "$SUMMARY") failed, \
$(grep -c '^UNRUN' "$SUMMARY") unrun"
    note "step 7 is yours: put these numbers, plus environment.txt, in CHANGELOG.md
          and state the UNRUN items plainly as unrun"

    sync
    if [[ "$SELF_STOP" == "1" ]] && command -v brev >/dev/null 2>&1; then
        note "stopping instance — storage-only billing from here"
        brev stop "$(hostname)"
    else
        note "SELF_STOP off or brev CLI missing — STOP THE INSTANCE YOURSELF"
    fi
}

main "$@"
