#!/usr/bin/env bash
# gate_in_container.sh — run scripts/run_gate.sh inside the pinned NGC TensorRT image.
#
# The gpu-verify checklist has to run on the exact TensorRT/CUDA/DALI the project
# pins, not on whatever the rented host happens to ship. A bare-metal attempt
# failed because the image's CUDA 13.x toolkit could not initialise on a host
# driver older than the CUDA major version; this harness sidesteps the host
# entirely by building and running inside the same base image dockerfile.trt
# uses.
#
#   MODEL=~/model.onnx VIDEO=~/long.mp4 ./scripts/gate_in_container.sh
#
# Everything the gate needs is bind-mounted, so results land on the host at
# $RESULTS even though the gate runs in the container.
#
# Environment:
#   MODEL              .onnx or .engine model, required by gate steps 2-5
#   VIDEO              >=1000 frame video, required by gate step 4
#   RESULTS            host directory for gate-results        (default ~/gate-results)
#   DALI_ROOT          staged DALI tree, optional             (default ~/dependencies/dali)
#   NGC_CONTAINER_TAG  image tag, from scripts/versions.sh    (default: pinned value)
#   NAME               container name                         (default rfdetr-gate)
#
# Requires: docker with the NVIDIA container runtime, and `--gpus all` access.
set -uo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# shellcheck source=scripts/versions.sh
source "$(dirname "${BASH_SOURCE[0]}")/versions.sh"

MODEL="${MODEL:-}"
VIDEO="${VIDEO:-}"
RESULTS="${RESULTS:-$HOME/gate-results}"
DALI_ROOT="${DALI_ROOT:-$HOME/dependencies/dali}"
NAME="${NAME:-rfdetr-gate}"
IMAGE="nvcr.io/nvidia/tensorrt:${NGC_CONTAINER_TAG}-py3"

if ! command -v docker >/dev/null 2>&1; then
    echo "error: docker not found" >&2
    exit 1
fi

mounts=(-v "$REPO:/workspace")
[[ -d "$DALI_ROOT" ]] && mounts+=(-v "$DALI_ROOT:/deps/dali")
[[ -n "$MODEL" && -f "$MODEL" ]] && mounts+=(-v "$MODEL:/model.${MODEL##*.}")
[[ -n "$VIDEO" && -f "$VIDEO" ]] && mounts+=(-v "$VIDEO:/video.${VIDEO##*.}")
mkdir -p "$RESULTS"
mounts+=(-v "$RESULTS:/results")

docker rm -f "$NAME" >/dev/null 2>&1

docker run -d --name "$NAME" --gpus all --network host \
    "${mounts[@]}" -w /workspace "$IMAGE" sleep infinity >/dev/null || {
    echo "error: could not start $IMAGE" >&2
    exit 1
}

# The image ships the compiler, CMake, ninja and the CUDA toolkit, but not the
# media development headers the ffmpeg backend needs.
if ! docker exec "$NAME" bash -c '
        set -e
        export DEBIAN_FRONTEND=noninteractive
        apt-get update -qq
        apt-get install -y -qq --no-install-recommends \
            ninja-build pkg-config ca-certificates \
            libavcodec-dev libavformat-dev libavutil-dev libswscale-dev libsdl2-dev
        # TensorRT is a system package in the image; give CMake the
        # include/ + lib/ prefix layout TENSORRT_ROOTDIR expects.
        mkdir -p /opt/trt-prefix
        ln -sfn /usr/include/x86_64-linux-gnu /opt/trt-prefix/include
        ln -sfn /usr/lib/x86_64-linux-gnu /opt/trt-prefix/lib
        ldconfig
    '; then
    echo "error: container setup failed" >&2
    exit 1
fi

# The watchdog has to live outside the container: `brev stop`/`shutdown` inside it
# would only affect the container. run_gate.sh is told to leave stopping alone.
gate_env=(
    TENSORRT_ROOTDIR=/opt/trt-prefix
    RESULTS=/results
    SELF_STOP=0
    WATCHDOG=0
)
[[ -d "$DALI_ROOT" ]] && gate_env+=(DALI_ROOT=/deps/dali)
[[ -n "$MODEL" && -f "$MODEL" ]] && gate_env+=(MODEL="/model.${MODEL##*.}")
[[ -n "$VIDEO" && -f "$VIDEO" ]] && gate_env+=(VIDEO="/video.${VIDEO##*.}")
# Pass the gate's own opt-outs through, so a re-run that only needs the device
# checks does not rebuild the card-independent matrix.
[[ -n "${SKIP_BUILD_MATRIX:-}" ]] && gate_env+=(SKIP_BUILD_MATRIX="$SKIP_BUILD_MATRIX")
[[ -n "${SKIP_DEFAULT_PATH:-}" ]] && gate_env+=(SKIP_DEFAULT_PATH="$SKIP_DEFAULT_PATH")
[[ -n "${CUDA_ARCH:-}" ]] && gate_env+=(CUDA_ARCH="$CUDA_ARCH")

docker exec -d -w /workspace "$NAME" env "${gate_env[@]}" \
    bash -c './scripts/run_gate.sh > /results/gate-container.log 2>&1'

echo "gate launched in container $NAME"
echo "  image:   $IMAGE"
echo "  results: $RESULTS"
echo "  follow:  docker exec $NAME tail -f /results/gate-container.log"
echo
echo "The container cannot stop the instance. Arm a watchdog on the host:"
echo "  sudo systemd-run --on-active=3h --unit=gate-watchdog brev stop \$(hostname)"
