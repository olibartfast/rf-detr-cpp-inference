#!/usr/bin/env bash
# setup_colab.sh — prepare a Google Colab GPU runtime for scripts/run_gate.sh.
#
# Colab is not a rented box: no Docker, a preinstalled CUDA that is not the pinned
# one, a driver that may predate it, and a runtime that is recycled without
# warning. This installs what the gate needs and writes the environment it must
# run with to ${ENV_FILE}, which the notebook sources before every gate command:
#
#   ./scripts/colab/setup_colab.sh
#   source /content/rfdetr_gate_env.sh && ./scripts/run_gate.sh
#
# Driven by scripts/colab/gpu_gate.ipynb; procedure in specs/rented-gpu-runbook.md
# ("Google Colab").
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=scripts/versions.sh
source "${REPO}/scripts/versions.sh"

ENV_FILE="${ENV_FILE:-/content/rfdetr_gate_env.sh}"
DALI_ROOT="${DALI_ROOT:-/content/dependencies/dali}"
SKIP_DALI="${SKIP_DALI:-0}"

# CUDA_VERSION 13.3 -> apt suffix 13-3, install prefix /usr/local/cuda-13.3.
CUDA_DASHED="${CUDA_VERSION//./-}"
CUDA_HOME="/usr/local/cuda-${CUDA_VERSION}"
# CUDA 13.x needs an R580+ driver. Older drivers can still run it on data-centre
# cards (T4, L4, A100, ...) through the forward-compatibility package.
CUDA_MIN_DRIVER=580

SUDO=""
[[ "$(id -u)" -ne 0 ]] && SUDO="sudo"

step() { echo; echo "=== $* ==="; }

step "GPU"
nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv
gpu_arch="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '. ')"
driver_major="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)"
echo "CUDA_ARCH=${gpu_arch}  driver major=${driver_major}  nproc=$(nproc)  RAM=$(free -g | awk '/Mem:/{print $2}')G"

step "CUDA ${CUDA_VERSION} toolkit from the NVIDIA apt repository"
# shellcheck source=/dev/null
. /etc/os-release
distro="ubuntu${VERSION_ID//./}"
keyring="$(mktemp -d)/cuda-keyring.deb"
wget -q -O "$keyring" \
    "https://developer.download.nvidia.com/compute/cuda/repos/${distro}/x86_64/cuda-keyring_1.1-1_all.deb"
$SUDO dpkg -i "$keyring"
$SUDO apt-get update -qq
packages=(
    "cuda-toolkit-${CUDA_DASHED}"
    ninja-build pkg-config zstd
    libavcodec-dev libavformat-dev libavutil-dev libswscale-dev libsdl2-dev
    libgtest-dev ffmpeg
    xvfb xdotool   # scripts/check_display.sh: --display on a headless runtime
)
compat_dir=""
if [[ "$driver_major" -lt "$CUDA_MIN_DRIVER" ]]; then
    echo "driver ${driver_major} < ${CUDA_MIN_DRIVER}: adding cuda-compat-${CUDA_DASHED}"
    packages+=("cuda-compat-${CUDA_DASHED}")
    compat_dir="${CUDA_HOME}/compat"
fi
DEBIAN_FRONTEND=noninteractive $SUDO apt-get install -y -qq --no-install-recommends "${packages[@]}"

# The apt cmake on older Ubuntu predates CUDA 13 support in its CUDA language
# detection; the PyPI build is current.
python3 -m pip install --quiet --upgrade cmake

if [[ "$SKIP_DALI" != "1" ]]; then
    step "DALI ${DALI_VERSION} (pip source — Colab has no Docker)"
    DALI_SOURCE=pip "${REPO}/scripts/fetch_dali.sh" "$DALI_ROOT"
fi

step "environment -> ${ENV_FILE}"
{
    echo "# written by scripts/colab/setup_colab.sh on $(date -Is)"
    echo "export PATH=\"${CUDA_HOME}/bin:\$PATH\""
    echo "export CUDACXX=\"${CUDA_HOME}/bin/nvcc\""
    # Colab's own CUDA (/usr/local/cuda -> 12.x) sits on LD_LIBRARY_PATH; the
    # pinned toolkit, and compat when needed, must come first.
    echo "export LD_LIBRARY_PATH=\"${compat_dir:+${compat_dir}:}${CUDA_HOME}/lib64:\${LD_LIBRARY_PATH:-}\""
    echo "export CUDA_ARCH=\"${gpu_arch}\""
    echo "export DALI_ROOT=\"${DALI_ROOT}\""
    # Nothing to stop: Colab recycles the runtime itself, and the watchdog's
    # shutdown fallback is meaningless inside its container.
    echo "export WATCHDOG=0 SELF_STOP=0"
} > "$ENV_FILE"
cat "$ENV_FILE"

step "sanity check"
# shellcheck source=/dev/null
source "$ENV_FILE"
nvcc --version | tail -2
# A driver too old even for compat fails here, in seconds, rather than as a
# TensorRT engine-build error twenty minutes into the gate.
cat > /tmp/cuda_probe.cu <<'EOF'
#include <cstdio>
#include <cuda_runtime.h>
int main() {
    int driver = 0, runtime = 0;
    cudaDriverGetVersion(&driver);
    cudaRuntimeGetVersion(&runtime);
    cudaError_t err = cudaFree(nullptr);
    std::printf("driver API %d, runtime %d, context: %s\n", driver, runtime, cudaGetErrorString(err));
    return err == cudaSuccess ? 0 : 1;
}
EOF
nvcc -o /tmp/cuda_probe /tmp/cuda_probe.cu && /tmp/cuda_probe
echo "setup done"
