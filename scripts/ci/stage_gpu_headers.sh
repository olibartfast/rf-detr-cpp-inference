#!/usr/bin/env bash
# Stage headers-only TensorRT and DALI prefixes for the CI compile gate.
#
# CI runners have no GPU and the gate never links or runs these libraries — it
# only has to compile src/backends/tensorrt_backend.cpp and src/gpu/ under
# -DWERROR=ON, which needs headers alone. The real distributions are far too
# large for a per-PR job (the TensorRT tarball is 6.2 GB, the DALI wheel 380 MB
# of which the headers are a few), so each prefix is assembled from the smallest
# artifact that carries the pinned headers:
#
#   TensorRT  <- libnvinfer-headers-dev + libnvonnxparsers-dev from NVIDIA's CUDA
#                apt repo (~130 KB together), unpacked with `dpkg-deb -x` so the
#                dependency on the 2 GB libnvinfer10 runtime is never resolved.
#   DALI      <- include/ out of the pip wheel, which is the same tree
#                scripts/fetch_dali.sh copies out of the Triton container.
#
# The shared libraries are stubs. The dependency resolver checks that the files
# exist before declaring the package found, and the gate builds only the static
# rfdetr_inference_lib target, so nothing ever links against them. A build that
# has to *run* needs the real thing: scripts/fetch_dali.sh and the TensorRT
# tarball pinned in cmake/deps/packages/TensorRT.cmake.
#
#   ./scripts/ci/stage_gpu_headers.sh [dest]   # default dest: ~/dependencies
#
# Then configure with -DTENSORRT_ROOTDIR=<dest>/tensorrt-headers
#                     -DDALI_ROOT=<dest>/dali-headers
set -euo pipefail

# Must match cmake/deps/packages/TensorRT.cmake (PROVIDED_VERSION + the cuda tag
# in PROVIDED_URL) and the DALI build inside the Triton image pinned in
# scripts/fetch_dali.sh. See "Known pin duplications" in specs/tech-stack.md.
TENSORRT_DEB_VERSION="${TENSORRT_DEB_VERSION:-10.13.3.9-1+cuda13.0}"
DALI_VERSION="${DALI_VERSION:-1.51.2}"

CUDA_REPO="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64"
DALI_INDEX="https://pypi.nvidia.com/nvidia-dali-cuda120"

DEST="${1:-${HOME}/dependencies}"
TRT_DIR="${DEST}/tensorrt-headers"
DALI_DIR="${DEST}/dali-headers"

tmp="$(mktemp -d)"
trap 'rm -rf "${tmp}"' EXIT

# A .so that exists and exports nothing. Enough for the dependency resolver's
# existence check; useless to a linker, which is the point — see the header.
stub_lib() {
    local out="$1"
    mkdir -p "$(dirname "${out}")"
    echo 'void rfdetr_ci_stub(void) {}' | cc -shared -x c - -o "${out}"
}

if [[ -f "${TRT_DIR}/include/NvInfer.h" ]]; then
    echo "TensorRT headers already staged at ${TRT_DIR}"
else
    echo "Staging TensorRT ${TENSORRT_DEB_VERSION} headers -> ${TRT_DIR}"
    for pkg in libnvinfer-headers-dev libnvonnxparsers-dev; do
        curl -fsSL -o "${tmp}/${pkg}.deb" \
            "${CUDA_REPO}/${pkg}_${TENSORRT_DEB_VERSION}_amd64.deb"
        dpkg-deb -x "${tmp}/${pkg}.deb" "${tmp}/trt-root"
    done
    mkdir -p "${TRT_DIR}/include"
    cp -a "${tmp}/trt-root/usr/include/x86_64-linux-gnu/." "${TRT_DIR}/include/"
    stub_lib "${TRT_DIR}/lib/libnvinfer.so"
    stub_lib "${TRT_DIR}/lib/libnvonnxparser.so"
fi

if [[ -f "${DALI_DIR}/include/dali/c_api.h" ]]; then
    echo "DALI headers already staged at ${DALI_DIR}"
else
    echo "Staging DALI ${DALI_VERSION} headers -> ${DALI_DIR}"
    wheel="nvidia_dali_cuda120-${DALI_VERSION}-py3-none-manylinux2014_x86_64.whl"
    curl -fsSL -o "${tmp}/${wheel}" "${DALI_INDEX}/${wheel}"
    unzip -q "${tmp}/${wheel}" 'nvidia/dali/include/*' -d "${tmp}/dali-root"
    mkdir -p "${DALI_DIR}"
    cp -a "${tmp}/dali-root/nvidia/dali/include" "${DALI_DIR}/"
    # PROVIDED_LIBRARIES for DALI are relative to the root, not root/lib —
    # fetch_dali.sh puts the real ones there too.
    stub_lib "${DALI_DIR}/libdali.so"
    stub_lib "${DALI_DIR}/libdali_operators.so"
fi

for guard in "${TRT_DIR}/include/NvInfer.h" "${TRT_DIR}/include/NvOnnxParser.h" \
             "${DALI_DIR}/include/dali/c_api.h" "${DALI_DIR}/include/dali/operators.h"; do
    if [[ ! -f "${guard}" ]]; then
        echo "error: ${guard} missing after staging" >&2
        exit 1
    fi
done

echo "Staged: TENSORRT_ROOTDIR=${TRT_DIR} DALI_ROOT=${DALI_DIR}"
