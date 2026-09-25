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
# tarball, both resolved from the versions pinned in versions.env.
#
#   ./scripts/ci/stage_gpu_headers.sh [dest]   # default dest: ~/dependencies
#
# Then configure with -DTENSORRT_ROOTDIR=<dest>/tensorrt-headers
#                     -DDALI_ROOT=<dest>/dali-headers
#
# TRT_HEADERS=compat stages TENSORRT_COMPAT_VERSION instead, into
# <dest>/tensorrt-compat-headers, from the public headers in the NVIDIA/TensorRT OSS
# repository. That is the forward-compat compile gate: it proves the backend still
# builds against a newer TensorRT before anything pins it. Configure with
# -DTENSORRT_ROOTDIR=<dest>/tensorrt-compat-headers.
#
# TRT_HEADERS=legacy is the backward-compat gate: the previous stack,
# TENSORRT_LEGACY_VERSION (apt headers, into <dest>/tensorrt-legacy-headers) with
# DALI_LEGACY_VERSION (into <dest>/dali-legacy-headers), so the TensorRT 10.x and
# DALI 1.x branches keep compiling after the pin moved past them. Configure with
# -DTENSORRT_ROOTDIR=<dest>/tensorrt-legacy-headers -DDALI_ROOT=<dest>/dali-legacy-headers.
set -euo pipefail

# TENSORRT_DEB_VERSION is derived as ${TENSORRT_VERSION}-1+cuda${CUDA_VERSION};
# DALI_VERSION is read straight from versions.env, where it is documented to
# track the DALI build inside the Triton image scripts/fetch_dali.sh pulls.
# Either can still be overridden from the environment.
# shellcheck source=scripts/versions.sh
source "$(dirname "${BASH_SOURCE[0]}")/../versions.sh"

CUDA_REPO="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64"
DALI_INDEX="https://pypi.nvidia.com/nvidia-dali-cuda130"

DEST="${1:-${HOME}/dependencies}"
TRT_HEADERS="${TRT_HEADERS:-pinned}"
case "${TRT_HEADERS}" in
    pinned) TRT_DIR="${DEST}/tensorrt-headers" ;;
    compat) TRT_DIR="${DEST}/tensorrt-compat-headers" ;;
    legacy) TRT_DIR="${DEST}/tensorrt-legacy-headers" ;;
    *)
        echo "error: TRT_HEADERS must be 'pinned', 'compat' or 'legacy', got '${TRT_HEADERS}'" >&2
        exit 1
        ;;
esac
if [[ "${TRT_HEADERS}" == legacy ]]; then
    DALI_DIR="${DEST}/dali-legacy-headers"
    DALI_VERSION="${DALI_LEGACY_VERSION}"
else
    DALI_DIR="${DEST}/dali-headers"
fi

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
elif [[ "${TRT_HEADERS}" == compat ]]; then
    # 11.3.0.99 -> v11.3, the tag scheme of github.com/NVIDIA/TensorRT for 11.x. (10.x
    # tags are v<major>.<minor>.<patch> and keep NvOnnxParser.h in a submodule, which is
    # why the legacy gate stages from apt instead.)
    trt_tag="v$(printf '%s' "${TENSORRT_COMPAT_VERSION}" | cut -d. -f1-2)"
    echo "Staging TensorRT ${TENSORRT_COMPAT_VERSION} headers (OSS ${trt_tag}) -> ${TRT_DIR}"
    # Sparse, blobless: only include/ is ever downloaded, not the plugin and sample trees.
    git -c advice.detachedHead=false clone --quiet --depth 1 --branch "${trt_tag}" --filter=blob:none --sparse \
        https://github.com/NVIDIA/TensorRT.git "${tmp}/trt-oss"
    git -C "${tmp}/trt-oss" sparse-checkout set include
    mkdir -p "${TRT_DIR}"
    cp -a "${tmp}/trt-oss/include" "${TRT_DIR}/"
    stub_lib "${TRT_DIR}/lib/libnvinfer.so"
    stub_lib "${TRT_DIR}/lib/libnvonnxparser.so"
else
    deb_version="${TENSORRT_DEB_VERSION}"
    if [[ "${TRT_HEADERS}" == legacy ]]; then
        # The legacy TensorRT was built for an older CUDA than CUDA_VERSION, so take its
        # newest +cuda<v> build from the repo index instead of pinning a second CUDA.
        deb_version="$(curl -fsSL "${CUDA_REPO}/Packages.gz" | gunzip \
            | grep -o "libnvinfer-headers-dev_${TENSORRT_LEGACY_VERSION}-1+cuda[0-9.]*_amd64" \
            | sed -e 's/^libnvinfer-headers-dev_//' -e 's/_amd64$//' | sort -V | tail -n1)"
        if [[ -z "${deb_version}" ]]; then
            echo "error: no TensorRT ${TENSORRT_LEGACY_VERSION} headers in ${CUDA_REPO}" >&2
            exit 1
        fi
    fi
    echo "Staging TensorRT ${deb_version} headers -> ${TRT_DIR}"
    for pkg in libnvinfer-headers-dev libnvonnxparsers-dev; do
        curl -fsSL -o "${tmp}/${pkg}.deb" \
            "${CUDA_REPO}/${pkg}_${deb_version}_amd64.deb"
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
    # The manylinux tag changes between releases (1.x manylinux2014, 2.x manylinux_2_28),
    # so take the x86_64 wheel name from the index rather than spelling it out.
    wheel="$(curl -fsSL "${DALI_INDEX}/" \
        | grep -o "nvidia_dali_cuda130-${DALI_VERSION}-py3-none-[a-z0-9_]*x86_64\.whl" | head -n1)"
    if [[ -z "${wheel}" ]]; then
        echo "error: no x86_64 DALI ${DALI_VERSION} wheel at ${DALI_INDEX}" >&2
        exit 1
    fi
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
