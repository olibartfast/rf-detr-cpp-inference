#!/usr/bin/env bash
# Stage the DALI C++ libraries and headers for the GPU preprocessing path.
#
# NVIDIA publishes no standalone C++ DALI distribution — the headers and shared
# libraries live inside a pip wheel whose filename carries an opaque build
# number, so a pinned download URL cannot be kept working. This extracts them
# from a pinned Triton container instead, which is reproducible and needs no
# Python environment.
#
#   ./scripts/fetch_dali.sh [dest]        # default dest: ~/dependencies/dali
#   TRITON_IMAGE=nvcr.io/nvidia/tritonserver:26.01-py3 ./scripts/fetch_dali.sh
#
# Then configure with -DDALI_ROOT=<dest>.
set -euo pipefail

# TRITON_IMAGE defaults to nvcr.io/nvidia/tritonserver:${NGC_CONTAINER_TAG}-py3,
# derived from versions.env. Setting it in the environment still wins.
# shellcheck source=scripts/versions.sh
source "$(dirname "${BASH_SOURCE[0]}")/versions.sh"
DEST="${1:-${HOME}/dependencies/dali}"
WHEEL_DIR="/opt/tritonserver/backends/dali/wheel/dali/nvidia/dali"

if [[ -f "${DEST}/include/dali/c_api.h" ]]; then
    echo "DALI already staged at ${DEST}"
    exit 0
fi

mkdir -p "${DEST}"
dest_parent="$(cd "${DEST}/.." && pwd)"
dest_name="$(basename "${DEST}")"

# `cp -a .` copies the hidden .libs/ directory as well. It is not optional:
# libdali.so has DT_NEEDED entries for ~24 vendored libraries (libjpeg, ffmpeg,
# aws-sdk, ...) that RUNPATH resolves through $ORIGIN/.libs.
docker run --rm -v "${dest_parent}:/out" "${TRITON_IMAGE}" \
    sh -lc "cp -a ${WHEEL_DIR}/include /out/${dest_name}/ \
         && cp -a ${WHEEL_DIR}/.libs /out/${dest_name}/ \
         && cp -a ${WHEEL_DIR}/libdali.so ${WHEEL_DIR}/libdali_core.so \
                  ${WHEEL_DIR}/libdali_kernels.so ${WHEEL_DIR}/libdali_operators.so \
                  /out/${dest_name}/"

if [[ ! -f "${DEST}/include/dali/c_api.h" ]]; then
    echo "error: ${DEST}/include/dali/c_api.h missing after extraction" >&2
    echo "       the DALI wheel layout in ${TRITON_IMAGE} may have changed" >&2
    exit 1
fi

unresolved="$(ldd "${DEST}/libdali.so" 2>/dev/null | grep -c 'not found' || true)"
if [[ "${unresolved}" != "0" ]]; then
    echo "warning: ${unresolved} unresolved libdali.so dependencies" >&2
    ldd "${DEST}/libdali.so" 2>/dev/null | grep 'not found' >&2 || true
fi

echo "DALI staged at ${DEST}"
echo "Configure with: -DDALI_ROOT=${DEST}"
