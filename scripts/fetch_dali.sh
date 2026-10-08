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
#   DALI_SOURCE=pip ./scripts/fetch_dali.sh   # no Docker (e.g. Google Colab)
#
# DALI_SOURCE=pip installs nvidia-dali-cuda<major>0==${DALI_VERSION} and its
# dependency wheels into a scratch directory with pip and copies the same files
# out of it. The Triton image's DALI backend is that same wheel tree, so the
# layout matches; the provenance differs (PyPI build vs the NGC one), which is
# why Docker stays the default.
#
# Then configure with -DDALI_ROOT=<dest>.
set -euo pipefail

# TRITON_IMAGE defaults to nvcr.io/nvidia/tritonserver:${NGC_CONTAINER_TAG}-py3,
# derived from versions.env. Setting it in the environment still wins.
# shellcheck source=scripts/versions.sh
source "$(dirname "${BASH_SOURCE[0]}")/versions.sh"
DEST="${1:-${HOME}/dependencies/dali}"
NV_DIR="/opt/tritonserver/backends/dali/wheel/dali/nvidia"
DALI_SOURCE="${DALI_SOURCE:-docker}"

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
#
# DALI also dlopen()s nvImageCodec (the image decoder) and its codec libraries
# from sibling wheel directories. ldd cannot see them, and without them
# --gpu-preprocess fails at decode time, so they are flattened next to libdali.so,
# where the $ORIGIN entries of DALI's and the codec extensions' RUNPATHs reach them.
copy_dali() { # nv_dir, out_dir — runs inside the container or on the host
    local nv="$1" out="$2" wheel="$1/dali"
    printf '%s' "cp -a ${wheel}/include ${out}/ \
         && cp -a ${wheel}/.libs ${out}/ \
         && cp -a ${wheel}/libdali.so ${wheel}/libdali_core.so \
                  ${wheel}/libdali_kernels.so ${wheel}/libdali_operators.so \
                  ${out}/ \
         && cp -a ${nv}/nvimgcodec/libnvimgcodec.so.0 ${nv}/nvimgcodec/extensions \
                  ${out}/ \
         && for lib_dir in cu13/lib libnvcomp/lib64 nvjpeg2k/lib nvtiff/lib; do \
                cp -a ${nv}/\${lib_dir}/*.so* ${out}/ || exit 1; \
            done"
}

case "${DALI_SOURCE}" in
    docker)
        docker run --rm -v "${dest_parent}:/out" "${TRITON_IMAGE}" \
            sh -lc "$(copy_dali "${NV_DIR}" "/out/${dest_name}")"
        ;;
    pip)
        # The wheel name carries the CUDA major (cuda130 for CUDA 13.x).
        dali_wheel="nvidia-dali-cuda${CUDA_VERSION%%.*}0==${DALI_VERSION}"
        pip_dir="$(mktemp -d)"
        trap 'rm -rf "${pip_dir}"' EXIT
        python3 -m pip install --quiet --target "${pip_dir}" \
            --extra-index-url https://pypi.nvidia.com "${dali_wheel}"
        sh -c "$(copy_dali "${pip_dir}/nvidia" "${DEST}")"
        ;;
    *)
        echo "error: DALI_SOURCE must be docker or pip, got '${DALI_SOURCE}'" >&2
        exit 1
        ;;
esac

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
