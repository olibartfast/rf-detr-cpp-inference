#!/usr/bin/env bash
# scripts/versions.sh — loads versions.env into the environment.
#
# Source it, never execute it:
#     source "$(dirname "${BASH_SOURCE[0]}")/versions.sh"
#
# Values already set in the environment are left alone, so every pin stays
# overridable on the command line exactly as before this file existed:
#     TRITON_IMAGE=nvcr.io/nvidia/tritonserver:26.01-py3 ./scripts/fetch_dali.sh
#
# The repo root is resolved from this file's own location, so callers at any
# depth (scripts/, scripts/ci/, the root) work without passing anything.

RFDETR_REPO_ROOT="${RFDETR_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RFDETR_VERSIONS_ENV="${RFDETR_VERSIONS_ENV:-${RFDETR_REPO_ROOT}/versions.env}"

if [[ ! -f "${RFDETR_VERSIONS_ENV}" ]]; then
    echo "versions.sh: ${RFDETR_VERSIONS_ENV} not found" >&2
    return 1 2>/dev/null || exit 1
fi

while IFS= read -r _line || [[ -n "${_line}" ]]; do
    # Strip surrounding whitespace, then skip blanks and comments.
    _line="${_line#"${_line%%[![:space:]]*}"}"
    _line="${_line%"${_line##*[![:space:]]}"}"
    [[ -z "${_line}" || "${_line}" == \#* ]] && continue
    [[ "${_line}" == *=* ]] || continue

    _key="${_line%%=*}"
    _value="${_line#*=}"
    [[ "${_key}" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue

    # Strip one layer of surrounding quotes, as a shell would.
    if [[ "${_value}" == \"*\" || "${_value}" == \'*\' ]]; then
        _value="${_value:1:${#_value}-2}"
    fi

    # Environment wins. `+x` distinguishes "unset" from "set but empty".
    if [[ -z "${!_key+x}" ]]; then
        export "${_key}=${_value}"
    fi
done < "${RFDETR_VERSIONS_ENV}"
unset _line _key _value

# --- Derived coordinates ------------------------------------------------------
# TENSORRT_SHORT_VERSION is also derived in cmake/versions.cmake — keep those two in
# step. The remaining three have no CMake consumer and are shell-only.
#   10.13.3.9            -> 10.13.3                        download URL directory
#   10.13.3.9 + cuda13.0 -> 10.13.3.9-1+cuda13.0           apt/.deb package suffix
: "${TENSORRT_SHORT_VERSION:=$(printf '%s' "${TENSORRT_VERSION}" | cut -d. -f1-3)}"
: "${TENSORRT_DEB_VERSION:=${TENSORRT_VERSION}-1+cuda${CUDA_VERSION}}"
: "${TRITON_IMAGE:=nvcr.io/nvidia/tritonserver:${NGC_CONTAINER_TAG}-py3}"
: "${TENSORRT_IMAGE:=nvcr.io/nvidia/tensorrt:${NGC_CONTAINER_TAG}-py3}"
export TENSORRT_SHORT_VERSION TENSORRT_DEB_VERSION TRITON_IMAGE TENSORRT_IMAGE
