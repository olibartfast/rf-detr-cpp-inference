#!/usr/bin/env bash
# scripts/check_version_sync.sh — guards the pins that cannot read versions.env.
#
# CMake and the shell scripts consume versions.env directly (cmake/versions.cmake,
# scripts/versions.sh). Four formats cannot: a Dockerfile's ARG defaults must be
# literals, and conanfile.txt, deploy/requirements.txt and the argparse defaults
# in deploy/export_*.py are plain data files with no include mechanism. They
# restate the values; this script fails when the restatement drifts.
#
#   ./scripts/check_version_sync.sh
#
# Run by .github/workflows/lint.yml, and by hand after editing versions.env.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/versions.sh
source "${repo_root}/scripts/versions.sh"

failures=0

# expect <file> <what> <value> <expected-line-regex>
# Passes when at least one line of <file> matches the regex.
expect() {
    local file="$1" what="$2" value="$3" pattern="$4"
    if grep -Eq -- "${pattern}" "${repo_root}/${file}"; then
        printf '  ok    %-26s %-28s %s\n' "${what}" "${value}" "${file}"
    else
        printf '  DRIFT %-26s %-28s %s\n' "${what}" "${value}" "${file}"
        printf '        versions.env implies a line matching: %s\n' "${pattern}"
        printf '        found instead:\n'
        # Show the nearest lines so the fix is obvious without opening the file.
        grep -nE -- "$(printf '%s' "${pattern}" | sed -E 's/=.*/=/; s/\^//')" \
            "${repo_root}/${file}" | sed 's/^/          /' || true
        failures=$((failures + 1))
    fi
}

# Escape a version so it is a literal inside an extended regex.
lit() { printf '%s' "$1" | sed -e 's/[][\.*^$+?(){}|/]/\\&/g'; }

echo "Checking pins that cannot read versions.env:"

# --- Dockerfile ARG defaults and derived base images --------------------------
expect Dockerfile "EXECUTORCH_VERSION" "${EXECUTORCH_VERSION}" "^ARG EXECUTORCH_VERSION=$(lit "${EXECUTORCH_VERSION}")$"
expect Dockerfile "TENSORRT_VERSION" "${TENSORRT_VERSION}" "^ARG TENSORRT_VERSION=$(lit "${TENSORRT_VERSION}")$"
expect Dockerfile "NGC_CONTAINER_TAG" "${NGC_CONTAINER_TAG}" "^ARG NGC_CONTAINER_TAG=$(lit "${NGC_CONTAINER_TAG}")$"
expect Dockerfile "DOCKER_BASE_IMAGE" "${DOCKER_BASE_IMAGE}" "^ARG DOCKER_BASE_IMAGE=$(lit "${DOCKER_BASE_IMAGE}")$"

# --- Conan recipe references --------------------------------------------------
expect conanfile.txt "FFMPEG_VERSION" "${FFMPEG_VERSION}" "^ffmpeg/$(lit "${FFMPEG_VERSION}")$"
expect conanfile.txt "SDL_VERSION" "${SDL_VERSION}" "^sdl/$(lit "${SDL_VERSION}")$"
expect conanfile.txt "GTEST_VERSION" "${GTEST_VERSION}" "^gtest/$(lit "${GTEST_VERSION}")$"
expect conanfile.txt "OPENCV_VERSION" "${OPENCV_VERSION}" "opencv/$(lit "${OPENCV_VERSION}")"

# --- Python export tooling ----------------------------------------------------
expect deploy/requirements.txt "RFDETR_VERSION" "${RFDETR_VERSION}" \
    "^rfdetr\[onnx\]==$(lit "${RFDETR_VERSION}")$"
for f in deploy/export_detection.py deploy/export_segmentation.py; do
    expect "${f}" "ONNX_OPSET_VERSION" "${ONNX_OPSET_VERSION}" \
        "--opset_version', default=$(lit "${ONNX_OPSET_VERSION}"),"
done

echo
if [[ "${failures}" -ne 0 ]]; then
    echo "${failures} pin(s) drifted from versions.env." >&2
    echo "Update the file(s) above, or change versions.env if the bump is intended." >&2
    exit 1
fi
echo "All pins agree with versions.env."
