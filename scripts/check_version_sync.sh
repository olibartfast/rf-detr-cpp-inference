#!/usr/bin/env bash
# scripts/check_version_sync.sh — guards the pins that cannot read versions.env.
#
# CMake and the shell scripts consume versions.env directly (cmake/versions.cmake,
# scripts/versions.sh). Five formats cannot: the three backend Dockerfiles' ARG
# defaults must be literals, conanfile.txt, deploy/requirements.txt and the
# argparse defaults in deploy/export_*.py are plain data files with no include
# mechanism, and the README's version tables are the one place prose states pins
# (everything else in docs/ and specs/ points at versions.env). They restate the
# values; this script fails when the restatement drifts.
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
    if [[ ! -f "${repo_root}/${file}" ]]; then
        printf '  MISSING %-26s %-28s %s\n' "${what}" "${value}" "${file}"
        failures=$((failures + 1))
        return
    fi
    if grep -Eq -- "${pattern}" "${repo_root}/${file}"; then
        printf '  ok      %-26s %-28s %s\n' "${what}" "${value}" "${file}"
    else
        printf '  DRIFT   %-26s %-28s %s\n' "${what}" "${value}" "${file}"
        printf '          versions.env implies a line matching: %s\n' "${pattern}"
        printf '          found instead:\n'
        # Show the nearest lines so the fix is obvious without opening the file.
        grep -nE -- "$(printf '%s' "${pattern}" | sed -E 's/=.*/=/; s/\^//')" \
            "${repo_root}/${file}" | sed 's/^/          /' || true
        failures=$((failures + 1))
    fi
}

# expect_all <what> <value> <expected-line-regex> <file> [<file> ...]
# The same pin restated in several files (e.g. a build arg in each backend Dockerfile).
expect_all() {
    local what="$1" value="$2" pattern="$3"
    shift 3
    local f
    for f in "$@"; do expect "$f" "$what" "$value" "$pattern"; done
}

# Escape a version so it is a literal inside an extended regex.
lit() { printf '%s' "$1" | sed -e 's/[][\.*^$+?(){}|/]/\\&/g'; }

echo "Checking pins that cannot read versions.env:"

# --- Backend Dockerfile ARG defaults and derived base images ------------------
expect_all "EXECUTORCH_VERSION" "${EXECUTORCH_VERSION}" \
    "^ARG EXECUTORCH_VERSION=$(lit "${EXECUTORCH_VERSION}")$" \
    dockerfile.executorch
expect_all "TENSORRT_VERSION" "${TENSORRT_VERSION}" \
    "^ARG TENSORRT_VERSION=$(lit "${TENSORRT_VERSION}")$" \
    dockerfile.trt
expect_all "NGC_CONTAINER_TAG" "${NGC_CONTAINER_TAG}" \
    "^ARG NGC_CONTAINER_TAG=$(lit "${NGC_CONTAINER_TAG}")$" \
    dockerfile.trt
expect_all "CUDA_ARCHITECTURES" "${CUDA_ARCHITECTURES}" \
    "^ARG CUDA_ARCHITECTURES=$(lit "${CUDA_ARCHITECTURES}")$" \
    dockerfile.trt
expect_all "DOCKER_BASE_IMAGE" "${DOCKER_BASE_IMAGE}" \
    "^ARG DOCKER_BASE_IMAGE=$(lit "${DOCKER_BASE_IMAGE}")$" \
    dockerfile.onnxrt dockerfile.executorch dockerfile.trt

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

# --- README version tables ----------------------------------------------------
# "Versions at a Glance" and the Python export tooling table in README.md.
expect README.md "ONNX_RUNTIME_VERSION" "${ONNX_RUNTIME_VERSION}" \
    "^\| \*\*ONNX Runtime\*\* \| \*\*$(lit "${ONNX_RUNTIME_VERSION}")\*\* \|"
expect README.md "TENSORRT_VERSION" "${TENSORRT_VERSION}" \
    "^\| \*\*TensorRT\*\* \| \*\*$(lit "${TENSORRT_VERSION}")\*\* \+ CUDA Toolkit \*\*$(lit "${CUDA_VERSION}")\*\* series"
expect README.md "TENSORRT_LEGACY_VERSION" "${TENSORRT_LEGACY_VERSION}" \
    "^\| \*\*TensorRT\*\* \|.*compile-checked against $(lit "${TENSORRT_LEGACY_VERSION}")[;)]"
expect README.md "TENSORRT_COMPAT_VERSION" "${TENSORRT_COMPAT_VERSION}" \
    "^\| \*\*TensorRT\*\* \|.*also against $(lit "${TENSORRT_COMPAT_VERSION}")\)"
expect README.md "EXECUTORCH_VERSION" "${EXECUTORCH_VERSION}" \
    "^\| \*\*ExecuTorch\*\* \| \*\*$(lit "${EXECUTORCH_VERSION}")\*\* \|"
expect README.md "DALI_VERSION" "${DALI_VERSION}" \
    "^\| NVIDIA DALI \| $(lit "${DALI_VERSION}") \(staged from .nvcr\.io/nvidia/tritonserver:$(lit "${NGC_CONTAINER_TAG}")-py3."
expect README.md "DALI_LEGACY_VERSION" "${DALI_LEGACY_VERSION}" \
    "^\| NVIDIA DALI \|.*compile-checked against $(lit "${DALI_LEGACY_VERSION}")\)"
expect README.md "FFMPEG_VERSION/SDL_VERSION" "${FFMPEG_VERSION}/${SDL_VERSION}" \
    "^\| FFmpeg / SDL2 \|.*Conan pins $(lit "${FFMPEG_VERSION}") / $(lit "${SDL_VERSION}")\)"
expect README.md "OPENCV_VERSION" "${OPENCV_VERSION}" \
    "^\| OpenCV \|.*Conan coordinate $(lit "${OPENCV_VERSION}")\)"
expect README.md "GTEST_VERSION" "${GTEST_VERSION}" "^\| GoogleTest \| $(lit "${GTEST_VERSION}") "
expect README.md "GOOGLE_BENCHMARK_VERSION" "${GOOGLE_BENCHMARK_VERSION}" \
    "^\| Google Benchmark \| $(lit "${GOOGLE_BENCHMARK_VERSION}") \|"
for extra in onnx executorch tensorrt; do
    expect README.md "RFDETR_VERSION [${extra}]" "${RFDETR_VERSION}" \
        "^\| .rfdetr\[${extra}\]. \| .==$(lit "${RFDETR_VERSION}"). \|"
done
expect README.md "ONNX_OPSET_VERSION" "${ONNX_OPSET_VERSION}" \
    "^\| .rfdetr\[onnx\]. \|.*ONNX opset $(lit "${ONNX_OPSET_VERSION}") \|"
expect README.md "EXECUTORCH_VERSION [pip]" "${EXECUTORCH_VERSION}" \
    "^\| .rfdetr\[executorch\]. \|.*pinned $(lit "${EXECUTORCH_VERSION}") runtime"

echo
if [[ "${failures}" -ne 0 ]]; then
    echo "${failures} pin(s) drifted from versions.env." >&2
    echo "Update the file(s) above, or change versions.env if the bump is intended." >&2
    exit 1
fi
echo "All pins agree with versions.env."
