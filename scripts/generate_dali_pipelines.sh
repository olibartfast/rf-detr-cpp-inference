#!/usr/bin/env bash
# Serialize the RF-DETR DALI preprocessing pipelines.
#
# Runs the generator inside a pinned Triton container so no local nvidia-dali pip
# install is needed — the same acquisition story as scripts/fetch_dali.sh.
#
#   ./scripts/generate_dali_pipelines.sh 560                 # -> data/dali/
#   ./scripts/generate_dali_pipelines.sh 560 /tmp/pipelines
#
# The resolution must match the model's input resolution. The C++ side validates
# the produced tensor size against the TensorRT input binding and fails with an
# explicit message if they disagree, so a stale file cannot go unnoticed.
set -euo pipefail

TRITON_IMAGE="${TRITON_IMAGE:-nvcr.io/nvidia/tritonserver:25.12-py3}"
RESOLUTION="${1:?usage: $0 <resolution> [output-dir]}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${2:-${repo_root}/data/dali}"

mkdir -p "${OUTPUT_DIR}"
output_abs="$(cd "${OUTPUT_DIR}" && pwd)"

# --gpus all is required: DALI serialization builds the pipeline, which needs a
# device for the "mixed" decoder and the GPU operators.
docker run --rm --gpus all \
    -v "${repo_root}/deploy/dali:/gen:ro" \
    -v "${output_abs}:/out" \
    "${TRITON_IMAGE}" \
    python3 /gen/generate_preprocess_pipeline.py \
        --resolution "${RESOLUTION}" \
        --output-dir /out

echo "Pipelines written to ${output_abs}"
ls -la "${output_abs}"
