#!/usr/bin/env bash
# Build a TensorRT engine from an ONNX export, inside the pinned NGC container.
#
# TENSORRT_IMAGE defaults to nvcr.io/nvidia/tensorrt:${NGC_CONTAINER_TAG}-py3,
# derived from versions.env. Setting it in the environment still wins.
#
# TensorRT 11 has no --fp16: the engine takes the ONNX model's own precision, so mount
# an FP16-converted model for an FP16 engine (docs/export.md, "TensorRT 11 and FP16").
set -euo pipefail

# shellcheck source=scripts/versions.sh
source "$(dirname "${BASH_SOURCE[0]}")/scripts/versions.sh"

docker run --rm -it --gpus=all \
    -v $(pwd)/exports:/exports \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v $HOME/Downloads/rfdetr-medium.onnx:/workspace/model.onnx \
    -w /workspace \
    "${TENSORRT_IMAGE}" \
    /bin/bash -cx "trtexec --onnx=model.onnx \
                            --saveEngine=/exports/model.engine \
                            --memPoolSize=workspace:4096 \
                            --useCudaGraph \
                            --useSpinWait \
                            --warmUp=500 \
                            --avgRuns=1000 \
                            --duration=10"
