#!/usr/bin/env bash
# Build a TensorRT engine from an ONNX export, inside the pinned NGC container.
#
# TENSORRT_IMAGE defaults to nvcr.io/nvidia/tensorrt:${NGC_CONTAINER_TAG}-py3,
# derived from versions.env. Setting it in the environment still wins.
#
# --fp16 is passed only when the container's trtexec still has it. TensorRT 11 removed
# the flag with weak typing: the engine then takes the ONNX model's own precision, so
# mount an FP16-converted model for an FP16 engine (docs/export.md, "TensorRT 11 and FP16").
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
    /bin/bash -cx "fp16=\$(trtexec --help 2>&1 | grep -qE -- '^[[:space:]]*--fp16([[:space:]]|$)' && echo --fp16 || true); \
                   trtexec --onnx=model.onnx \
                            --saveEngine=/exports/model.engine \
                            --memPoolSize=workspace:4096 \
                            \${fp16} \
                            --useCudaGraph \
                            --useSpinWait \
                            --warmUp=500 \
                            --avgRuns=1000 \
                            --duration=10"
