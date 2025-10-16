export NGC_TAG_VERSION=25.09

docker run --rm -it --gpus=all \
    -v $(pwd)/exports:/exports \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v $HOME/Downloads/inference_model.onnx:/workspace/inference_model.onnx \
    -w /workspace \
    nvcr.io/nvidia/tensorrt:${NGC_TAG_VERSION}-py3 \
    /bin/bash -cx "trtexec --onnx=inference_model.onnx \
                            --saveEngine=/exports/model.engine \
                            --memPoolSize=workspace:4096 \
                            --fp16 \
                            --useCudaGraph \
                            --useSpinWait \
                            --warmUp=500 \
                            --avgRuns=1000 \
                            --duration=10"