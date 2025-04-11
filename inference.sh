#!/bin/bash
MODEL_PATH=$HOME/Downloads/inference_model.onnx
IMAGE_PATH=data/dog.jpg
LABELS_PATH=data/coco-labels-91.txt
build/inference_app ${MODEL_PATH} ${IMAGE_PATH} ${LABELS_PATH}