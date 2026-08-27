# Docker

The parametric `Dockerfile` and its inference-backend × media-backend image matrix.

> Part of the [RF-DETR C++ Inference](../README.md) documentation.

---

A single parametric `Dockerfile` builds the full **inference-backend × media-backend** matrix via two build args:

| `INFERENCE_BACKEND` | `MEDIA_BACKEND` | Image |
|---------------------|-----------------|-------|
| `onnx` (default)    | `ffmpeg` (default) | ONNX Runtime + FFmpeg/SDL2/stb |
| `onnx`              | `opencv`        | ONNX Runtime + OpenCV |
| `tensorrt`          | `ffmpeg`        | TensorRT + FFmpeg/SDL2/stb |
| `tensorrt`          | `opencv`        | TensorRT + OpenCV |
| `executorch`        | `ffmpeg`        | ExecuTorch + FFmpeg/SDL2/stb |
| `executorch`        | `opencv`        | ExecuTorch + OpenCV |

> **ExecuTorch images build the ExecuTorch C++ runtime from source** (there is no distro
> or registry package), so the first build is slow — it clones ExecuTorch with recursive
> submodules and installs a CPU-only `torch` wheel for the operator codegen. Pin a
> different runtime with `--build-arg EXECUTORCH_VERSION=<tag>`; it defaults to `v1.4.0`
> to match the exporter used by `rfdetr[executorch]==1.9.4`, and enables the optimized
> kernel set that 1.9.1+ `.pte` files need. The build applies the
> upstream `extension_evalue_util` install fix automatically. ExecuTorch links
> statically, so the runtime image ships no extra shared libraries and needs no GPU.

Build all six variants:

```bash
# ONNX Runtime (CPU) — FFmpeg/SDL2/stb media backend (default)
docker build -t rfdetr-onnx-ffmpeg .
# ONNX Runtime (CPU) — OpenCV media backend
docker build -t rfdetr-onnx-opencv --build-arg MEDIA_BACKEND=opencv .
# TensorRT (GPU) — FFmpeg/SDL2/stb media backend
docker build -t rfdetr-trt-ffmpeg --build-arg INFERENCE_BACKEND=tensorrt .
# TensorRT (GPU) — OpenCV media backend
docker build -t rfdetr-trt-opencv --build-arg INFERENCE_BACKEND=tensorrt --build-arg MEDIA_BACKEND=opencv .
# ExecuTorch (CPU) — FFmpeg/SDL2/stb media backend
docker build -t rfdetr-et-ffmpeg --build-arg INFERENCE_BACKEND=executorch .
# ExecuTorch (CPU) — OpenCV media backend
docker build -t rfdetr-et-opencv --build-arg INFERENCE_BACKEND=executorch --build-arg MEDIA_BACKEND=opencv .
```

Run (mount your model, image, and labels under `/data`):

```bash
# ONNX Runtime — use an .onnx model
docker run -v $(pwd)/data:/data -v $(pwd)/exports:/exports rfdetr-onnx-ffmpeg \
  /exports/model.onnx /data/dog.jpg /data/coco-labels-91.txt

# TensorRT — requires --gpus all and a .engine/.trt model
docker run --gpus all -v $(pwd)/data:/data -v $(pwd)/exports:/exports rfdetr-trt-opencv \
  /exports/model.engine /data/dog.jpg /data/coco-labels-91.txt

# ExecuTorch — CPU only, use a .pte model exported with the xnnpack delegate
docker run -v $(pwd)/data:/data -v $(pwd)/exports:/exports rfdetr-et-ffmpeg \
  /exports/model.pte /data/dog.jpg /data/coco-labels-91.txt
```

> The ONNX Runtime images are multi-stage and slim (Ubuntu 24.04 runtime). The TensorRT images use the `nvcr.io/nvidia/tensorrt:25.12-py3` base for the bundled CUDA/TensorRT runtime.
