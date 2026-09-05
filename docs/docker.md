# Docker

Three backend Dockerfiles build the **inference-backend × media-backend** image matrix. The
backend is chosen by which file you pass to `-f`; there is no bare `Dockerfile`, so a build
never silently defaults to a backend.

> Part of the [RF-DETR C++ Inference](../README.md) documentation.

---

| File | Backend | `MEDIA_BACKEND` | Image |
|------|---------|-----------------|-------|
| `dockerfile.onnxrt` | ONNX Runtime (CPU) | `ffmpeg` (default) | ONNX Runtime + FFmpeg/SDL2/stb |
| `dockerfile.onnxrt` | ONNX Runtime (CPU) | `opencv`        | ONNX Runtime + OpenCV |
| `dockerfile.executorch` | ExecuTorch (CPU, `.pte`) | `ffmpeg` (default) | ExecuTorch + FFmpeg/SDL2/stb |
| `dockerfile.executorch` | ExecuTorch (CPU, `.pte`) | `opencv`        | ExecuTorch + OpenCV |
| `dockerfile.trt` | TensorRT (GPU) | `ffmpeg` (default) | TensorRT + FFmpeg/SDL2/stb |
| `dockerfile.trt` | TensorRT (GPU) | `opencv`        | TensorRT + OpenCV |

`dockerfile.trt` also takes a `GPU_PIPELINE` build arg (`off` default | `dali` | `cuda` | `on`)
that layers DALI GPU preprocessing and CUDA segmentation postprocessing onto the TensorRT
backend — see the file's header for the full matrix.

> **ExecuTorch images build the ExecuTorch C++ runtime from source** (there is no distro
> or registry package), so the first build is slow — it clones ExecuTorch with recursive
> submodules and installs a CPU-only `torch` wheel for the operator codegen. Pin a
> different runtime with `--build-arg EXECUTORCH_VERSION=<tag>`; it defaults to `v1.4.0`
> to match the exporter used by `rfdetr[executorch]==1.10.0`, and enables the optimized
> kernel set that 1.9.1+ `.pte` files need. The build applies the
> upstream `extension_evalue_util` install fix automatically. ExecuTorch links
> statically, so the runtime image ships no extra shared libraries and needs no GPU.

Build all six variants:

```bash
# ONNX Runtime (CPU) — FFmpeg/SDL2/stb media backend (default)
docker build -f dockerfile.onnxrt -t rfdetr-onnx-ffmpeg .
# ONNX Runtime (CPU) — OpenCV media backend
docker build -f dockerfile.onnxrt -t rfdetr-onnx-opencv --build-arg MEDIA_BACKEND=opencv .
# TensorRT (GPU) — FFmpeg/SDL2/stb media backend
docker build -f dockerfile.trt -t rfdetr-trt-ffmpeg .
# TensorRT (GPU) — OpenCV media backend
docker build -f dockerfile.trt -t rfdetr-trt-opencv --build-arg MEDIA_BACKEND=opencv .
# ExecuTorch (CPU) — FFmpeg/SDL2/stb media backend
docker build -f dockerfile.executorch -t rfdetr-et-ffmpeg .
# ExecuTorch (CPU) — OpenCV media backend
docker build -f dockerfile.executorch -t rfdetr-et-opencv --build-arg MEDIA_BACKEND=opencv .
# TensorRT + GPU pipeline (DALI preprocessing + CUDA postprocessing)
docker build -f dockerfile.trt -t rfdetr-trt-gpu --build-arg GPU_PIPELINE=on .
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

# TensorRT + GPU pipeline
docker run --gpus all -v $(pwd)/data:/data -v $(pwd)/exports:/exports rfdetr-trt-gpu \
  /exports/model.engine /data/dog.jpg /data/coco-labels-91.txt \
  --gpu-preprocess --gpu-postprocess --segmentation
```

> The ONNX Runtime and ExecuTorch images are multi-stage and slim (Ubuntu 24.04 runtime). The
> TensorRT images use the `nvcr.io/nvidia/tensorrt:25.12-py3` base for the bundled
> CUDA/TensorRT runtime, and pull the DALI staging image only when `GPU_PIPELINE=dali|on`.
>
> **Both base images must stay Ubuntu 24.04.** The FFmpeg runtime library names baked into the
> runtime stage (`libavcodec60`, `libx264-164`, …) are the 24.04 set; a future `NGC_CONTAINER_TAG`
> bump that changes the TensorRT base's distro would break `dockerfile.trt` while leaving the
> CPU images green.

### Network access during the build

All three images need outbound HTTPS at build time. GTest (and, when enabled, Google
Benchmark) is resolved via `FetchContent`, which `git clone`s
`github.com/google/googletest.git` at configure time, unconditionally and in every backend.
`dockerfile.onnxrt` additionally downloads the ONNX Runtime archive from the GitHub
release; `dockerfile.executorch` additionally clones `github.com/pytorch/executorch.git`
and installs the CPU `torch` wheel from PyPI. A build where these are blocked fails with
`fatal: could not read Username for 'https://github.com'` (git cannot prompt without a TTY).

`dockerfile.trt` has **exactly one** GitHub dependency: the gtest clone — TensorRT comes
from the NGC base image (shimmed, not downloaded), DALI from the staged Triton image, and
stb/font8x8 are vendored in-tree. To build it — or any of the three past the gtest clone —
without GitHub access, pre-seed the googletest source at `third_party/googletest/`
(gitignored; the Dockerfiles detect it and pass `FETCHCONTENT_SOURCE_DIR_GTEST` so
configure skips the clone):

```bash
# On a connected machine, cloning the pinned tag (GTEST_VERSION in versions.env):
git clone --depth 1 --branch release-1.12.1 \
  https://github.com/google/googletest.git third_party/googletest
# Or reuse a source tree an earlier local build already fetched:
cp -r build/_deps/gtest-src third_party/googletest
```

Note `-DDEPS_OFFLINE=ON` does not cover this: GTest is `FETCHCONTENT`-acquired, which the
offline mode explicitly does not resolve. There is no offline path for the ONNX Runtime
archive download or the ExecuTorch from-source build yet.
