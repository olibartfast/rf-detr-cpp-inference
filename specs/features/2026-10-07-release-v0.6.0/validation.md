# Validation: Release v0.6.0

## Pre-release verification (2026-10-07, branch `feature/v0.6.0-verification`)

Hardware: RTX 3060 Laptop (sm_86, driver 610.43.02). Builds in `rfdetr-trt-builder:26.08`
(TensorRT 11.2.1.2) with `--gpus all`, configured `-DUSE_ONNX_RUNTIME=OFF -DUSE_TENSORRT=ON
-DUSE_CUDA_PREPROCESS=ON -DUSE_CUDA_POSTPROCESS=ON -DWERROR=ON`: 0 warnings.

### Greyscale and CMYK JPEG through the CUDA preprocessor

Closes open question 3 of the Phase 6 spec.

- PASS: fixtures `small_gray.jpg` (PIL mode `L`, 1 component) and `small_cmyk.jpg` (PIL mode
  `CMYK`, Adobe marker, 4 components), both made from `small.jpg`.
- PASS: `GpuParityCudaPreprocess.GreyscaleJpegMatchesCpu`. `probe()` accepts the greyscale JPEG
  with the stb dimensions; nvJPEG decodes it to BGR without error; tensor max |Δ| vs the CPU path
  = **0.0162** (≤ `1e-1`).
- PASS: `GpuParityCudaPreprocess.CmykJpegTakesStbFallback`. `probe()` rejects the CMYK JPEG
  and stb decodes it.
- PASS: `compute-sanitizer --tool memcheck --leak-check full` on both tests: 0 errors, 0 bytes
  leaked. Full `unit_tests` on the device: 78/78 pass.
- PASS: app, `rfdetr-seg-nano` at 576 (engine built from the ONNX), CPU vs
  `--gpu-preprocess --gpu-postprocess` on greyscale and CMYK versions of `data/dog.jpg`
  (768×576). Same detections and classes on both paths. Tolerances are score `0.06`, box centre
  1 % of the longer side (7.7 px):

  | Image | GPU decode | Detections | Max score Δ | Max centre Δ | Mask pixels Δ |
  |-------|------------|------------|-------------|--------------|---------------|
  | greyscale | nvJPEG | 3 / 3 | 0.0011 | 0.75 px | ≤ 0.36 % |
  | CMYK | stb fallback | 4 / 4 | 0.0037 | 0.15 px | ≤ 0.10 % |

### `model.export(format="tensorrt", fp16=True)` end to end

- Environment: `nvcr.io/nvidia/tensorrt:26.08-py3` (`TENSORRT_IMAGE`, TensorRT 11.2.1.2,
  polygraphy 0.53.5), `pip install -c <tensorrt==11.2.1.2> "rfdetr[tensorrt]==1.11.2"` with CPU
  torch 2.14.1. The constraint matters: the extra only asks for `tensorrt>=8.6.1`, and an engine
  from any other TensorRT version would not deserialize in the pinned runtime.
- PASS: `RFDETRNano().export(format="tensorrt", fp16=...)` and the same for `RFDETRSegNano`, with
  no `output_name`. rfdetr logs "TensorRT 11.2.1.2 is strongly typed; building the FP16 engine
  from a cast graph" and writes `rfdetr-nano_fp16.trt` / `rfdetr-seg-nano_fp16.trt`.
- PASS: FP16 is real. Engine size 60.2 vs 112.9 MB (detection) and 66.7 vs 128.2 MB
  (segmentation). `trtexec --loadEngine` GPU compute mean 2.03 vs 7.68 ms and 3.64 vs 13.24 ms
  (3.8× / 3.6×).
- PASS: every engine loads in the C++ TensorRT backend; I/O tensors are float32 (polygraphy:
  `dets`, `labels`, `masks` all `float32`).
- PASS: FP16 vs FP32 engine on `data/dog.jpg`, against the tolerances in requirements.md:

  | Model | Detections | Max score Δ | Max centre Δ | Mask IoU (78×78, per query) |
  |-------|------------|-------------|--------------|-----------------------------|
  | detection nano | 3 / 3, same classes | 0.0040 | 0.16 px | — |
  | segmentation nano | 3 / 3, same classes | 0.0031 | 0.13 px | ≥ 0.9987 (4 queries above 0.5) |

### Unverified, stated in the release notes

- `--display` playback on the GPU path: no display on the verification machine.

## Release gate

Filled in on `release/v0.6.0`.
