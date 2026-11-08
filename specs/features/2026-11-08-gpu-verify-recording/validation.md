# Validation — GPU verification record (v0.7.1 gate)

## Environment

| | |
|---|---|
| GPU | NVIDIA RTX A4000, 15,352 MiB, compute capability 8.6 |
| Driver | 580.196.07 |
| Container | `nvcr.io/nvidia/tensorrt:26.08-py3` |
| CUDA | 13.4 (image toolkit, nvcc 13.4.59) |
| TensorRT | 11.2.1.2 |
| DALI | 2.2.0 (`nvidia-dali-cuda130`), staged by `fetch_dali.sh` |
| Model | `rfdetr-seg-medium`, 432x432 input, 4 outputs |
| Video | 1000 frames, 768x576, 25 fps |
| Commit | `a6a3472` |

## Results

| Check | Result |
|-------|--------|
| Full TensorRT + CUDA preprocessing + CUDA postprocessing build | PASS |
| Full TensorRT + DALI preprocessing + CUDA postprocessing build | PASS |
| Independent halves (`USE_CUDA_PREPROCESS` / `USE_DALI` / `USE_CUDA_POSTPROCESS` alone) | PASS on the same commit, earlier bare-metal run |
| Configure guards (each GPU option + ONNX Runtime, both preprocessors together) | PASS on the same commit, earlier bare-metal run |
| Combinations CPU/CPU, GPU-pre/CPU-post, CPU-pre/GPU-post, GPU/GPU | PASS (smoke) |
| Four-combination parity, `build-gpu` | PASS |
| Four-combination parity, `build-gpu-dali` | PASS |
| `compute-sanitizer` 1000-frame run | PASS on memory; see below |
| Benchmarks, GPU build | PASS |
| Default ONNX Runtime build + 11 CTest entries | PASS (local, CPU only) |

## compute-sanitizer

The first attempt was terminated by an external `SIGTERM` at 10.7 minutes, so it proved nothing and
was rerun rather than counted. The rerun completed the full 1000 frames:

```
Processed 1000 frames. Output: output_video.mp4
========= ERROR SUMMARY: 2 errors
```

Both reported "errors" are the same CUDA API diagnostic raised once at startup, inside
`createInferRuntime_INTERNAL` → `cuDeviceGetAttribute`:

```
CUDA API Error: Fabric handle support is dependent on IMEX channels.
Please ensure a valid IMEX channel has been set up on the system.
```

That is TensorRT probing multi-node fabric support on a single-GPU PCIe host with no IMEX channel
configured. It is not a memory finding: no `Invalid __global__ read/write`, no out-of-bounds, no
misaligned access was reported, and the run exited 0 after processing all frames.

## Benchmarks

`-DBENCHMARKS=ON`, `build-gpu-bench`, 4 vCPU. GPU preprocessing, transfer and segmentation
postprocessing are the rows that exist only on this hardware.

| Benchmark | Time | Throughput |
|-----------|------|------------|
| `BM_CudaPreprocessFrame/432` | 727 µs | 1376 /s |
| `BM_CudaPreprocessEncoded/432` | 2.30 ms | 434 /s |
| `BM_CpuPreprocess/432` | 3.24 ms | 309 /s |
| `BM_CpuPreprocessEncoded/432` | 12.16 ms | 82 /s |
| `BM_CpuPreprocessUpload/432` | 3.30 ms | 303 /s |
| `BM_TransferRoundTrip` | 772 µs | 9.08 GiB/s |
| `BM_GpuSegPostprocess` | 318 ms | 3.14 /s |
| `BM_CpuSegPostprocess` | 1729 ms | 0.58 /s |

CUDA frame preprocessing runs 4.5x faster than the CPU path and the decoded-JPEG path 5.3x; GPU
segmentation postprocessing is 5.4x the CPU implementation.

## Parity numbers

Measured by `integration_test_gpu_parity.cpp`, `data/dog.jpg` at 768x576, 4 detections on every
path. Columns are the worst case over detections.

| Comparison | max &vert;&Delta;score&vert; | max centre &Delta; (px) | min mask IoU |
|------------|------------|----------------------|--------------|
| CUDA postprocess only | 0 | 7.6e-06 | 1.0 |
| GPU preprocess (CUDA) | 0.0100856 | 1.97531 | 0.989344 |
| GPU preprocess + CUDA postprocess (CUDA) | 0 | 3.05e-05 | 1.0 |
| CUDA preprocess, PNG fallback | 0.0186591 | 2.41779 | 0.991254 |
| CUDA postprocess only (DALI build) | 0 | 7.6e-06 | 1.0 |
| GPU preprocess (DALI) | 0.0173271 | 5.67328 | 0.983827 |
| GPU preprocess + CUDA postprocess (DALI) | 0 | 6.1e-05 | 1.0 |

The bound actually enforced by the test is `{0.06, 1% of the longest side in px, 0.95}` for any GPU
preprocessor and `{1e-3, 1 px, 0.999}` where both sides share the CPU tensor. The numbers above sit
inside the former and outside the latter, which is why the two groups are separate: the engine
amplifies a 1e-6 tensor difference into several logits, so the GPU-preprocess path cannot be held to
the same-input bound. The reasoning is documented at `preprocess_tolerance` in the test.
