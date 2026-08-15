---
name: gpu-verify
description: Runs the manual GPU and alternate-backend verification that CI cannot — build the TensorRT/DALI/CUDA matrix, run the four pre/post combinations against the parity tolerances, compute-sanitizer a long video run, record benchmarks. Trigger when the user says "verify the GPU path", "run the GPU gate", "test TensorRT manually", "parity check", or invokes /gpu-verify.
---

# GPU Verify

**CI runners are all `ubuntu-latest` with no GPU. TensorRT, ExecuTorch, DALI and CUDA paths are
never built or run by CI** ([specs/tech-stack.md](../../../specs/tech-stack.md)). This checklist is
the only thing standing between those paths and an unverified release. It is the roadmap Phase 4
exit gate, and it is also required before any release that touches them.

Written for any coding agent; every step is a shell command a human can run.

## Prerequisites

- NVIDIA GPU with the CUDA Toolkit installed manually (TensorRT implies CUDA 13.x)
- DALI staged once: `./scripts/fetch_dali.sh` → `~/dependencies/dali`
- A `.engine` or `.onnx` model, plus a test image and a video of at least 1000 frames
- Checked-in `.dali` pipelines exist for resolutions **432** and **576** only; anything else needs
  `./scripts/generate_dali_pipelines.sh <res>` with `--gpus all` Docker

## 1. Build the matrix

```bash
# TensorRT + full GPU pipeline
cmake -S . -B build-gpu -G Ninja -DUSE_ONNX_RUNTIME=OFF -DUSE_TENSORRT=ON \
      -DUSE_GPU_PIPELINE=ON -DDALI_ROOT=$HOME/dependencies/dali \
      -DCMAKE_BUILD_TYPE=Release -DWERROR=ON
cmake --build build-gpu --parallel
```

Also confirm the halves build independently — `-DUSE_DALI=ON` alone (no nvcc needed) and
`-DUSE_CUDA_POSTPROCESS=ON` alone — and that either one with `USE_ONNX_RUNTIME=ON` still fails at
configure time with a `FATAL_ERROR`. That guard is an architectural commitment, not a nicety.

## 2. The four combinations

Run every fixture in `tests/data/gpu_parity/` through all four:

| Combination | Flags |
|-------------|-------|
| CPU / CPU | *(none)* |
| GPU-pre / CPU-post | `--gpu-preprocess` |
| CPU-pre / GPU-post | `--gpu-postprocess --segmentation` |
| GPU / GPU | `--gpu-preprocess --gpu-postprocess --segmentation` |

`--gpu-postprocess` requires `--segmentation`; there is no GPU postprocess for detection or
keypoint, deliberately ([specs/roadmap.md](../../../specs/roadmap.md) → Deferred).

### Tolerances — every one is a number, none is negotiable

- [ ] Preprocessed tensor: `max |Δ| ≤ 2e-2` (a tolerance gate, never equality — DALI resize will
      not bit-match the CPU bilinear)
- [ ] Detection sets match on class and count, scores within `1e-3`
- [ ] Box centres within 1 px
- [ ] Mask IoU ≥ 0.999
- [ ] Assertions are on the **set** of detections, not the order — score-sort ties are the one
      legitimate ordering difference

## 3. The dense fixture

- [ ] The dense fixture yields **> 100** above-threshold detections, and both paths agree on the
      count. Natural images yield 10–50 and cannot distinguish a truncating postprocessor from a
      correct one, so a run that skips this fixture has not run the gate.

## 4. Memory and long-run safety

```bash
compute-sanitizer --tool memcheck ./build-gpu/inference_app <model> <video> <labels> \
    --segmentation --gpu-preprocess --gpu-postprocess
```

- [ ] A **1000-frame** video run completes with no leak and **no `compute-sanitizer` findings**
- [ ] Particular attention to `daliOutputRelease` ordering — release **after** the TensorRT enqueue.
      Getting it wrong produces intermittent garbage, not a crash, so a single clean short run
      proves nothing ([specs/gpu-pipeline.md](../../../specs/gpu-pipeline.md))

## 5. Benchmarks

```bash
cmake -S . -B build-gpu-bench -G Ninja -DUSE_ONNX_RUNTIME=OFF -DUSE_TENSORRT=ON \
      -DUSE_GPU_PIPELINE=ON -DDALI_ROOT=$HOME/dependencies/dali \
      -DCMAKE_BUILD_TYPE=Release -DBENCHMARKS=ON
cmake --build build-gpu-bench --parallel && ./build-gpu-bench/benchmarks
```

- [ ] Four stages timed separately — preprocess, H2D+infer, D2H, postprocess — for a still image
      and a video run, in all four combinations
- [ ] Numbers recorded **including the flat ones**. Expect a large win in segmentation postprocess
      and little or no end-to-end gain from GPU preprocessing on single still images; recording
      that it is flat is the result, not a failure

## 6. The default path is unchanged

- [ ] The default ONNX Runtime CPU build produces **bit-identical** results to before the change.
      Nothing in the GPU pipeline is allowed to move the default path.
- [ ] `ctest --test-dir build --output-on-failure -R UnitTests` passes on a normal build
- [ ] On a machine with `USE_CUDA_POSTPROCESS=ON` but **no** device, GPU tests report `SKIPPED`,
      not `FAILED`, and the skip is visible in the output

## 7. Record it

- [ ] `CHANGELOG.md` updated with what was verified, on which hardware, and with which driver,
      CUDA, TensorRT and DALI versions
- [ ] Anything **not** verified is stated plainly. An unrun check is reported as unrun, never
      implied to have passed
