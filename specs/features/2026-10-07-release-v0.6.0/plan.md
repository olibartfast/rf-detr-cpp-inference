# Plan

1. (GPU) Add `small_gray.jpg` and `small_cmyk.jpg` to `tests/data/gpu_parity/` and the tests
   `GpuParityCudaPreprocess.GreyscaleJpegMatchesCpu` / `.CmykJpegTakesStbFallback` in
   `tests/unit/test_gpu_parity.cpp`. Run them, memcheck them, and run the app CPU vs
   `--gpu-preprocess --gpu-postprocess` on greyscale and CMYK versions of `data/dog.jpg`, in
   `rfdetr-trt-builder:26.08`.
2. (GPU) In `TENSORRT_IMAGE` (TensorRT 11.2.1.2), install `rfdetr[tensorrt]==1.11.2` constrained
   to the container's `tensorrt`, export nano detection and segmentation with `fp16=True` and
   `fp16=False`, and run every engine in the C++ TensorRT backend.
3. Record results here, update the fixture README, `docs/export.md` and `CHANGELOG.md`; merge to
   `develop`.
4. Create `release/v0.6.0` from `develop`; bump the four version statements; cut `CHANGELOG.md`
   with the migration note and Known issues; record v0.6.0 in the roadmap Status.
5. Verify: version sync, lint gate, default build and tests.
6. Merge to `master` `--no-ff`, tag `v0.6.0`, back-merge to `develop`, delete the branch. Push
   only after user approval.
