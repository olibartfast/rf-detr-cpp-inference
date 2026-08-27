# Architecture

How the pieces fit together: the GPU pipeline, the multi-threaded video ring buffer, model output shapes, and the processing stages.

> Part of the [RF-DETR C++ Inference](../README.md) documentation.

---

## GPU Pipeline
An opt-in pipeline that moves preprocessing and segmentation postprocessing onto
the GPU, on the same CUDA stream as the TensorRT execution context — no Triton
server involved. Design constraints: [specs/gpu-pipeline.md](../specs/gpu-pipeline.md) — remaining
phases: [specs/roadmap.md](../specs/roadmap.md).

```
image bytes ──> DALI "encoded" pipeline ──┐
                (nvJPEG decode, resize,   │
                 normalize — on GPU)      ├──> device float[1,3,res,res]
video frame ──> DALI "frame" pipeline ────┘         │
                (BGR→RGB, resize, normalize)        v
                                          TensorRT enqueue (same stream)
                                                    │
                          ┌─────────────────────────┴───────────────┐
                          v                                         v
                detection / keypoint                          segmentation
                D2H outputs, existing                   CUDA postprocess on stream:
                CPU postprocess                         sigmoid → top-k → box decode
                                                        → mask resize+threshold,
                                                        one packed D2H at the end
```

## What each half does
- **DALI preprocessing** (`-DUSE_DALI=ON`, `--gpu-preprocess`): image decode
  (nvJPEG), resize, and ImageNet normalization run on the GPU and write straight
  into the TensorRT input binding. For still images only the compressed bytes
  cross to the GPU; for video the preprocess pipeline stage becomes a passthrough
  and DALI runs on the backend's stream inside the inference stage. Works with
  detection, segmentation, and keypoint models.
- **CUDA segmentation postprocessing** (`-DUSE_CUDA_POSTPROCESS=ON`,
  `--gpu-postprocess`): score sigmoid, global top-k, box decode, and the
  per-instance mask resize + threshold (the CPU path's dominant cost) run as CUDA
  kernels reading the TensorRT output bindings in place, with a single packed
  device-to-host transfer of the final results. Segmentation only.

Both halves are compile-time optional and runtime opt-in: a GPU-pipeline build
still defaults to the CPU paths, and the ONNX Runtime build is unaffected. The
backend interface gained optional device-side I/O entry points
(`supports_device_io()`, `run_inference_device()`, device pointers, stream) that
only the TensorRT backend implements.

## DALI pipeline files
`--gpu-preprocess` loads serialized DALI pipelines named
`preprocess_encoded_<resolution>.dali` (still images) and
`preprocess_frame_<resolution>.dali` (video frames) from `--dali-pipeline-dir`
(default `data/dali`), where `<resolution>` is the model input resolution
auto-detected from the engine. Pre-generated pipelines for resolutions **432**
and **576** are checked in under [data/dali](../data/dali).

For other resolutions, regenerate them (runs inside the pinned Triton container,
needs `--gpus all`; no local DALI pip install required):

```bash
./scripts/generate_dali_pipelines.sh 560           # -> data/dali/
```

The generator itself is [deploy/dali/generate_preprocess_pipeline.py](../deploy/dali/generate_preprocess_pipeline.py)
(uses the `nvidia-dali` Python package inside the container). The C++ side
validates the produced tensor size against the TensorRT input binding, so a
stale pipeline file fails loudly rather than silently degrading results.

## Version pinning
DALI libraries and pipeline serialization both come from the same pinned
container, `nvcr.io/nvidia/tritonserver:25.12-py3` (override with
`TRITON_IMAGE=...` on both scripts), keeping the DALI/CUDA/TensorRT triple
consistent with the TensorRT 10.13.3.9 / CUDA 13.x pin above.

## Video Processing
Video files are processed using a **four-stage ring buffer pipeline** that maximizes throughput with zero frame copies between stages:

```
                   free_slots (recycled)
                 +-------------------------+
                 |                         |
                 v                         |
 +--------+ idx  +-----------+ idx  +------++ idx  +------+
 | Decode | ---> | Preprocess| ---> | Infer| ----> | Draw |
 +--------+      +-----------+      +------+       +------+
  media           resize+norm        run model      annotate +
  decode into     into slot.tensor   postprocess    media encode
  slot.raw_frame  (pre-allocated)    into slot.*    + optional
                                                    preview
```

The default media/display backend uses FFmpeg for video decode/encode, SDL2 for
preview, and stb for image I/O. `-DUSE_OPENCV=ON` swaps those pieces for OpenCV
`videoio`, `highgui`, and `imgcodecs`.

- **4 `std::jthread`s** run concurrently, one per stage
- **Pre-allocated `FrameSlot`s** are reused via a ring buffer (default size: 8)
- Stages pass slot indices (not frames) through **bounded queues** with backpressure
- The inference stage owns its own `RFDETRInference` instance — no locks on the hot path
- Graceful shutdown via poison pill (`SIZE_MAX`) propagated through all queues
- Frame ordering is preserved (all stages are single-threaded FIFO)
- With `--gpu-preprocess`, the preprocess stage becomes a passthrough: DALI's `frame` pipeline runs on the backend's CUDA stream inside the inference stage, and the CPU cost of the bilinear resample disappears from the pipeline entirely

Use `--display` to open a live preview window (press ESC to quit early).

## Technical Details


## Model Outputs
### Detection Model
- **dets**: `float32[batch, num_queries, 4]` - Bounding boxes in `cxcywh` format (normalized)
- **labels**: `float32[batch, num_queries, num_classes]` - Class logits

### Segmentation Model
- **dets**: `float32[batch, num_queries, 4]` - Bounding boxes in `cxcywh` format (normalized)
- **labels**: `float32[batch, num_queries, num_classes]` - Class logits
- **masks**: `float32[batch, num_queries, mask_h, mask_w]` - Segmentation masks (e.g., 108x108)

### Keypoint Model
- **dets**: `float32[batch, num_queries, 4]` - Bounding boxes in `cxcywh` format (normalized)
- **labels**: `float32[batch, num_queries, num_classes+1]` - Class logits (index 0 = background)
- **keypoints**: `float32[batch, num_queries, C*K_max, 8]` - Keypoints (8 channels per keypoint)

## C++ Result Types
Postprocessing APIs expose decoded boxes as `std::vector<BoundingBox>`, with `x_min`, `y_min`, `x_max`, and `y_max` fields in pixel-space `xyxy` format. Segmentation masks use `std::vector<rfdetr::media::Mask>`, and keypoints use `std::vector<std::vector<KeypointResult>>` for per-detection keypoint metadata.

## Processing Pipeline
1. **Preprocessing**:
   - Resize image to model input resolution (auto-detected)
   - Convert BGR to RGB
   - Normalize with ImageNet statistics
   - Convert to CHW format

2. **Inference**:
   - Run ONNX Runtime session
   - Auto-detect output tensor names from model

3. **Postprocessing**:
   - **Detection**: Select predictions above confidence threshold
   - **Segmentation**: 
     - Apply sigmoid to class logits
     - Top-k selection across all classes and queries
     - Resize masks to original image dimensions using bilinear interpolation
     - Apply threshold to create binary masks
   - Convert bounding boxes from `cxcywh` to `xyxy` format
   - Scale coordinates to original image size

4. **Visualization**:
   - Draw bounding boxes with class labels
   - Overlay segmentation masks with transparency (alpha = 0.5)
   - Use deterministic colors based on class IDs
