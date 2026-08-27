# Usage

Running the inference application: every mode, every command-line flag, and the `Config` fields behind them.

> Part of the [RF-DETR C++ Inference](../README.md) documentation.

---



## Prepare Input Files
- The RF-DETR model file (`.onnx` for ONNX Runtime, `.onnx`/`.engine`/`.trt` for TensorRT)
- An input image (e.g., `image.jpg`) or video file (e.g., `video.mp4`)
- A COCO labels file (e.g., `coco-labels-91.txt`)

## Run Inference
After building the project, run the inference application:

### Object Detection

```bash
./build/inference_app /path/to/model.onnx /path/to/image.jpg /path/to/coco-labels-91.txt
```

### Instance Segmentation

```bash
./build/inference_app /path/to/model.onnx /path/to/image.jpg /path/to/coco-labels-91.txt --segmentation
```

### Keypoint Detection

```bash
./build/inference_app /path/to/model.onnx /path/to/image.jpg /path/to/coco-labels-91.txt --keypoint
```

> [!WARNING]
> Keypoint models exported with `rfdetr` 1.8.2 or later use the active-first schema (`[17]`) and are
> not decodable by the default build, which still expects background-first `{0, 17}`. See the
> keypoint warning in [export.md](export.md#keypoint-model-export).

### Video Processing

```bash
./build/inference_app /path/to/model.onnx /path/to/video.mp4 /path/to/coco-labels-91.txt
```

With live preview window:

```bash
./build/inference_app /path/to/model.onnx /path/to/video.mp4 /path/to/coco-labels-91.txt --display
```

Video with segmentation:

```bash
./build/inference_app /path/to/model.onnx /path/to/video.mp4 /path/to/coco-labels-91.txt --segmentation
```

Supported video formats: `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`, `.flv`, `.wmv`. Output is written to `output_video.mp4`.

### Tuning Flags

The inference parameters can be overridden without recompiling:

| Flag | Default | Effect |
|------|---------|--------|
| `--threshold <val>` | `0.5` | Confidence threshold for keeping a detection; must be in `[0, 1]` |
| `--resolution <px>` | auto-detect | Model input resolution; omit to detect it from the model |
| `--max-detections <n>` | `300` | Top-k cap on the number of query/class pairs ranked before thresholding (upstream's `num_select`) |
| `--mask-threshold <val>` | `0.0` | Mask logit cutoff for binary mask generation (segmentation only); may be negative |
| `--background-class-id <n\|none>` | `0` | Exported logit slot holding background, excluded before ranking; negative counts from the end, `none` keeps every slot |

```bash
./build/inference_app /path/to/model.onnx /path/to/image.jpg /path/to/coco-labels-91.txt \
  --threshold 0.7 --max-detections 100

./build/inference_app /path/to/model.onnx /path/to/image.jpg /path/to/coco-labels-91.txt \
  --segmentation --mask-threshold 0.5
```

These flags work with all modes (`--segmentation`, `--keypoint`, video input, and `--display`). `--resolution`
is only useful for models that accept an input size other than the one recorded in the model file — the
auto-detected value is correct for a normally exported model.

`--background-class-id` mirrors the argument rfdetr 1.9.4 added to its own ONNX/TFLite decoders. The default
`0` matches the shipped RF-DETR exports, whose logit 0 is background and whose logit *n* is COCO category *n*
— which is exactly how `data/coco-labels-91.txt` is indexed. Change it only for a checkpoint with a different
class layout: `none` for one where every logit slot is a real class (a fine-tuned model with contiguous
0-based ids), or `-1` for one whose background sits in the final slot. Getting it wrong shifts every reported
label by one.

### How detections are selected

RF-DETR scores classes with independent sigmoids rather than a softmax, so one query can legitimately clear
the threshold on several classes at once. Postprocessing therefore ranks the flattened *(query, class)* grid
and keeps the top `--max-detections` pairs **before** applying `--threshold`, which is what
`PostProcess._select_topk` does upstream — a per-query argmax would silently drop every class but the
strongest (the bug rfdetr 1.9.3 fixed in its own exported-model decoders). Results come back in
descending-score order, with exact ties broken by ascending flattened query/class index, so a given model and
image always produce the same ordering. Detection, segmentation, keypoint, and the CUDA postprocess kernels
all share that rule.

### Using Pre-built TensorRT Engine

If you have a pre-built TensorRT engine file (`.engine` or `.trt`), use it directly:

```bash
./build/inference_app /path/to/model.engine /path/to/image.jpg /path/to/coco-labels-91.txt --segmentation
```

### GPU Pipeline Flags (TensorRT builds with the GPU pipeline compiled in)

```bash
# DALI GPU preprocessing + CUDA GPU segmentation postprocessing:
./build/inference_app /path/to/model.engine /path/to/image.jpg /path/to/coco-labels-91.txt \
  --segmentation --gpu-preprocess --gpu-postprocess
```

- `--gpu-preprocess` — decode/resize/normalize on the GPU with DALI (build with `-DUSE_DALI=ON`)
- `--gpu-postprocess` — segmentation mask decode/resize/threshold with CUDA kernels (build with `-DUSE_CUDA_POSTPROCESS=ON`); segmentation only, requires `--segmentation`
- `--dali-pipeline-dir <dir>` — where the serialized `.dali` pipeline files live (default: `data/dali`)

Both flags default off — the CPU paths remain the default even in a GPU-pipeline build. See [GPU Pipeline](architecture.md#gpu-pipeline).

**Features:**
- The output image is saved as `output_image.jpg`; video output is saved as `output_video.mp4`
- Detection/segmentation results (bounding boxes, labels, scores, and mask pixels) are printed to the console
- Input resolution is automatically detected from the model (supports 432x432, 560x560, etc.)
- Segmentation mode draws colored masks with transparency overlays
- Uses top-k selection (default: 300 detections) for efficient processing
- Video files are automatically detected by extension and processed with the multi-threaded pipeline

## Configuration
`Config` (`src/rfdetr_inference.hpp`) holds the inference settings. Most of them are reachable from the
command line — see [Tuning Flags](#tuning-flags) — and the rest require editing `src/main.cpp`:

| `Config` field | Default | CLI override |
|----------------|---------|--------------|
| `model_type` | `ModelType::DETECTION` | `--segmentation` / `--keypoint` |
| `threshold` | `0.5` | `--threshold <val>` |
| `resolution` | auto-detected from the model | `--resolution <px>` |
| `max_detections` | `300` (top-k selection) | `--max-detections <n>` |
| `mask_threshold` | `0.0` (binary mask generation) | `--mask-threshold <val>` |
| `background_class_id` | `0` (background-first exports) | `--background-class-id <n\|none>` |
| `gpu_preprocess` / `gpu_postprocess` | `false` | `--gpu-preprocess` / `--gpu-postprocess` |
| `dali_pipeline_dir` | `data/dali` | `--dali-pipeline-dir <dir>` |
| `gpu_device_id` | `0` | — (edit `src/main.cpp`) |
| `means` / `stds` | ImageNet `[0.485, 0.456, 0.406]` / `[0.229, 0.224, 0.225]` | — (edit `src/main.cpp`) |
| `keypoint_*`, `skeleton`, `draw_uncertainty` | COCO 17-keypoint layout | — (edit `src/main.cpp`) |

`src/main.cpp` leaves every field it does not override at its `Config` default, so changing a default in
`src/rfdetr_inference.hpp` is enough for the fields with no CLI flag.

## Example Custom Configuration
When embedding `RFDETRInference` rather than using the CLI:

```cpp
Config config;
config.resolution = 0;              // Auto-detect
config.threshold = 0.6f;            // Higher confidence threshold
config.max_detections = 100;        // Fewer detections
config.mask_threshold = 0.5f;       // More conservative masks
config.model_type = ModelType::SEGMENTATION;

RFDETRInference inference(model_path, label_path, config);
```
