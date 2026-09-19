# Usage

Running the inference application: every run mode and every command-line flag.

> Part of the [RF-DETR C++ Inference](../README.md) documentation.
>
> This page is the operational reference — *what* each mode and flag does. For *why* and
> *when* — tuning strategy, custom class layouts, the GPU pipeline, embedding the library in
> your own code — see [advanced-usage.md](advanced-usage.md).

---

## Prepare Input Files

Three things, in the order the binary takes them:

1. **A model** — `.onnx` for ONNX Runtime, `.onnx`/`.engine`/`.trt` for TensorRT, `.pte` for
   ExecuTorch. Export one with [export.md](export.md).
2. **An input** — an image (e.g. `image.jpg`) or a video (e.g. `video.mp4`).
3. **A labels file** — one label per line; `data/coco-labels-91.txt` ships with the repo.

```bash
./build/inference_app <model> <input> <labels> [flags...]
```

---

## Run Inference

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

> [!NOTE]
> The official keypoint checkpoint is background-first and decodes with the default config. An
> active-first (`[17]`) export needs `--keypoint-counts 17 --background-class-id none`. See the
> keypoint note in [export.md](export.md#keypoint-model-export).

### Video Processing

```bash
./build/inference_app /path/to/model.onnx /path/to/video.mp4 /path/to/coco-labels-91.txt
```

With a live preview window (ESC quits early):

```bash
./build/inference_app /path/to/model.onnx /path/to/video.mp4 /path/to/coco-labels-91.txt --display
```

Video with segmentation:

```bash
./build/inference_app /path/to/model.onnx /path/to/video.mp4 /path/to/coco-labels-91.txt --segmentation
```

Supported containers: `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`, `.flv`, `.wmv`. Video files are
recognised by extension and processed with the multi-threaded pipeline; output is written to
`output_video.mp4`.

### Using a Pre-built TensorRT Engine

On a TensorRT build, a `.engine` or `.trt` file is loaded directly, skipping ONNX-to-TensorRT
conversion:

```bash
./build/inference_app /path/to/model.engine /path/to/image.jpg /path/to/coco-labels-91.txt --segmentation
```

---

## Command-Line Flags

The complete list. Every flag works with every mode.

### Mode

| Flag | Effect |
|------|--------|
| *(none)* | Object detection — the default |
| `--segmentation` | Instance segmentation; requires a model exported with masks |
| `--keypoint` | Keypoint detection; requires a keypoint export |
| `--display` | Open a live preview window while processing a video |

### Inference parameters

Overridable without recompiling.

| Flag | Default | Effect |
|------|---------|--------|
| `--threshold <val>` | `0.5` | Confidence threshold for keeping a detection; must be in `[0, 1]` |
| `--resolution <px>` | auto-detect | Model input resolution; omit to detect it from the model |
| `--max-detections <n>` | `300` | Top-k cap on the number of query/class pairs ranked before thresholding (upstream's `num_select`) |
| `--mask-threshold <val>` | `0.0` | Mask logit cutoff for binary mask generation (segmentation only); may be negative |
| `--background-class-id <n\|none>` | `0` | Exported logit slot holding background, excluded before ranking; negative counts from the end, `none` keeps every slot |
| `--keypoint-counts <n[,n...]>` | `0,17` | `num_keypoints_per_class`; use `17` for an active-first `[17]` keypoint export (pair with `--background-class-id none`) |
| `--output <path>` | `output_image.jpg` / `output_video.mp4` | Output path for the saved image or video |

```bash
./build/inference_app /path/to/model.onnx /path/to/image.jpg /path/to/coco-labels-91.txt \
  --threshold 0.7 --max-detections 100

./build/inference_app /path/to/model.onnx /path/to/image.jpg /path/to/coco-labels-91.txt \
  --segmentation --mask-threshold 0.5
```

`--resolution` is only useful for a model that accepts an input size other than the one recorded
in the model file — the auto-detected value is correct for a normally exported model.
`--background-class-id` must match your export's logit layout; getting it wrong shifts every
reported label by one. The layouts and what to pass for each are in
[Customizing Labels and Class Layouts](advanced-usage.md#customizing-labels-and-class-layouts).

### GPU pipeline

Available only on a TensorRT build compiled with the GPU pipeline, and **off by default** even
then.

| Flag | Requires | Effect |
|------|----------|--------|
| `--gpu-preprocess` | `-DUSE_DALI=ON` | Decode/resize/normalize on the GPU with DALI |
| `--gpu-postprocess` | `-DUSE_CUDA_POSTPROCESS=ON` | Segmentation mask decode/resize/threshold in CUDA kernels; segmentation only |
| `--dali-pipeline-dir <dir>` | `-DUSE_DALI=ON` | Where the serialized `.dali` pipeline files live (default `data/dali`) |

```bash
./build/inference_app /path/to/model.engine /path/to/image.jpg /path/to/coco-labels-91.txt \
  --segmentation --gpu-preprocess --gpu-postprocess
```

See [The GPU Pipeline at Runtime](advanced-usage.md#the-gpu-pipeline-at-runtime) for what each
half does, and [architecture.md](architecture.md#gpu-pipeline) for the design.

---

## Output

- The annotated image is saved as `output_image.jpg`, video as `output_video.mp4` — override
  either with `--output <path>`.
- Detection and segmentation results (bounding boxes, labels, scores, and mask pixels) are
  printed to the console.
- Input resolution is detected automatically from the model (432x432, 560x560, and so on).
- Segmentation mode draws colored masks with transparency overlays; keypoint mode draws the
  COCO skeleton.
- Results come back in descending-score order, and ties break deterministically — the same model
  and image always produce the same ordering. Why, and how the top-k selection interacts with
  `--threshold`: [How detections are selected](advanced-usage.md#how-detections-are-selected).

---

## Going Further

- [advanced-usage.md](advanced-usage.md) — tuning strategy, class layouts, the GPU pipeline,
  the `Config` reference, and embedding `RFDETRInference` in your own code
- [building.md](building.md) — build a different backend
- [export.md](export.md) — produce a model to run
