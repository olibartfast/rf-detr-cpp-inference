#!/usr/bin/env python3
"""Serialize the RF-DETR DALI preprocessing pipelines.

Two variants, matching rfdetr::gpu::DaliPreprocessor::Source:

  encoded  external_source IMAGE (cpu, uint8, ndim=1) -> GPU decode -> resize -> normalize
  frame    external_source FRAME (gpu, uint8, HWC)    -> BGR->RGB   -> resize -> normalize

Both reproduce rfdetr::media::preprocess_bgr_image (src/media.cpp). Read the
notes below before changing anything here: three of the four steps differ from
the YOLO pipelines this was adapted from, and each difference is silent.

  1. NO LETTERBOX. RF-DETR stretches to resolution x resolution with independent
     x and y scale factors. There is no fn.paste, no aspect-preserving resize and
     no padding, because the C++ box decode scales by orig/res per axis with no
     gain or pad term. Adding letterboxing here shifts every box.

  2. antialias=False. DALI antialiases when downscaling by default; the C++
     reference is a plain 4-tap bilinear sample at (dst + 0.5) * scale - 0.5.
     With antialiasing on, tensors diverge on every image larger than the model
     resolution, worst on the largest ones.

  3. ImageNet normalization, not /255 alone. The C++ path divides by 255 and then
     applies mean/std, so DALI's single-step crop_mirror_normalize needs
     mean * 255 and std * 255.

  4. RGB. The encoded variant decodes directly to RGB, so the BGR->RGB swap the
     C++ path performs on interleaved bytes disappears. The frame variant is fed
     BGR by the video reader and converts explicitly.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from nvidia.dali import fn, pipeline_def, types

# Defaults mirror the Config struct in src/rfdetr_inference.hpp.
DEFAULT_MEAN = (0.485, 0.456, 0.406)
DEFAULT_STD = (0.229, 0.224, 0.225)


def _normalize(image, resolution: int, mean, std):
    """Shared tail: stretch to a square then scale/offset into CHW float."""
    resized = fn.resize(
        image,
        device="gpu",
        resize_x=resolution,
        resize_y=resolution,
        interp_type=types.INTERP_LINEAR,
        antialias=False,
    )
    return fn.crop_mirror_normalize(
        resized,
        device="gpu",
        dtype=types.FLOAT,
        output_layout="CHW",
        mean=[m * 255.0 for m in mean],
        std=[s * 255.0 for s in std],
    )


@pipeline_def
def encoded_pipeline(resolution: int, mean, std):
    encoded = fn.external_source(name="IMAGE", device="cpu", ndim=1, dtype=types.UINT8)
    decoded = fn.decoders.image(encoded, device="mixed", output_type=types.RGB)
    return _normalize(decoded, resolution, mean, std)


@pipeline_def
def frame_pipeline(resolution: int, mean, std):
    frame = fn.external_source(name="FRAME", device="gpu", ndim=3, dtype=types.UINT8, layout="HWC")
    rgb = fn.color_space_conversion(frame, device="gpu", image_type=types.BGR, output_type=types.RGB)
    return _normalize(rgb, resolution, mean, std)


def _triplet(text: str) -> tuple[float, float, float]:
    parts = [float(value) for value in text.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("expected three comma-separated values")
    return parts[0], parts[1], parts[2]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--resolution", type=int, required=True, help="Model input resolution (square)")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for the .dali files")
    parser.add_argument("--mean", type=_triplet, default=DEFAULT_MEAN, help="Comma-separated RGB mean in [0, 1]")
    parser.add_argument("--std", type=_triplet, default=DEFAULT_STD, help="Comma-separated RGB std in [0, 1]")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--num-threads", type=int, default=2)
    parser.add_argument(
        "--variant",
        choices=("encoded", "frame", "both"),
        default="both",
        help="Which pipeline(s) to serialize",
    )
    args = parser.parse_args()

    if args.resolution <= 0:
        parser.error("--resolution must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # prefetch_queue_depth=1: batch size is 1 and the C++ side feeds one sample
    # then immediately waits for it, so deeper prefetching only adds latency.
    common = dict(
        batch_size=1,
        num_threads=args.num_threads,
        device_id=args.device_id,
        prefetch_queue_depth=1,
    )
    builders = {
        "encoded": encoded_pipeline,
        "frame": frame_pipeline,
    }
    variants = builders.keys() if args.variant == "both" else (args.variant,)

    for name in variants:
        output = args.output_dir / f"preprocess_{name}_{args.resolution}.dali"
        pipeline = builders[name](resolution=args.resolution, mean=args.mean, std=args.std, **common)
        pipeline.serialize(filename=str(output))
        print(output)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
