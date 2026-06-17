# ------------------------------------------------------------------------
# RF-DETR Keypoint Export
# Based on RF-DETR export module
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0
# ------------------------------------------------------------------------

"""
Export ONNX model for RF-DETR keypoint detection deployment

This script uses the RF-DETR library's built-in export functionality.
The RFDETRKeypointPreview class handles the export internally and outputs:
- Bounding boxes (dets)
- Class labels (labels)
- Keypoints (keypoints)
"""

import argparse
import shutil
from pathlib import Path

DEFAULT_KEYPOINT_RESOLUTION = 576
KEYPOINT_SHAPE_BLOCK_SIZE = 24


def upstream_onnx_filename() -> str:
    return "rfdetr-keypoint-preview.onnx"


def compat_onnx_filename() -> str:
    return "rfdetr-keypoint.onnx"


def resolve_exported_path(export_result, output_dir: Path) -> Path:
    if export_result is not None:
        exported_path = Path(export_result)
        if exported_path.exists():
            return exported_path

    expected_path = output_dir / upstream_onnx_filename()
    if expected_path.exists():
        return expected_path

    raise RuntimeError(
        "RF-DETR export completed but the ONNX file was not found. "
        f"Expected {expected_path}"
    )


def main():
    parser = argparse.ArgumentParser('RF-DETR Keypoint Export Script',
                                     description='Export RF-DETR keypoint detection model to ONNX format')

    # Export options that will be passed to the model's export() method
    parser.add_argument('--output_dir', default=None, type=str,
                        help='Path to save exported model (default: output)')
    parser.add_argument('--opset_version', default=17, type=int,
                        help='ONNX opset version (default: 17)')
    parser.add_argument('--batch_size', default=1, type=int,
                        help='Batch size for export (default: 1)')
    parser.add_argument('--input_size', default=None, type=int,
                        help='Input image size (default: model resolution, 576). '
                             'Must be divisible by patch_size * num_windows (24).')
    parser.add_argument('--device', default=None, type=str,
                        help='Device for export, e.g. cpu or cuda (default: auto)')

    args = parser.parse_args()

    if args.input_size is not None and args.input_size % KEYPOINT_SHAPE_BLOCK_SIZE != 0:
        raise ValueError(
            f"--input_size must be divisible by {KEYPOINT_SHAPE_BLOCK_SIZE} "
            "(patch_size=12 * num_windows=2)"
        )

    print("="*60)
    print("RF-DETR Keypoint Model Export")
    print("="*60)

    # Initialize the keypoint model
    print("\n[1/2] Loading RF-DETR Keypoint model...")
    from rfdetr import RFDETRKeypointPreview

    model_kwargs = {}
    if args.device:
        model_kwargs['device'] = args.device
    model = RFDETRKeypointPreview(**model_kwargs)

    input_size = args.input_size or getattr(model.model, 'resolution', DEFAULT_KEYPOINT_RESOLUTION)
    output_dir = Path(args.output_dir or "output")

    # Build export kwargs from arguments
    export_kwargs = {
        'output_dir': str(output_dir),
        'opset_version': args.opset_version,
        'batch_size': args.batch_size,
    }

    if args.input_size is not None:
        export_kwargs['shape'] = (args.input_size, args.input_size)

    # Export using the model's built-in export method
    print("\n[2/2] Exporting to ONNX format...")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Input size: {input_size}x{input_size}")
    print(f"  - ONNX opset: {args.opset_version}")

    exported_path = resolve_exported_path(model.export(**export_kwargs), output_dir)

    compat_path = output_dir / compat_onnx_filename()
    if exported_path.name != compat_onnx_filename() and exported_path != compat_path:
        shutil.copy2(exported_path, compat_path)

    print(f"\nExported ONNX file: {exported_path}")
    if compat_path.exists() and compat_path != exported_path:
        print(f"Compatibility copy: {compat_path}")

    print("\n" + "="*60)
    print("✓ Export complete!")
    print("="*60)
    print("\nModel outputs:")
    print("  - dets: Bounding boxes [batch, num_queries, 4]")
    print("  - labels: Class logits [batch, num_queries, num_keypoint_classes]")
    print("  - keypoints: Keypoints [batch, num_queries, C*K_max, 8]")
    print("\nNote:")
    print("  - RFDETRKeypointPreview is a Preview API — tensor layout may change in future releases.")
    print(f"  - Upstream export filename: {upstream_onnx_filename()}")
    print(f"  - Compatibility copy written as: {compat_onnx_filename()}")
    print("="*60)


if __name__ == '__main__':
    main()
