import argparse

try:
    from .export_common import resolve_exported_path
except ImportError:  # Running the file directly from the deploy directory.
    from export_common import resolve_exported_path


def default_output_name(model_type: str) -> str:
    return f"rfdetr-{model_type}"


def main():
    parser = argparse.ArgumentParser(
        'RF-DETR Detection Export Script',
        description='Export RF-DETR detection model to ONNX format')

    parser.add_argument('--output_dir', default=None, type=str,
                        help='Path to save exported model (default: output)')
    parser.add_argument('--output_name', default=None, type=str,
                        help='Output filename stem without extension (default: rfdetr-{model_type})')
    parser.add_argument('--opset_version', default=17, type=int,
                        help='ONNX opset version (default: 17)')
    parser.add_argument('--batch_size', default=1, type=int,
                        help='Batch size for export (default: 1)')
    parser.add_argument('--input_size', default=640, type=int,
                        help='Input image size (default: 640)')
    parser.add_argument('--model_type', default='medium', type=str,
                        choices=['nano', 'small', 'medium', 'large', 'xlarge', '2xlarge'],
                        help='Model type (default: medium)')
    parser.add_argument('--device', default=None, type=str,
                        help='Device for export, e.g. cpu or cuda (default: auto)')
    args = parser.parse_args()

    print("=" * 60)
    print("RF-DETR Detection Model Export")
    print("=" * 60)
    print(f"\n[1/2] Loading RF-DETR Detection model ({args.model_type})...")

    model_kwargs = {'device': args.device} if args.device else {}
    model_classes = {
        'nano': 'RFDETRNano', 'small': 'RFDETRSmall', 'medium': 'RFDETRMedium',
        'large': 'RFDETRLarge', 'xlarge': 'RFDETRXLarge', '2xlarge': 'RFDETR2XLarge',
    }
    from rfdetr import __dict__ as rfdetr_namespace
    model = rfdetr_namespace[model_classes[args.model_type]](**model_kwargs)

    export_kwargs = {
        'opset_version': args.opset_version,
        'batch_size': args.batch_size,
        'output_name': args.output_name or default_output_name(args.model_type),
    }
    if args.output_dir:
        export_kwargs['output_dir'] = args.output_dir
    if args.input_size is not None:
        export_kwargs['shape'] = (args.input_size, args.input_size)

    print("\n[2/2] Exporting to ONNX format...")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Input size: {args.input_size}x{args.input_size}")
    print(f"  - ONNX opset: {args.opset_version}")
    exported_path = resolve_exported_path(model.export(**export_kwargs), "ONNX")
    print(f"\nExported ONNX file: {exported_path}")

    print("\n" + "=" * 60)
    print("✓ Export complete!")
    print("=" * 60)
    print("\nModel outputs:")
    print("  - dets: Bounding boxes [batch, num_queries, 4]")
    print("  - labels: Class logits [batch, num_queries, num_classes]")
    print("\nNote: This is a detection model.")
    print("=" * 60)


if __name__ == '__main__':
    main()
