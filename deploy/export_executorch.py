import argparse


def expected_pte_filename(model_type: str) -> str:
    return f"rfdetr-{model_type}.pte"


def main():
    parser = argparse.ArgumentParser('RF-DETR ExecuTorch Export Script',
                                     description='Export RF-DETR detection model to ExecuTorch (.pte) format')

    # Export options that will be passed to the model's export() method
    parser.add_argument('--output_dir', default=None, type=str,
                        help='Path to save exported model (default: output)')
    parser.add_argument('--batch_size', default=1, type=int,
                        help='Batch size for export (default: 1)')
    parser.add_argument('--input_size', default=640, type=int,
                        help='Input image size (default: 640)')
    parser.add_argument('--model_type', default='medium', type=str,
                        choices=['nano', 'small', 'medium', 'large', 'xlarge', '2xlarge'],
                        help='Model type (default: medium)')
    parser.add_argument('--backend', default='xnnpack', type=str,
                        choices=['xnnpack', 'coreml', 'qnn'],
                        help='ExecuTorch backend: xnnpack (CPU, fp32), coreml (Apple, fp16), '
                             'qnn (Qualcomm HTP, fp16) (default: xnnpack)')
    parser.add_argument('--soc', default=None, type=str,
                        help='Target SoC for the qnn backend, e.g. SM8650 (required when --backend qnn)')
    parser.add_argument('--device', default=None, type=str,
                        help='Device for export, e.g. cpu or cuda (default: auto)')
    args = parser.parse_args()

    if args.backend == 'qnn' and not args.soc:
        parser.error("--soc is required when --backend qnn (e.g. --soc SM8650)")

    # Warn before the export rather than only after it: lowering is slow, and this project's C++
    # backend links only the xnnpack or portable delegate, so a coreml/qnn .pte cannot run here.
    if args.backend != 'xnnpack':
        print(f"WARNING: --backend {args.backend} produces a .pte this project's C++ ExecuTorch")
        print(f"         backend cannot execute ({args.backend} delegate is never linked; only")
        print("         -DEXECUTORCH_DELEGATE=xnnpack|portable exist). Use --backend xnnpack")
        print("         unless you are targeting a different ExecuTorch runtime.\n")

    print("="*60)
    print("RF-DETR ExecuTorch Model Export")
    print("="*60)

    # Initialize the detection model
    print(f"\n[1/2] Loading RF-DETR Detection model ({args.model_type})...")
    model = None
    model_kwargs = {}
    if args.device:
        model_kwargs['device'] = args.device

    if args.model_type == 'nano':
        from rfdetr import RFDETRNano
        model = RFDETRNano(**model_kwargs)
    elif args.model_type == 'small':
        from rfdetr import RFDETRSmall
        model = RFDETRSmall(**model_kwargs)
    elif args.model_type == 'medium':
        from rfdetr import RFDETRMedium
        model = RFDETRMedium(**model_kwargs)
    elif args.model_type == 'large':
        from rfdetr import RFDETRLarge
        model = RFDETRLarge(**model_kwargs)
    elif args.model_type == 'xlarge':
        from rfdetr import RFDETRXLarge
        model = RFDETRXLarge(**model_kwargs)
    elif args.model_type == '2xlarge':
        from rfdetr import RFDETR2XLarge
        model = RFDETR2XLarge(**model_kwargs)
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")

    # Build export kwargs from arguments
    export_kwargs = {
        'format': 'executorch',
        'backend': args.backend,
        'batch_size': args.batch_size,
    }

    # Add output_dir if specified
    if args.output_dir:
        export_kwargs['output_dir'] = args.output_dir

    if args.input_size is not None:
        export_kwargs['shape'] = (args.input_size, args.input_size)

    if args.soc:
        export_kwargs['soc'] = args.soc

    # Export using the model's built-in export method
    print("\n[2/2] Exporting to ExecuTorch format...")
    print(f"  - Backend:    {args.backend}")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Input size: {args.input_size}x{args.input_size}")
    if args.soc:
        print(f"  - Target SoC: {args.soc}")

    model.export(**export_kwargs)

    output_dir = args.output_dir or "output"
    print(f"\nExpected ExecuTorch file: {output_dir}/{expected_pte_filename(args.model_type)}")

    print("\n" + "="*60)
    print("✓ Export complete!")
    print("="*60)
    print("\nModel outputs:")
    print("  - dets: Bounding boxes [batch, num_queries, 4]")
    print("  - labels: Class logits [batch, num_queries, num_classes]")
    print("\nNote: ExecuTorch export requires 'pip install rfdetr[executorch]==1.9.1'.")
    print("      The extra only constrains ExecuTorch to >=1.3,<2.0, so check what it")
    print("      installed ('pip show executorch'): the .pte must be exported with the")
    print("      same version as the C++ runtime, which this project pins to v1.4.0.")
    print("      1.9.1 recombines undelegated addmm into aten.linear (~2.5x faster XNNPACK")
    print("      inference); re-export .pte files produced with 1.9.0 to pick it up.")
    if args.backend == 'xnnpack':
        print("      Build the C++ side with -DUSE_ONNX_RUNTIME=OFF -DUSE_EXECUTORCH=ON")
        print("      -DEXECUTORCH_DELEGATE=xnnpack (the default).")
    else:
        # The delegate is baked into the .pte and must be linked into the runtime to self-register.
        # This project's CMake only offers xnnpack and portable, so a coreml/qnn program has no
        # delegate to bind to here and fails at run time with "Backend ... is not registered".
        print(f"      WARNING: this project's C++ backend CANNOT run a '{args.backend}' .pte.")
        print("      -DEXECUTORCH_DELEGATE accepts only xnnpack or portable, so the")
        print(f"      {args.backend} delegate is never linked and loading fails at run time with")
        print(f"      \"Backend ...{args.backend} is not registered\". Re-export with")
        print("      --backend xnnpack to run it here; keep this file for another ExecuTorch runtime.")
    print("="*60)


if __name__ == '__main__':
    main()
