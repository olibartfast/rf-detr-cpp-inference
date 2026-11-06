"""Export fixed-batch float32 RF-DETR detection, segmentation and pose directly to LiteRT."""
import argparse
from contextlib import nullcontext

try:
    from .export_common import resolve_exported_path
except ImportError:
    from export_common import resolve_exported_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model_type', choices=['nano', 'small', 'medium', 'large', 'xlarge', '2xlarge'], default='nano')
    tasks = parser.add_mutually_exclusive_group()
    tasks.add_argument('--segmentation', action='store_true')
    tasks.add_argument('--keypoint', action='store_true', help='Export RFDETRKeypointPreview with the Linear reshape adapter')
    parser.add_argument('--input_size', type=int, default=None,
                        help='Square resolution; default is the model native resolution')
    parser.add_argument('--output_dir', default='output')
    parser.add_argument('--output_name', default=None)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    if args.input_size is not None and args.input_size <= 0:
        parser.error('--input_size must be positive')

    from rfdetr import __dict__ as rfdetr_namespace
    family = 'RFDETRSeg' if args.segmentation else 'RFDETR'
    suffixes = {'nano': 'Nano', 'small': 'Small', 'medium': 'Medium',
                'large': 'Large', 'xlarge': 'XLarge', '2xlarge': '2XLarge'}
    model_class = 'RFDETRKeypointPreview' if args.keypoint else family + suffixes[args.model_type]
    model = rfdetr_namespace[model_class](device=args.device)
    export_kwargs = dict(format='litert', batch_size=1, output_dir=args.output_dir)
    if args.output_name:
        export_kwargs['output_name'] = args.output_name
    if args.input_size is not None:
        export_kwargs['shape'] = (args.input_size, args.input_size)
    if args.keypoint:
        try:
            from .litert_keypoint_export import keypoint_linear_export
        except ImportError:
            from litert_keypoint_export import keypoint_linear_export
        context = keypoint_linear_export()
    else:
        context = nullcontext()
    with context:
        path = resolve_exported_path(model.export(**export_kwargs), 'LiteRT')
    print(f'Exported LiteRT file: {path}')
    print('Input: NCHW float32, batch 1, ImageNet normalization')
    print('Outputs: normalized cxcywh boxes, class logits' +
          (', keypoint parameters [1,Q,K,8]' if args.keypoint else
           ', NQHW mask logits' if args.segmentation else ''))


if __name__ == '__main__':
    main()
