"""Export-only workaround for LiteRT's rank-4 batch-matmul lowering limitation.

The pinned RF-DETR exporter prepares a private model copy. Flattening Linear's
leading dimensions before the affine operation is mathematically equivalent and
keeps that operation rank 2; its original output shape is restored immediately.
No trained parameter, live model, or installed package file is modified.
"""
from contextlib import contextmanager
import inspect


@contextmanager
def keypoint_linear_export():
    import torch
    from torch import nn
    from torch.nn import functional as functional
    from rfdetr.export._litert.exporter import LiteRTExporter

    class FlatLinear(nn.Linear):
        def forward(self, value):
            if value.ndim <= 2:
                return functional.linear(value, self.weight, self.bias)
            flat = value.reshape(-1, self.in_features)
            result = functional.linear(flat, self.weight, self.bias)
            return result.reshape(*value.shape[:-1], self.out_features)

    # These are private upstream hooks: refuse a changed interface rather than
    # silently exporting through a path that no longer applies the rewrite.
    hooks = {
        '_convert': ('self', 'graph'),
        '_prepare_module_for_tracing': ('self', 'graph'),
        '_import_converter': ('self',),
        '_prepare_output_dir': ('self',),
        '_resolve_export_name': ('self', 'backbone_only'),
        '_convert_and_save': ('self', 'litert_torch', 'wrapped_model', 'input_tensors', 'output_file'),
    }
    for name, parameters in hooks.items():
        method = getattr(LiteRTExporter, name, None)
        if not callable(method):
            raise RuntimeError(f'Incompatible RF-DETR LiteRT exporter hook: {name}')
        # Bind the arguments we pass before installing either replacement.
        try:
            signature = inspect.signature(method)
            if name == '_resolve_export_name':
                signature.bind(None, backbone_only=False)
            else:
                signature.bind(*([None] * len(parameters)))
        except (TypeError, ValueError) as error:
            raise RuntimeError(f'Incompatible RF-DETR LiteRT exporter hook: {name}') from error

    original_prepare = LiteRTExporter._prepare_module_for_tracing
    original_convert = LiteRTExporter._convert

    def prepare(exporter, graph):
        wrapped, sample = original_prepare(exporter, graph)
        with torch.inference_mode():
            expected = tuple(value.detach().clone() for value in wrapped(sample))
            for module in wrapped.modules():
                if type(module) is nn.Linear:
                    module.__class__ = FlatLinear
            actual = wrapped(sample)
            if len(actual) != len(expected):
                raise RuntimeError('LiteRT keypoint rewrite changed the output count')
            for reference, result in zip(expected, actual, strict=True):
                torch.testing.assert_close(result, reference, atol=1e-5, rtol=1e-5)
        print('LiteRT keypoint export: Linear reshape parity passed (atol=rtol=1e-5)')
        return wrapped, sample

    def convert(exporter, graph):
        if 'keypoints' not in graph.output_names:
            return original_convert(exporter, graph)
        # RF-DETR rejects keypoints before preparation. Repeat only its conversion
        # orchestration, retaining the real output names and metadata for the
        # base exporter's sidecar writer. The upstream save hook still installs
        # pixel_row_sampling and handles converter errors.
        converter = exporter._import_converter()
        output_dir = exporter._prepare_output_dir()
        output_file = output_dir / f"{exporter._resolve_export_name(backbone_only=graph.backbone_only)}.tflite"
        wrapped, sample = exporter._prepare_module_for_tracing(graph)
        exporter._convert_and_save(converter, wrapped, sample, output_file)
        return output_file

    LiteRTExporter._prepare_module_for_tracing = prepare
    LiteRTExporter._convert = convert
    try:
        yield
    finally:
        LiteRTExporter._prepare_module_for_tracing = original_prepare
        LiteRTExporter._convert = original_convert
