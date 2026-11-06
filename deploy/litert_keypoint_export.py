"""Export-only workaround for LiteRT's rank-4 batch-matmul lowering limitation.

The pinned RF-DETR exporter prepares a private model copy. Flattening Linear's
leading dimensions before the affine operation is mathematically equivalent and
keeps that operation rank 2; its original output shape is restored immediately.
No trained parameter, live model, or installed package file is modified.
"""
from contextlib import contextmanager


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

    original_prepare = LiteRTExporter._prepare_module_for_tracing

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

    LiteRTExporter._prepare_module_for_tracing = prepare
    try:
        yield
    finally:
        LiteRTExporter._prepare_module_for_tracing = original_prepare
