"""Export adapter checks; run in the rfdetr[litert] export environment."""
import unittest
from unittest.mock import patch

import torch
from torch import nn
from rfdetr.export._litert.exporter import LiteRTExporter

from deploy.litert_keypoint_export import keypoint_linear_export


class TinyHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 5)

    def forward(self, value):
        return (self.linear(value),)


class KeypointLinearExportTest(unittest.TestCase):
    def test_preserves_rank_four_outputs_and_parameters(self):
        head = TinyHead().eval()
        sample = torch.randn(1, 2, 3, 8)
        expected = head(sample)[0].detach().clone()
        weight = head.linear.weight
        bias = head.linear.bias
        with patch.object(LiteRTExporter, '_prepare_module_for_tracing', return_value=(head, sample)) as original:
            with keypoint_linear_export():
                rewritten, actual_sample = LiteRTExporter._prepare_module_for_tracing(None, None)
                actual = rewritten(actual_sample)[0]
            self.assertIs(LiteRTExporter._prepare_module_for_tracing, original)
        self.assertEqual(actual.shape, expected.shape)
        self.assertIs(head.linear.weight, weight)
        self.assertIs(head.linear.bias, bias)
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

    def test_restores_exporter_hook_when_export_raises(self):
        original = LiteRTExporter._prepare_module_for_tracing
        with self.assertRaisesRegex(RuntimeError, 'conversion failed'):
            with keypoint_linear_export():
                self.assertIsNot(LiteRTExporter._prepare_module_for_tracing, original)
                raise RuntimeError('conversion failed')
        self.assertIs(LiteRTExporter._prepare_module_for_tracing, original)


if __name__ == '__main__':
    unittest.main()
