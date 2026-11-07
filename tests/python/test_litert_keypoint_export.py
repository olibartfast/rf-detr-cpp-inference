"""Export adapter checks; run in the rfdetr[litert] export environment."""
import unittest
from pathlib import Path
from types import SimpleNamespace
from contextlib import ExitStack
from unittest.mock import patch

import torch
from torch import nn
from rfdetr.export._litert.exporter import LiteRTExporter, LiteRTConfig

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
        original_convert = LiteRTExporter._convert
        with self.assertRaisesRegex(RuntimeError, 'conversion failed'):
            with keypoint_linear_export():
                self.assertIsNot(LiteRTExporter._prepare_module_for_tracing, original)
                self.assertIsNot(LiteRTExporter._convert, original_convert)
                raise RuntimeError('conversion failed')
        self.assertIs(LiteRTExporter._prepare_module_for_tracing, original)
        self.assertIs(LiteRTExporter._convert, original_convert)

    def test_keypoints_export_retains_graph_metadata_and_upstream_save(self):
        head = TinyHead().eval()
        sample = torch.randn(1, 2, 3, 8)
        metadata = SimpleNamespace(format='litert')
        names = ('dets', 'labels', 'keypoints')
        graph = SimpleNamespace(model=head, input_tensors=sample, output_names=names,
                                backbone_only=False, metadata=metadata)
        exporter = LiteRTExporter(LiteRTConfig(output_name='pose'))
        original_convert = LiteRTExporter._convert
        # The unadapted upstream path rejects this same graph before preparation.
        with self.assertRaisesRegex(NotImplementedError, 'keypoint'):
            original_convert(exporter, graph)
        with ExitStack() as patches:
            patches.enter_context(patch.object(LiteRTExporter, 'check_dependencies'))
            patches.enter_context(patch('rfdetr.export.base._switch_to_export_mode'))
            patches.enter_context(patch.object(LiteRTExporter, '_import_converter', return_value=object()))
            patches.enter_context(patch.object(LiteRTExporter, '_prepare_output_dir', return_value=Path('output')))
            patches.enter_context(patch.object(LiteRTExporter, '_prepare_module_for_tracing',
                                               return_value=(head, sample)))
            save = patches.enter_context(patch.object(LiteRTExporter, '_convert_and_save'))
            derive = patches.enter_context(patch.object(LiteRTExporter, '_metadata_for_artifact',
                                                        side_effect=lambda value, path: value))
            write = patches.enter_context(patch('rfdetr.export.base.write_metadata'))
            with keypoint_linear_export():
                result = exporter(graph)
            self.assertEqual(result, Path('output/pose.tflite'))
            save.assert_called_once()
            self.assertIs(save.call_args.args[1], head)
            derive.assert_called_once_with(metadata, result)
            write.assert_called_once_with(result, metadata)
        self.assertIs(graph.metadata, metadata)
        self.assertEqual(graph.output_names, names)
        self.assertIs(LiteRTExporter._convert, original_convert)

    def test_rejects_incompatible_private_hook_before_mutation(self):
        original = LiteRTExporter._prepare_module_for_tracing
        with patch.object(LiteRTExporter, '_convert_and_save', None):
            with self.assertRaisesRegex(RuntimeError, '_convert_and_save'):
                with keypoint_linear_export():
                    self.fail('incompatible exporter entered the adapter')
        self.assertIs(LiteRTExporter._prepare_module_for_tracing, original)


if __name__ == '__main__':
    unittest.main()
