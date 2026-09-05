import contextlib
import io
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "deploy"))

from deploy import export_common
from deploy import export_detection
from deploy import export_executorch
from deploy import export_keypoint
from deploy import export_segmentation


class FakeModel:
    extension = ".onnx"
    calls = []

    def __init__(self, **kwargs):
        self.model = types.SimpleNamespace(resolution=576)

    def export(self, **kwargs):
        type(self).calls.append(kwargs)
        output_dir = Path(kwargs.get("output_dir", "output"))
        output_dir.mkdir(parents=True, exist_ok=True)
        artifact = output_dir / f"{kwargs['output_name']}{self.extension}"
        artifact.write_bytes(b"artifact")
        return artifact


class FakePteModel(FakeModel):
    extension = ".pte"


class ExportScriptTests(unittest.TestCase):
    def setUp(self):
        FakeModel.calls = []
        FakePteModel.calls = []

    @staticmethod
    def fake_rfdetr():
        module = types.ModuleType("rfdetr")
        names = (
            "RFDETRNano", "RFDETRSmall", "RFDETRMedium", "RFDETRLarge",
            "RFDETRXLarge", "RFDETR2XLarge", "RFDETRSegNano", "RFDETRSegSmall",
            "RFDETRSegMedium", "RFDETRSegLarge", "RFDETRSegXLarge", "RFDETRSeg2XLarge",
            "RFDETRKeypointPreview",
        )
        for name in names:
            setattr(module, name, FakePteModel if name == "RFDETRNano" else FakeModel)
        return module

    def run_main(self, module, args):
        stdout = io.StringIO()
        with mock.patch.dict(sys.modules, {"rfdetr": self.fake_rfdetr()}), \
             mock.patch.object(sys, "argv", [module.__file__, *args]), \
             contextlib.redirect_stdout(stdout):
            module.main()
        return stdout.getvalue()

    def test_returned_path_must_exist(self):
        with self.assertRaisesRegex(RuntimeError, "does not exist"):
            export_common.resolve_exported_path("/definitely/missing/model.onnx", "ONNX")

    def test_none_return_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "without returning"):
            export_common.resolve_exported_path(None, "ONNX")

    def test_detection_default_name_and_returned_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = self.run_main(export_detection, ["--output_dir", tmp, "--model_type", "small"])
            self.assertEqual(FakeModel.calls[-1]["output_name"], "rfdetr-small")
            self.assertIn(str(Path(tmp) / "rfdetr-small.onnx"), output)

    def test_segmentation_custom_name_is_forwarded(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = self.run_main(export_segmentation, ["--output_dir", tmp, "--model_type", "small",
                                                                 "--output_name", "custom-seg"])
            self.assertEqual(FakeModel.calls[-1]["output_name"], "custom-seg")
            self.assertIn(str(Path(tmp) / "custom-seg.onnx"), output)

    def test_keypoint_default_name_and_compatibility_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = self.run_main(export_keypoint, ["--output_dir", tmp])
            self.assertEqual(FakeModel.calls[-1]["output_name"], "rfdetr-keypoint-preview")
            self.assertTrue((Path(tmp) / "rfdetr-keypoint.onnx").is_file())
            self.assertIn("Compatibility copy", output)

    def test_executorch_default_suppresses_upstream_suffix(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = self.run_main(export_executorch, ["--output_dir", tmp, "--model_type", "nano"])
            self.assertEqual(FakePteModel.calls[-1]["output_name"], "rfdetr-nano")
            self.assertIn(str(Path(tmp) / "rfdetr-nano.pte"), output)


if __name__ == "__main__":
    unittest.main()
