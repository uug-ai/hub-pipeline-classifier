from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import onnxruntime as ort
import torch
from ultralytics.models.yolo import YOLO

from scripts.export_triton_reid import export_repository


class ReIDExportTest(unittest.TestCase):
    def test_exports_dynamic_embeddings_without_overwriting_existing_model(self):
        torch.set_num_threads(1)
        with TemporaryDirectory() as directory:
            weights = Path(directory) / 'classification.pt'
            YOLO('yolo11n-cls.yaml').save(str(weights))
            path = export_repository(str(weights), Path(directory) / 'models', max_batch_size=2)
            session = ort.InferenceSession(str(path / '1' / 'model.onnx'), providers=['CPUExecutionProvider'])
            self.assertEqual(session.get_inputs()[0].name, 'images')
            self.assertEqual(session.get_inputs()[0].shape, ['batch', 3, 224, 224])
            self.assertEqual(session.get_outputs()[0].name, 'embeddings')
            self.assertEqual(len(session.get_outputs()[0].shape), 2)
            self.assertIn('kind: KIND_GPU', (path / 'config.pbtxt').read_text(encoding='utf-8'))
            with self.assertRaises(FileExistsError):
                export_repository(str(weights), Path(directory) / 'models')


if __name__ == '__main__':
    unittest.main()