import types
import unittest
from unittest import mock

import numpy as np
import torch
from ultralytics.models.yolo.detect.predict import DetectionPredictor

from utils.TritonDetectionPredictor import TritonDetectionPredictor


class TritonDetectionPredictorTest(unittest.TestCase):
    def setUp(self):
        self.predictor = object.__new__(TritonDetectionPredictor)
        self.predictor.args = types.SimpleNamespace(conf=0.5, classes=[0], max_det=300)
        self.predictor.batch = (['image.jpg'],)
        self.predictor.model = types.SimpleNamespace(names={0: 'person', 2: 'car'})
        self.image = torch.zeros((1, 3, 512, 512))
        self.orig_images = [np.zeros((512, 512, 3), dtype=np.uint8)]

    def test_filters_end_to_end_detections(self):
        predictions = torch.tensor([[
            [10, 20, 30, 40, 0.9, 0],
            [50, 60, 70, 80, 0.4, 0],
            [90, 100, 110, 120, 0.8, 2],
        ]])

        results = self.predictor.postprocess(predictions, self.image, self.orig_images)

        self.assertEqual(len(results), 1)
        self.assertEqual(len(results[0].boxes), 1)
        self.assertEqual(results[0].names[int(results[0].boxes[0].cls)], 'person')
        self.assertAlmostEqual(float(results[0].boxes[0].conf), 0.9)
        self.assertEqual(results[0].boxes[0].xyxy.tolist(), [[10.0, 20.0, 30.0, 40.0]])

    def test_uses_standard_postprocessing_for_raw_outputs(self):
        predictions = torch.zeros((1, 84, 100))

        with mock.patch.object(DetectionPredictor, 'postprocess', return_value=['raw']) as postprocess:
            results = self.predictor.postprocess(predictions, self.image, self.orig_images)

        self.assertEqual(results, ['raw'])
        postprocess.assert_called_once_with(predictions, self.image, self.orig_images)


if __name__ == '__main__':
    unittest.main()