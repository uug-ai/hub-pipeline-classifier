from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from utils.TritonReIDEncoder import TritonReIDEncoder
from utils.TritonReIDTracking import track_with_triton_reid
from utils.TrackingConfig import tracking_config
from utils.TritonDetectionPredictor import TritonDetectionPredictor
from test_tracking_config import appearance_settings
from ultralytics.models.yolo import YOLO
import torch


class TritonReIDEncoderTest(unittest.TestCase):
    def setUp(self):
        self.image = np.zeros((64, 64, 3), dtype=np.uint8)
        self.image[:, :] = [0, 0, 255]
        self.boxes = np.array([[20, 20, 30, 30, 0]] * 3, dtype=np.float32)
        self.client = mock.Mock()

    def encoder(self, **kwargs):
        return TritonReIDEncoder('http://triton:8000/yolo11n_reid', client=self.client, **kwargs)

    def test_batches_preserve_order_and_normalize_embeddings(self):
        self.client.infer.side_effect = [
            SimpleNamespace(as_numpy=lambda name: np.array([[3, 4], [0, 2]], dtype=np.float32)),
            SimpleNamespace(as_numpy=lambda name: np.array([[2, 0]], dtype=np.float32)),
        ]
        result = self.encoder(batch_size=2)(self.image, self.boxes)
        np.testing.assert_allclose(result, [[0.6, 0.8], [0, 1], [1, 0]])
        self.assertEqual(self.client.infer.call_count, 2)
        inputs = self.client.infer.call_args_list[0].kwargs['inputs']
        self.assertEqual(tuple(inputs[0].shape()), (2, 3, 224, 224))
        self.assertEqual(inputs[0].datatype(), 'FP32')
        self.assertEqual(inputs[0].name(), 'images')

    def test_crop_is_rgb_chw_scaled_and_clipped(self):
        encoder = self.encoder()
        self.client.infer.return_value.as_numpy.return_value = np.ones((1, 2), dtype=np.float32)
        with mock.patch.object(encoder.http, 'InferInput') as input_type:
            encoder(self.image, np.array([[0, 0, 20, 20, 0]], dtype=np.float32))
        result = input_type.return_value.set_data_from_numpy.call_args.args[0][0]
        self.assertEqual(result.shape, (3, 224, 224))
        self.assertEqual(result.dtype, np.float32)
        np.testing.assert_array_equal(result[0], 1)
        np.testing.assert_array_equal(result[1:], 0)

    def test_empty_detections_do_not_make_requests(self):
        self.assertEqual(self.encoder()(self.image, []), [])
        self.client.infer.assert_not_called()

    def test_request_statistics_reset_between_messages(self):
        encoder = self.encoder()
        self.client.infer.return_value.as_numpy.return_value = np.ones((3, 2), dtype=np.float32)
        encoder(self.image, self.boxes)
        self.assertEqual(encoder.request_count, 1)
        self.assertGreaterEqual(encoder.request_seconds, 0)
        encoder.reset_stats()
        self.assertEqual(encoder.request_count, 0)
        self.assertEqual(encoder.request_seconds, 0)

    def test_invalid_remote_outputs_fail(self):
        for features in (None, np.zeros((3, 2)), np.ones((2, 2)), np.ones((3, 1, 1)),
                         np.full((3, 2), np.nan), np.full((3, 2), np.inf)):
            with self.subTest(features=features):
                self.client.infer.return_value.as_numpy.return_value = features
                with self.assertRaises(ValueError):
                    self.encoder()(self.image, self.boxes)

    def test_request_errors_propagate_without_local_fallback(self):
        self.client.infer.side_effect = TimeoutError('server unavailable')
        with self.assertRaisesRegex(TimeoutError, 'server unavailable'):
            self.encoder()(self.image, self.boxes)

    def test_invalid_urls_are_rejected(self):
        for url in ('', 'triton:8000/model', 'http://triton:8000', 'http://triton/v2/models/reid',
                    'http://triton/reid?key=value', 'http://user:password@triton/reid'):
            with self.subTest(url=url), self.assertRaises(ValueError):
                TritonReIDEncoder(url, client=self.client)

    def test_real_tracking_callbacks_use_remote_encoder_without_local_weights(self):
        torch.set_num_threads(1)
        model = YOLO('yolov8n.yaml')
        predictions = torch.tensor([[[5, 5, 35, 35, 0.95, 0]]])
        settings = appearance_settings(TRACKER_REID_BACKEND='triton')
        self.client.infer.return_value.as_numpy.return_value = np.array([[3, 4]], dtype=np.float32)
        with mock.patch('utils.TritonReIDTracking.TritonReIDEncoder', return_value=self.encoder()), \
                mock.patch('ultralytics.trackers.bot_sort.ReID', side_effect=AssertionError('Local weights loaded')), \
                mock.patch.object(TritonDetectionPredictor, 'inference', return_value=predictions), \
                tracking_config(settings) as path:
            for unused_frame in range(2):
                result = track_with_triton_reid(
                    settings, model, source=self.image, predictor=TritonDetectionPredictor,
                    tracker=path, persist=True, imgsz=64, verbose=False, save=False,
                )
            self.assertEqual(len(result[0].boxes.id), 1)
            self.assertEqual(model.predictor.trackers[0].frame_id, 2)
            self.assertEqual(model.predictor.args.device, 'cpu')
            np.testing.assert_allclose(
                model.predictor.trackers[0].tracked_stracks[0].smooth_feat, [0.6, 0.8], atol=1e-6)
            self.assertEqual(self.client.infer.call_count, 2)

            model.predictor.trackers[0].reset()
            self.assertEqual(model.predictor.trackers[0].tracked_stracks, [])
            model.predictor = None
            track_with_triton_reid(
                settings, model, source=self.image, predictor=TritonDetectionPredictor,
                tracker=path, persist=True, imgsz=64, verbose=False, save=False,
            )
            self.assertEqual(model.predictor.trackers[0].frame_id, 1)
            self.assertEqual(self.client.infer.call_count, 3)


if __name__ == '__main__':
    unittest.main()