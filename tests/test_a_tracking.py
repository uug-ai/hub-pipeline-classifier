from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch
import yaml
from ultralytics.models.yolo import YOLO
from ultralytics.data.augment import classify_transforms
from ultralytics.engine.results import Boxes
from ultralytics.trackers.bot_sort import BOTSORT
from ultralytics.trackers.byte_tracker import BYTETracker

from utils.TrackingConfig import tracking_config
from utils.TritonDetectionPredictor import TritonDetectionPredictor
from test_tracking_config import appearance_settings


class AppearanceTrackingIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.directory = TemporaryDirectory(prefix='test-appearance-')
        cls.addClassCleanup(cls.directory.cleanup)
        cls.model_path = str(Path(cls.directory.name) / 'appearance.pt')
        model = YOLO('yolo11n-cls.yaml')
        model.model.transforms = classify_transforms(224)
        model.save(cls.model_path)

    def test_real_encoder_tracks_reappearing_objects_and_resets(self):
        settings = appearance_settings(TRACKER_REID_MODEL=self.model_path)
        with tracking_config(settings) as path:
            tracker = BOTSORT(SimpleNamespace(**yaml.safe_load(Path(path).read_text(encoding='utf-8'))))

        frame = np.zeros((128, 128, 3), dtype=np.uint8)
        frame[20:100, 20:60] = [0, 0, 255]
        frame[20:100, 70:110] = [255, 0, 0]
        detections = Boxes(np.array([
            [20, 20, 60, 100, 0.95, 0],
            [70, 20, 110, 100, 0.95, 0],
        ], dtype=np.float32), frame.shape[:2])

        initial = tracker.update(detections, frame)
        self.assertEqual(len(initial), 2)
        for track in tracker.tracked_stracks:
            self.assertIsNotNone(track.smooth_feat)
            self.assertTrue(np.isfinite(track.smooth_feat).all())

        empty = Boxes(np.empty((0, 6), dtype=np.float32), frame.shape[:2])
        tracker.update(empty, frame)
        self.assertEqual(len(tracker.lost_stracks), 2)
        recovered = tracker.update(detections, frame)
        self.assertEqual(set(initial[:, 4]), set(recovered[:, 4]))

        tracker.reset()
        self.assertEqual(tracker.tracked_stracks, [])
        self.assertEqual(tracker.lost_stracks, [])
        self.assertEqual(tracker.frame_id, 0)

    def test_legacy_bytetrack_preset_loads_in_upgraded_library(self):
        with tracking_config(appearance_settings(TRACKER_REID_ENABLED=False)) as path:
            tracker = BYTETracker(SimpleNamespace(**yaml.safe_load(Path(path).read_text(encoding='utf-8'))))
        self.assertEqual(tracker.args.track_high_thresh, 0.5)
        self.assertEqual(tracker.args.new_track_thresh, 0.6)

    def test_appearance_resolves_ambiguous_overlapping_boxes(self):
        encoder = mock.Mock(side_effect=[
            [np.array([1.0, 0.0]), np.array([0.0, 1.0])],
            [np.array([0.0, 1.0]), np.array([1.0, 0.0])],
        ])
        with tracking_config(appearance_settings()) as path:
            config = SimpleNamespace(**yaml.safe_load(Path(path).read_text(encoding='utf-8')))
        with mock.patch('ultralytics.trackers.bot_sort.ReID', return_value=encoder):
            tracker = BOTSORT(config)

        frame = np.zeros((128, 128, 3), dtype=np.uint8)
        initial_boxes = Boxes(np.array([
            [20, 20, 80, 100, 0.95, 0],
            [40, 20, 100, 100, 0.95, 0],
        ], dtype=np.float32), frame.shape[:2])
        overlap_boxes = Boxes(np.array([
            [30, 20, 90, 100, 0.95, 0],
            [30, 20, 90, 100, 0.95, 0],
        ], dtype=np.float32), frame.shape[:2])
        initial = tracker.update(initial_boxes, frame)
        overlapping = tracker.update(overlap_boxes, frame)
        initial_ids = {int(row[-1]): int(row[4]) for row in initial}
        overlapping_ids = {int(row[-1]): int(row[4]) for row in overlapping}
        self.assertEqual(overlapping_ids, {0: initial_ids[1], 1: initial_ids[0]})
        self.assertEqual(encoder.call_count, 2)

    def test_triton_predictor_factory_runs_with_real_tracking_callbacks(self):
        model = YOLO('yolov8n.yaml')
        frame = np.zeros((128, 128, 3), dtype=np.uint8)
        frame[20:100, 20:80] = [0, 0, 255]
        predictions = torch.tensor([[[20, 20, 80, 100, 0.95, 0]]])
        settings = appearance_settings(TRACKER_REID_MODEL=self.model_path)
        with tracking_config(settings) as path:
            with mock.patch.object(TritonDetectionPredictor, 'inference', return_value=predictions):
                results = model.track(
                    source=frame,
                    predictor=TritonDetectionPredictor,
                    tracker=path,
                    persist=True,
                    imgsz=128,
                    conf=0.3,
                    verbose=False,
                    save=False,
                )
        self.assertIsInstance(model.predictor.trackers[0], BOTSORT)
        self.assertIsNotNone(model.predictor.trackers[0].encoder)
        self.assertEqual(len(results[0].boxes.id), 1)
        self.assertTrue(np.isfinite(model.predictor.trackers[0].tracked_stracks[0].smooth_feat).all())


if __name__ == '__main__':
    unittest.main()