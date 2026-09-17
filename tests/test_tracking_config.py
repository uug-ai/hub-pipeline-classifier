import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import yaml

import utils.VariableClass as variable_module
from utils.TrackingConfig import tracking_config


def appearance_settings(**overrides):
    settings = dict(
        TRACKER_CONFIG='bytetrack.yaml',
        TRACKER_REID_ENABLED=True,
        TRACKER_REID_BACKEND='local',
        TRITON_REID_URL='http://triton:8000/yolo11n_reid',
        TRITON_REID_IMAGE_SIZE=224,
        TRITON_REID_BATCH_SIZE=16,
        TRITON_REID_TIMEOUT=10.0,
        TRACKER_REID_MODEL='yolo11n-cls.pt',
        TRACKER_PROXIMITY_THRESH=0.5,
        TRACKER_APPEARANCE_THRESH=0.8,
        TRACKER_BUFFER=30,
        TRACKER_GMC_METHOD='none',
    )
    settings.update(overrides)
    return SimpleNamespace(**settings)


class TrackingConfigTest(unittest.TestCase):
    def test_remote_config_does_not_initialize_local_weights(self):
        settings = appearance_settings(TRACKER_REID_BACKEND='triton', TRACKER_REID_MODEL='')
        with tracking_config(settings) as path:
            config = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
        self.assertEqual(config['tracker_type'], 'botsort')
        self.assertFalse(config['with_reid'])
        self.assertEqual(config['model'], 'triton')

    def test_builtin_presets_preserve_legacy_thresholds(self):
        for name in ('bytetrack.yaml', 'botsort.yaml'):
            with self.subTest(name=name):
                settings = appearance_settings(TRACKER_REID_ENABLED=False, TRACKER_CONFIG=name)
                with tracking_config(settings) as path:
                    config = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
                    self.assertEqual(config['track_high_thresh'], 0.5)
                    self.assertEqual(config['new_track_thresh'], 0.6)
                    self.assertEqual(config['track_buffer'], 30)
                    self.assertTrue(config['fuse_score'])
                    self.assertFalse(config.get('with_reid', False))

    def test_disabled_preserves_existing_tracker_config(self):
        settings = appearance_settings(TRACKER_REID_ENABLED=False, TRACKER_CONFIG='custom.yaml')
        with tracking_config(settings) as path:
            self.assertEqual(path, 'custom.yaml')

    def test_enabled_creates_config_and_removes_it_after_use(self):
        with tracking_config(appearance_settings()) as path:
            config = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
            self.assertEqual(config['tracker_type'], 'botsort')
            self.assertTrue(config['with_reid'])
            self.assertEqual(config['model'], 'yolo11n-cls.pt')
            self.assertEqual(config['gmc_method'], 'none')
            self.assertEqual(config['track_high_thresh'], 0.5)
        self.assertFalse(Path(path).exists())

    def test_overrides_are_applied_and_workers_have_separate_paths(self):
        settings = appearance_settings(
            TRACKER_REID_MODEL='/models/appearance.pt',
            TRACKER_PROXIMITY_THRESH=0.2,
            TRACKER_APPEARANCE_THRESH=0.9,
            TRACKER_BUFFER=12,
            TRACKER_GMC_METHOD='sparseOptFlow',
        )
        with tracking_config(settings) as first_path, tracking_config(settings) as second_path:
            self.assertNotEqual(first_path, second_path)
            config = yaml.safe_load(Path(first_path).read_text(encoding='utf-8'))
            self.assertEqual(config['model'], '/models/appearance.pt')
            self.assertEqual(config['proximity_thresh'], 0.2)
            self.assertEqual(config['appearance_thresh'], 0.9)
            self.assertEqual(config['track_buffer'], 12)
            self.assertEqual(config['gmc_method'], 'sparseOptFlow')

    def test_cleanup_on_prediction_failure(self):
        with self.assertRaisesRegex(RuntimeError, 'prediction failed'):
            with tracking_config(appearance_settings()) as path:
                raise RuntimeError('prediction failed')
        self.assertFalse(Path(path).exists())

    def test_invalid_settings_are_rejected(self):
        for name, value in (
            ('TRACKER_REID_MODEL', ''),
            ('TRACKER_REID_MODEL', 'auto'),
            ('TRACKER_PROXIMITY_THRESH', -0.1),
            ('TRACKER_APPEARANCE_THRESH', 1.1),
            ('TRACKER_APPEARANCE_THRESH', float('nan')),
            ('TRACKER_BUFFER', 0),
            ('TRACKER_GMC_METHOD', 'invalid'),
        ):
            with self.subTest(name=name, value=value):
                with self.assertRaises(ValueError):
                    with tracking_config(appearance_settings(**{name: value})):
                        self.fail('Invalid configuration was accepted')


class TrackingEnvironmentTest(unittest.TestCase):
    def test_deployment_uses_triton_without_a_worker_gpu(self):
        path = Path(__file__).resolve().parent.parent / 'k8s-deployment.yaml'
        deployment = yaml.safe_load(path.read_text(encoding='utf-8'))
        container = deployment['spec']['template']['spec']['containers'][0]
        settings = {item['name']: item.get('value') for item in container['env']}
        self.assertNotIn('nvidia.com/gpu', container['resources']['limits'])
        self.assertNotIn('nvidia.com/gpu', container['resources'].get('requests', {}))
        self.assertEqual(settings['TRACKER_REID_ENABLED'], 'True')
        self.assertEqual(settings['TRACKER_REID_BACKEND'], 'triton')
        self.assertEqual(settings['VIDEO_DECODER'], 'opencv')
        self.assertTrue(settings['TRITON_REID_URL'].endswith('/yolo11n_reid'))

    def test_remote_environment_settings_are_parsed(self):
        settings = self.variables(
            TRACKER_REID_BACKEND=' TRITON ',
            TRITON_REID_URL='http://10.0.1.24:30314/yolo11n_reid',
            TRITON_REID_IMAGE_SIZE='224',
            TRITON_REID_BATCH_SIZE='8',
            TRITON_REID_TIMEOUT='5.5',
        )
        self.assertEqual(settings.TRACKER_REID_BACKEND, 'triton')
        self.assertEqual(settings.TRITON_REID_URL, 'http://10.0.1.24:30314/yolo11n_reid')
        self.assertEqual(settings.TRITON_REID_IMAGE_SIZE, 224)
        self.assertEqual(settings.TRITON_REID_BATCH_SIZE, 8)
        self.assertEqual(settings.TRITON_REID_TIMEOUT, 5.5)
        with self.assertRaisesRegex(ValueError, 'TRACKER_REID_BACKEND'):
            self.variables(TRACKER_REID_BACKEND='unsupported')

    def variables(self, **overrides):
        environment = {
            'COLOR_PREDICTION_INTERVAL': '5',
            'MIN_CLUSTERS': '3',
            'MAX_CLUSTERS': '3',
            'CLASSIFICATION_FPS': '2',
            'CLASSIFICATION_THRESHOLD': '0.3',
            'MAX_NUMBER_OF_PREDICTIONS': '100',
            'MIN_DISTANCE': '150',
            'MIN_DETECTIONS': '5',
            'ALLOWED_CLASSIFICATIONS': '0,2',
        }
        environment.update(overrides)
        with mock.patch.dict(os.environ, environment, clear=True), mock.patch.object(variable_module, 'load_dotenv'):
            return variable_module.VariableClass()

    def test_reid_is_disabled_by_default(self):
        settings = self.variables()
        self.assertFalse(settings.TRACKER_REID_ENABLED)
        self.assertEqual(settings.TRACKER_CONFIG, 'bytetrack.yaml')
        self.assertEqual(settings.TRACKER_REID_MODEL, 'yolo11n-cls.pt')

    def test_reid_environment_overrides_are_parsed(self):
        settings = self.variables(
            TRACKER_REID_ENABLED=' true ',
            TRACKER_REID_MODEL='/models/appearance.pt',
            TRACKER_PROXIMITY_THRESH='0.2',
            TRACKER_APPEARANCE_THRESH='0.9',
            TRACKER_BUFFER='12',
            TRACKER_GMC_METHOD='sparseOptFlow',
        )
        self.assertTrue(settings.TRACKER_REID_ENABLED)
        self.assertEqual(settings.TRACKER_REID_MODEL, '/models/appearance.pt')
        self.assertEqual(settings.TRACKER_PROXIMITY_THRESH, 0.2)
        self.assertEqual(settings.TRACKER_APPEARANCE_THRESH, 0.9)
        self.assertEqual(settings.TRACKER_BUFFER, 12)
        self.assertEqual(settings.TRACKER_GMC_METHOD, 'sparseOptFlow')

    def test_invalid_boolean_fails_instead_of_silently_disabling_reid(self):
        with self.assertRaisesRegex(ValueError, 'TRACKER_REID_ENABLED'):
            self.variables(TRACKER_REID_ENABLED='yes')


if __name__ == '__main__':
    unittest.main()