import unittest
from unittest import mock

import worker_launcher


class FakeWorker:
    def __init__(self, return_code=None):
        self.return_code = return_code
        self.terminated = False
        self.killed = False

    def poll(self):
        return self.return_code

    def terminate(self):
        self.terminated = True
        self.return_code = -15

    def wait(self, timeout=None):
        return self.return_code

    def kill(self):
        self.killed = True
        self.return_code = -9


class WorkerLauncherTest(unittest.TestCase):
    def test_worker_count_must_be_positive(self):
        with mock.patch.dict(worker_launcher.os.environ, {'WORKER_PROCESSES': '0'}):
            with self.assertRaisesRegex(ValueError, 'at least 1'):
                worker_launcher.worker_count()

    def test_main_starts_configured_workers_and_stops_peers_on_failure(self):
        workers = [FakeWorker(return_code=7)] + [FakeWorker() for _ in range(4)]

        with mock.patch.object(worker_launcher, 'worker_count', return_value=5), \
                mock.patch.object(worker_launcher.subprocess, 'Popen', side_effect=workers) as popen, \
                mock.patch.object(worker_launcher.signal, 'signal'):
            return_code = worker_launcher.main()

        self.assertEqual(return_code, 7)
        self.assertEqual(popen.call_count, 5)
        popen.assert_called_with([
            worker_launcher.sys.executable,
            'object_classification_yolov8.py',
        ])
        self.assertTrue(all(worker.terminated for worker in workers[1:]))


if __name__ == '__main__':
    unittest.main()