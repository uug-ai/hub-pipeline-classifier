import json
import unittest
from unittest import mock

from utils import VideoFrameReader


class FakeCapture:
    def __init__(self, frame_count=30, fps=30):
        self.frame_count = frame_count
        self.fps = fps
        self.position = 0
        self.retrieve_count = 0
        self.released = False

    def isOpened(self):
        return True

    def get(self, prop):
        values = {
            VideoFrameReader.cv2.CAP_PROP_FPS: self.fps,
            VideoFrameReader.cv2.CAP_PROP_FRAME_WIDTH: 640,
            VideoFrameReader.cv2.CAP_PROP_FRAME_HEIGHT: 360,
            VideoFrameReader.cv2.CAP_PROP_FRAME_COUNT: self.frame_count,
        }
        return values.get(prop, 0)

    def grab(self):
        if self.position >= self.frame_count:
            return False
        self.position += 1
        return True

    def retrieve(self):
        self.retrieve_count += 1
        return True, f'frame-{self.position - 1}'

    def release(self):
        self.released = True


class VideoFrameReaderTest(unittest.TestCase):
    def test_opencv_retrieves_only_sampled_frames(self):
        capture = FakeCapture(frame_count=30, fps=30)
        with mock.patch.object(VideoFrameReader.cv2, 'VideoCapture', return_value=capture):
            reader = VideoFrameReader.OpenCVSampledReader('video.mp4', 3, 2)
            frames = list(reader)
            reader.close()

        self.assertEqual(frames, [(0, 'frame-0'), (10, 'frame-10')])
        self.assertEqual(capture.retrieve_count, 2)
        self.assertTrue(capture.released)

    def test_nvdec_selects_frames_before_gpu_download(self):
        probe_payload = json.dumps({
            'streams': [{
                'width': 2560,
                'height': 1440,
                'avg_frame_rate': '240/7',
                'nb_frames': '1034',
            }],
            'format': {'duration': '30.145'},
        })
        probe_result = mock.Mock(stdout=probe_payload)

        with mock.patch.object(VideoFrameReader.shutil, 'which', side_effect=['/usr/bin/ffmpeg', '/usr/bin/ffprobe']), \
                mock.patch.object(VideoFrameReader.subprocess, 'run', side_effect=[probe_result, mock.Mock()]):
            reader = VideoFrameReader.FFmpegNvdecSampledReader('video.mp4', 2, 100)

        command = reader._command()
        video_filter = command[command.index('-vf') + 1]
        self.assertEqual(reader.frame_skip_factor, 17)
        self.assertIn('select=not(mod(n\\,17))', video_filter)
        self.assertIn('hwdownload', video_filter)
        self.assertEqual(command[command.index('-frames:v') + 1], '100')

    def test_auto_falls_back_to_opencv(self):
        fallback_reader = object()
        with mock.patch.object(
                VideoFrameReader,
                'FFmpegNvdecSampledReader',
                side_effect=VideoFrameReader.VideoReaderError('no CUDA device')), \
                mock.patch.object(
                    VideoFrameReader,
                    'OpenCVSampledReader',
                    return_value=fallback_reader):
            reader = VideoFrameReader.create_sampled_video_reader(
                'video.mp4', 2, 100, decoder='auto')

        self.assertIs(reader, fallback_reader)

    def test_explicit_nvdec_does_not_fall_back(self):
        with mock.patch.object(
                VideoFrameReader,
                'FFmpegNvdecSampledReader',
                side_effect=VideoFrameReader.VideoReaderError('no CUDA device')):
            with self.assertRaisesRegex(VideoFrameReader.VideoReaderError, 'no CUDA device'):
                VideoFrameReader.create_sampled_video_reader(
                    'video.mp4', 2, 100, decoder='nvdec')


if __name__ == '__main__':
    unittest.main()