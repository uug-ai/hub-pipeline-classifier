import json
import shutil
import subprocess
from fractions import Fraction

import cv2
import numpy as np


class VideoReaderError(RuntimeError):
    pass


class OpenCVSampledReader:
    backend = 'opencv'

    def __init__(self, path, classification_fps, max_predictions):
        self.capture = cv2.VideoCapture(path)
        if not self.capture.isOpened():
            raise VideoReaderError(f'Unable to open video file: {path}')

        self.fps = self.capture.get(cv2.CAP_PROP_FPS)
        self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.max_predictions = max_predictions
        self.frame_skip_factor = int(self.fps / classification_fps) if classification_fps else 0
        if self.frame_skip_factor <= 0:
            self.close()
            raise VideoReaderError('Video FPS is lower than CLASSIFICATION_FPS')

    def __iter__(self):
        frame_number = 0
        predicted_frames = 0
        while predicted_frames < self.max_predictions and frame_number < self.frame_count:
            if not self.capture.grab():
                break

            if frame_number % self.frame_skip_factor == 0:
                success, frame = self.capture.retrieve()
                if not success:
                    break
                yield frame_number, frame
                predicted_frames += 1
            frame_number += 1

    def close(self):
        if self.capture is not None:
            self.capture.release()
            self.capture = None


class FFmpegNvdecSampledReader:
    backend = 'nvdec'

    def __init__(self, path, classification_fps, max_predictions):
        ffmpeg_path = shutil.which('ffmpeg')
        ffprobe_path = shutil.which('ffprobe')
        if not ffmpeg_path or not ffprobe_path:
            raise VideoReaderError('FFmpeg and ffprobe are required for NVDEC')

        self.path = path
        self.ffmpeg_path = ffmpeg_path
        self.fps, self.width, self.height, self.frame_count = self._probe(ffprobe_path)
        self.max_predictions = max_predictions
        self.frame_skip_factor = int(self.fps / classification_fps) if classification_fps else 0
        if self.frame_skip_factor <= 0:
            raise VideoReaderError('Video FPS is lower than CLASSIFICATION_FPS')

        self._verify_nvdec()
        self.process = None

    def _probe(self, ffprobe_path):
        command = [
            ffprobe_path,
            '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height,avg_frame_rate,nb_frames,duration:format=duration',
            '-of', 'json',
            self.path,
        ]
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=30, check=True)
            payload = json.loads(result.stdout)
        except (subprocess.SubprocessError, OSError, json.JSONDecodeError) as exc:
            raise VideoReaderError(f'Unable to probe video metadata: {exc}') from exc
        if not payload.get('streams'):
            raise VideoReaderError('Video does not contain a readable stream')

        stream = payload['streams'][0]
        fps = float(Fraction(stream['avg_frame_rate']))
        duration = float(stream.get('duration') or payload.get('format', {}).get('duration') or 0)
        frame_count_value = stream.get('nb_frames')
        frame_count = int(frame_count_value) if frame_count_value and frame_count_value != 'N/A' else round(duration * fps)
        return fps, int(stream['width']), int(stream['height']), frame_count

    def _verify_nvdec(self):
        command = [
            self.ffmpeg_path,
            '-v', 'error',
            '-hwaccel', 'cuda',
            '-hwaccel_output_format', 'cuda',
            '-c:v', 'h264_cuvid',
            '-i', self.path,
            '-frames:v', '1',
            '-f', 'null',
            '-',
        ]
        try:
            subprocess.run(command, capture_output=True, timeout=30, check=True)
        except (subprocess.SubprocessError, OSError) as exc:
            raise VideoReaderError(f'NVDEC probe failed: {exc}') from exc

    def _command(self):
        select_filter = f'select=not(mod(n\\,{self.frame_skip_factor})),hwdownload,format=nv12,format=bgr24'
        return [
            self.ffmpeg_path,
            '-v', 'error',
            '-hwaccel', 'cuda',
            '-hwaccel_output_format', 'cuda',
            '-c:v', 'h264_cuvid',
            '-i', self.path,
            '-an',
            '-sn',
            '-vf', select_filter,
            '-fps_mode', 'passthrough',
            '-frames:v', str(self.max_predictions),
            '-f', 'rawvideo',
            '-pix_fmt', 'bgr24',
            'pipe:1',
        ]

    def __iter__(self):
        self.process = subprocess.Popen(
            self._command(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=self.width * self.height * 3,
        )
        frame_size = self.width * self.height * 3
        predicted_frames = 0

        while predicted_frames < self.max_predictions:
            frame_bytes = self._read_exact(frame_size)
            if not frame_bytes:
                break
            if len(frame_bytes) != frame_size:
                raise VideoReaderError('FFmpeg returned an incomplete video frame')

            frame = np.frombuffer(frame_bytes, dtype=np.uint8).reshape(self.height, self.width, 3)
            yield predicted_frames * self.frame_skip_factor, frame
            predicted_frames += 1

        return_code = self.process.wait()
        if return_code != 0:
            error = self.process.stderr.read().decode(errors='replace').strip()
            raise VideoReaderError(f'FFmpeg NVDEC failed: {error or return_code}')

    def _read_exact(self, size):
        chunks = []
        remaining = size
        while remaining:
            chunk = self.process.stdout.read(remaining)
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        return b''.join(chunks)

    def close(self):
        if self.process is None:
            return
        if self.process.stdout is not None:
            self.process.stdout.close()
        if self.process.stderr is not None:
            self.process.stderr.close()
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.process = None


def create_sampled_video_reader(path, classification_fps, max_predictions, decoder='auto', logging=False):
    decoder = decoder.strip().lower()
    if decoder not in {'auto', 'nvdec', 'opencv'}:
        raise ValueError(f'Unsupported video decoder: {decoder}')

    if decoder in {'auto', 'nvdec'}:
        try:
            return FFmpegNvdecSampledReader(path, classification_fps, max_predictions)
        except VideoReaderError as exc:
            if decoder == 'nvdec':
                raise
            if logging:
                print(f'NVDEC unavailable, falling back to OpenCV: {exc}')

    return OpenCVSampledReader(path, classification_fps, max_predictions)