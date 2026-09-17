from urllib.parse import urlparse
from time import perf_counter

import cv2
import numpy as np

from utils.TritonHTTPTransport import TritonHTTPTransport


class TritonReIDEncoder:
    """Encode object crops using the remote FP32 RGB/NCHW embedding contract."""

    def __init__(self, url, image_size=224, batch_size=16, timeout=10.0, client=None):
        endpoint = urlparse(url)
        if (endpoint.scheme not in {'http', 'https'} or not endpoint.hostname
                or endpoint.username or endpoint.password or endpoint.query or endpoint.fragment):
            raise ValueError('TRITON_REID_URL must be http(s)://host:port/model-name')
        model_name = endpoint.path.strip('/')
        if not model_name or '/' in model_name:
            raise ValueError('TRITON_REID_URL must contain a single model name, not /v2/models/...')
        if image_size < 1 or batch_size < 1 or not np.isfinite(timeout) or timeout <= 0:
            raise ValueError('ReID image size, batch size, and timeout must be positive')

        from tritonclient import http

        self.http = http
        self.client = TritonHTTPTransport(client if client is not None else http.InferenceServerClient(
            url=endpoint.netloc,
            ssl=endpoint.scheme == 'https',
            connection_timeout=timeout,
            network_timeout=timeout,
        ))
        self.model_name = model_name
        self.image_size = image_size
        self.batch_size = batch_size
        self.reset_stats()

    def reset_stats(self):
        self.request_count = 0
        self.request_seconds = 0.0

    def __call__(self, image, detections):
        if len(detections) == 0:
            return []
        embeddings = []
        feature_size = None
        for offset in range(0, len(detections), self.batch_size):
            batch = np.stack([
                self._crop(image, detection[:4])
                for detection in detections[offset:offset + self.batch_size]
            ])
            input_tensor = self.http.InferInput('images', batch.shape, 'FP32')
            input_tensor.set_data_from_numpy(batch)
            output = self.http.InferRequestedOutput('embeddings')
            start = perf_counter()
            response = self.client.infer(self.model_name, inputs=[input_tensor], outputs=[output])
            self.request_seconds += perf_counter() - start
            self.request_count += 1
            features = response.as_numpy('embeddings')
            if (features is None or features.ndim != 2 or features.shape[0] != len(batch)
                    or features.shape[1] == 0 or not np.isfinite(features).all()):
                raise ValueError('Triton ReID must return finite embeddings shaped [batch, features]')
            if feature_size is not None and features.shape[1] != feature_size:
                raise ValueError('Triton ReID embedding size changed between batches')
            feature_size = features.shape[1]
            features = features.astype(np.float32, copy=True)
            norms = np.linalg.norm(features, axis=1, keepdims=True)
            if not np.isfinite(norms).all() or np.any(norms <= 1e-12):
                raise ValueError('Triton ReID returned invalid or zero-length embeddings')
            embeddings.extend(features / norms)
        return embeddings

    def _crop(self, image, box):
        center_x, center_y, width, height = box
        if not np.isfinite(box).all() or width <= 0 or height <= 0:
            raise ValueError('Invalid ReID bounding box')
        image_height, image_width = image.shape[:2]
        left = int(np.clip(np.floor(center_x - width / 2), 0, image_width))
        right = int(np.clip(np.ceil(center_x + width / 2), 0, image_width))
        top = int(np.clip(np.floor(center_y - height / 2), 0, image_height))
        bottom = int(np.clip(np.ceil(center_y + height / 2), 0, image_height))
        if left >= right or top >= bottom:
            raise ValueError('ReID bounding box does not intersect the image')
        crop = cv2.resize(image[top:bottom, left:right], (self.image_size, self.image_size))
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        return np.ascontiguousarray(rgb.transpose(2, 0, 1), dtype=np.float32) / 255.0