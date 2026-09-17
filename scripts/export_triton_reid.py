import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn
from ultralytics.models.yolo import YOLO


class AppearanceEmbedding(nn.Module):
    """Pool the classification backbone instead of returning class probabilities."""

    def __init__(self, model):
        super().__init__()
        layers = list(model.model.children())[:-1]
        if not layers or any(layer.f != -1 for layer in layers):
            raise ValueError('Export requires a sequential YOLO classification backbone')
        self.backbone = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, images):
        return self.pool(self.backbone(images)).flatten(1)


def export_repository(weights, repository, model_name='yolo11n_reid', image_size=224, max_batch_size=16):
    if not model_name or any(character not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for character in model_name):
        raise ValueError('Model name must contain only letters, numbers, underscores, and hyphens')
    if image_size < 32 or image_size % 32 or max_batch_size < 1:
        raise ValueError('Image size must be a positive multiple of 32; batch size must be positive')
    repository = Path(repository)
    destination = repository / model_name
    if destination.exists():
        raise FileExistsError(f'Refusing to overwrite existing model: {destination}')
    model = YOLO(weights)
    if model.task != 'classify':
        raise ValueError('ReID export requires a classification checkpoint')
    encoder = AppearanceEmbedding(model.model).float().cpu().eval()
    sample = torch.rand((1, 3, image_size, image_size))
    with torch.inference_mode():
        feature_size = encoder(sample).shape[1]
    repository.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.reid-export-', dir=repository) as temporary:
        staging = Path(temporary) / model_name
        version = staging / '1'
        version.mkdir(parents=True)
        model_path = version / 'model.onnx'
        torch.onnx.export(
            encoder, sample, str(model_path), opset_version=17,
            input_names=['images'], output_names=['embeddings'],
            dynamic_axes={'images': {0: 'batch'}, 'embeddings': {0: 'batch'}},
        )
        onnx.checker.check_model(onnx.load(str(model_path)))
        session = ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])
        for batch_size in {1, max_batch_size}:
            images = torch.rand((batch_size, 3, image_size, image_size))
            with torch.inference_mode():
                expected = encoder(images).numpy()
            actual = session.run(['embeddings'], {'images': images.numpy()})[0]
            np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-5)
            if not np.isfinite(actual).all() or np.any(np.linalg.norm(actual, axis=1) <= 1e-12):
                raise ValueError('Export produced invalid embeddings')
        config = (
            f'name: "{model_name}"\n'
            'platform: "onnxruntime_onnx"\n'
            f'max_batch_size: {max_batch_size}\n'
            f'input [ {{ name: "images" data_type: TYPE_FP32 dims: [3, {image_size}, {image_size}] }} ]\n'
            f'output [ {{ name: "embeddings" data_type: TYPE_FP32 dims: [{feature_size}] }} ]\n'
            'instance_group [ { kind: KIND_GPU count: 1 gpus: [0] } ]\n'
        )
        (staging / 'config.pbtxt').write_text(config, encoding='utf-8')
        staging.rename(destination)
    return destination


def main():
    parser = argparse.ArgumentParser(description='Export a Triton appearance model with FP32 RGB [0,1] inputs.')
    parser.add_argument('--weights', default='yolo11n-cls.pt')
    parser.add_argument('--repository', required=True)
    parser.add_argument('--model-name', default='yolo11n_reid')
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--max-batch-size', type=int, default=16)
    args = parser.parse_args()
    torch.set_num_threads(1)
    path = export_repository(args.weights, args.repository, args.model_name, args.image_size, args.max_batch_size)
    print(f'Exported and validated Triton model: {path}')


if __name__ == '__main__':
    main()