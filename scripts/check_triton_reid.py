import argparse

import numpy as np

from utils.TritonReIDEncoder import TritonReIDEncoder


def main():
    parser = argparse.ArgumentParser(description='Verify live Triton readiness and appearance embedding inference.')
    parser.add_argument('--url', required=True)
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--batch-size', type=int, default=16)
    args = parser.parse_args()
    encoder = TritonReIDEncoder(args.url, image_size=args.image_size, batch_size=args.batch_size)
    try:
        if not encoder.client.is_model_ready(encoder.model_name):
            raise RuntimeError(f'Triton model is not ready: {encoder.model_name}')
        metadata = encoder.client.get_model_metadata(encoder.model_name)
        config = encoder.client.get_model_config(encoder.model_name)
        frame = np.zeros((128, 128, 3), dtype=np.uint8)
        frame[:, :64] = [0, 0, 255]
        frame[:, 64:] = [255, 0, 0]
        detections = np.array([[32, 64, 64, 128, 0], [96, 64, 64, 128, 1]], dtype=np.float32)
        embeddings = np.stack(encoder(frame, detections))
        np.testing.assert_allclose(np.linalg.norm(embeddings, axis=1), 1, atol=1e-6)
        print(f'Model ready: {metadata["name"]}, versions: {metadata["versions"]}')
        print(f'Instance placement: {config["instance_group"]}')
        print(f'Live embeddings: shape={embeddings.shape}, finite=True, normalized=True')
    finally:
        encoder.client.close()


if __name__ == '__main__':
    main()