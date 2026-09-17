import os


class TritonHTTPTransport:
    """Add lossless HTTP compression to a Triton client's inference requests."""

    def __init__(self, client, compression=None):
        if compression is None:
            compression = os.getenv('TRITON_HTTP_COMPRESSION', 'none').strip().lower()
        if compression not in {'none', 'gzip', 'deflate'}:
            raise ValueError('TRITON_HTTP_COMPRESSION must be none, gzip, or deflate')
        self.client = client
        self.compression = None if compression == 'none' else compression

    def __getattr__(self, name):
        return getattr(self.client, name)

    def infer(self, *args, **kwargs):
        if self.compression is not None:
            kwargs.setdefault('request_compression_algorithm', self.compression)
            kwargs.setdefault('response_compression_algorithm', self.compression)
        return self.client.infer(*args, **kwargs)