import unittest
import os
from unittest import mock

from utils.TritonHTTPTransport import TritonHTTPTransport


class TritonHTTPTransportTest(unittest.TestCase):
    def test_disabled_preserves_request(self):
        client = mock.Mock()
        transport = TritonHTTPTransport(client, 'none')
        self.assertIs(transport.infer('detector', inputs=['images']), client.infer.return_value)
        client.infer.assert_called_once_with('detector', inputs=['images'])

    def test_compresses_both_directions_without_changing_inputs(self):
        for algorithm in ('gzip', 'deflate'):
            with self.subTest(algorithm=algorithm):
                client = mock.Mock()
                inputs = [object()]
                TritonHTTPTransport(client, algorithm).infer('detector', inputs=inputs)
                client.infer.assert_called_once_with(
                    'detector', inputs=inputs,
                    request_compression_algorithm=algorithm,
                    response_compression_algorithm=algorithm,
                )

    def test_other_client_methods_are_preserved(self):
        client = mock.Mock()
        TritonHTTPTransport(client, 'gzip').close()
        client.close.assert_called_once_with()

    def test_invalid_compression_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'TRITON_HTTP_COMPRESSION'):
            TritonHTTPTransport(mock.Mock(), 'invalid')

    def test_environment_selects_compression(self):
        client = mock.Mock()
        with mock.patch.dict(os.environ, {'TRITON_HTTP_COMPRESSION': ' GZIP '}):
            TritonHTTPTransport(client).infer('model')
        self.assertEqual(client.infer.call_args.kwargs['request_compression_algorithm'], 'gzip')


if __name__ == '__main__':
    unittest.main()