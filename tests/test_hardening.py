import importlib
import sys
import types
import unittest
from unittest import mock


class FakeAMQPError(Exception):
    pass


class FakeConnectionClosed(FakeAMQPError):
    pass


class FakeChannelClosed(FakeAMQPError):
    pass


class FakePika(types.SimpleNamespace):
    pass


class FakeURLParameters:
    def __init__(self, url):
        self.url = url
        self.heartbeat = None
        self.blocked_connection_timeout = None


class FakeChannel:
    def __init__(self, body=None):
        self.is_closed = False
        self.body = body
        self.declared = False
        self.published = []

    def queue_declare(self, *args, **kwargs):
        self.declared = True

    def basic_get(self, *args, **kwargs):
        return None, None, self.body

    def basic_publish(self, **kwargs):
        self.published.append(kwargs)


class FakeBlockingConnection:
    instances = []

    def __init__(self, parameters):
        self.parameters = parameters
        self.is_open = True
        self.is_closed = False
        self.channels = []
        self.process_data_events_calls = 0
        FakeBlockingConnection.instances.append(self)

    def channel(self):
        channel = FakeChannel()
        self.channels.append(channel)
        return channel

    def process_data_events(self, time_limit=0):
        self.process_data_events_calls += 1

    def close(self):
        self.is_open = False
        self.is_closed = True


def install_fake_broker_dependencies():
    fake_pika = FakePika(
        URLParameters=FakeURLParameters,
        BlockingConnection=FakeBlockingConnection,
        exceptions=types.SimpleNamespace(
            AMQPError=FakeAMQPError,
            AMQPHeartbeatTimeout=FakeAMQPError,
            AMQPConnectionError=FakeAMQPError,
            ConnectionClosed=FakeConnectionClosed,
            ConnectionClosedByBroker=FakeConnectionClosed,
            ChannelClosed=FakeChannelClosed,
        ),
    )
    sys.modules['pika'] = fake_pika
    sys.modules['boto3'] = types.SimpleNamespace()
    sys.modules['confluent_kafka'] = types.SimpleNamespace(Producer=object, Consumer=object)


def import_broker_module():
    install_fake_broker_dependencies()
    sys.modules.pop('utils.message_brokers', None)
    FakeBlockingConnection.instances = []
    return importlib.import_module('utils.message_brokers')


class RabbitMQHardeningTest(unittest.TestCase):
    def test_receive_services_heartbeat_while_idle(self):
        message_brokers = import_broker_module()
        broker = message_brokers.RabbitMQ(
            queue_name='source',
            target_queue_name='target',
            exchange='',
            host='rabbitmq:5672',
            username='user',
            password='pass',
        )

        with mock.patch.object(message_brokers.time, 'sleep'):
            self.assertEqual(broker.receive_message(), [])

        self.assertGreaterEqual(broker.connection.process_data_events_calls, 3)

    def test_ensure_connection_reconnects_and_recreates_channels(self):
        message_brokers = import_broker_module()
        broker = message_brokers.RabbitMQ(
            queue_name='source',
            target_queue_name='target',
            exchange='',
            host='rabbitmq:5672',
            username='user',
            password='pass',
        )
        broker.connection.is_open = False
        broker.connection.is_closed = True

        with mock.patch('builtins.print'):
            broker.ensure_connection()

        self.assertEqual(len(FakeBlockingConnection.instances), 2)
        self.assertIsNotNone(broker.readChannel)
        self.assertIsNotNone(broker.publishChannel)


class FakeFrame:
    def copy(self):
        return self


class FakeVideoCapture:
    def __init__(self, fps, frame_count):
        self.fps = fps
        self.frame_count = frame_count
        self.released = False
        self.read_count = 0

    def isOpened(self):
        return True

    def get(self, prop):
        if prop == 5:
            return self.fps
        if prop == 7:
            return self.frame_count
        if prop in (3, 4):
            return 640
        return 0

    def read(self):
        if self.read_count < self.frame_count:
            self.read_count += 1
            return True, FakeFrame()
        return False, None

    def release(self):
        self.released = True


class FakeVideoWriter:
    instances = []

    def __init__(self, *args, **kwargs):
        self.released = False
        self.frames = []
        FakeVideoWriter.instances.append(self)

    @staticmethod
    def fourcc(*args):
        return 0

    def write(self, frame):
        self.frames.append(frame)

    def release(self):
        self.released = True


class FakeReturnJSON:
    def __init__(self):
        self.return_object = {'operation': 'classify', 'data': {}}

    def add_detected_object(self, classification_object):
        pass

    def save_returnjson(self, path):
        pass


class FakeRabbitMQ:
    def __init__(self):
        self.process_data_events_calls = 0
        self.sent = []

    def process_data_events(self):
        self.process_data_events_calls += 1

    def receive_message(self):
        return {'payload': {'key': 'video'}, 'source': 'vault'}

    def send_message(self, message):
        self.sent.append(message)


class FakeVault:
    def retrieve_media(self, **kwargs):
        return None


class FakeVar:
    LOGGING = False
    TIME_VERBOSE = False
    MODEL_NAME = 'fake.pt'
    QUEUE_NAME = 'source'
    QUEUE_EXCHANGE = ''
    QUEUE_HOST = 'rabbitmq:5672'
    QUEUE_USERNAME = 'user'
    QUEUE_PASSWORD = 'pass'
    STORAGE_URI = 'storage-uri'
    STORAGE_ACCESS_KEY = 'access-key'
    STORAGE_SECRET_KEY = 'secret-key'
    MEDIA_SAVEPATH = '/tmp/input.mp4'
    SAVE_VIDEO = False
    OUTPUT_MEDIA_SAVEPATH = '/tmp/output.mp4'
    CLASSIFICATION_FPS = 3
    FIND_DOMINANT_COLORS = False
    MIN_CLUSTERS = 3
    MAX_CLUSTERS = 3
    CREATE_BBOX_FRAME = False
    PLOT = False
    MAX_NUMBER_OF_PREDICTIONS = 1
    CLASSIFICATION_THRESHOLD = 0.3
    ALLOWED_CLASSIFICATIONS = []
    MIN_DISTANCE = 0
    MIN_DETECTIONS = 1
    CREATE_RETURN_JSON = True
    SAVE_RETURN_JSON = False
    RETURN_JSON_SAVEPATH = '/tmp/output.json'
    SAVE_BBOX_FRAME = False
    BBOX_FRAME_SAVEPATH = '/tmp/output.jpg'
    TARGET_QUEUE_NAME = ''


class FakeModel:
    def __init__(self):
        self.predictor = types.SimpleNamespace(
            trackers=[FakeTracker()],
            vid_path=['/tmp/previous.mp4'],
        )

    def track(self, **kwargs):
        return [types.SimpleNamespace(boxes=[], masks=None, names={})]


class FakeTracker:
    def __init__(self):
        self.reset_calls = 0

    def reset(self):
        self.reset_calls += 1


def import_classifier_with_fakes(capture):
    fake_cv2 = types.SimpleNamespace(
        CAP_PROP_FRAME_WIDTH=3,
        CAP_PROP_FRAME_HEIGHT=4,
        CAP_PROP_FPS=5,
        CAP_PROP_FRAME_COUNT=7,
        VideoCapture=lambda path: capture,
        VideoWriter=FakeVideoWriter,
        imshow=lambda *args, **kwargs: None,
        waitKey=lambda *args, **kwargs: None,
        imwrite=lambda *args, **kwargs: None,
        destroyAllWindows=mock.Mock(),
    )
    sys.modules['cv2'] = fake_cv2
    sys.modules['torch'] = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False)
    )
    sys.modules['numpy'] = types.SimpleNamespace(int32=lambda value: value)
    sys.modules['ultralytics'] = types.SimpleNamespace(YOLO=object)
    sys.modules['utils.ReturnObject'] = types.SimpleNamespace(ReturnJSON=FakeReturnJSON)
    sys.modules['utils.TranslateObject'] = types.SimpleNamespace(translate=lambda value: value)
    sys.modules['utils.VariableClass'] = types.SimpleNamespace(VariableClass=object)
    sys.modules['utils.ColorDetector'] = types.SimpleNamespace(FindObjectColors=object)
    sys.modules['utils.ClassificationObject'] = types.SimpleNamespace(ClassificationObject=object)
    sys.modules['utils.AnnotateFrame'] = types.SimpleNamespace(
        annotate_frame=lambda **kwargs: kwargs['frame'],
        annotate_bbox_frame=lambda **kwargs: kwargs['bbox_frame'],
    )
    sys.modules['utils.ClassificationObjectFunctions'] = types.SimpleNamespace(
        create_classification_object=lambda **kwargs: object(),
        edit_classification_object=lambda **kwargs: None,
        find_classification_object=lambda *args, **kwargs: object(),
    )
    sys.modules['utils.kerberos_vault'] = types.SimpleNamespace(KerberosVault=object)
    sys.modules['utils.message_brokers'] = types.SimpleNamespace(RabbitMQ=object)
    sys.modules.pop('object_classification_yolov8', None)
    module = importlib.import_module('object_classification_yolov8')
    return module, fake_cv2


class ClassifierCleanupTest(unittest.TestCase):
    def test_low_fps_message_releases_capture(self):
        capture = FakeVideoCapture(fps=1, frame_count=1)
        classifier, cv2 = import_classifier_with_fakes(capture)

        processed = classifier.process_message(
            FakeVar(), FakeModel(), FakeRabbitMQ(), FakeVault(), {'payload': {'key': 'video'}, 'source': 'vault'}
        )

        self.assertFalse(processed)
        self.assertTrue(capture.released)
        cv2.destroyAllWindows.assert_called_once()

    def test_successful_message_releases_capture_and_writer(self):
        capture = FakeVideoCapture(fps=30, frame_count=1)
        classifier, cv2 = import_classifier_with_fakes(capture)
        var = FakeVar()
        var.SAVE_VIDEO = True
        FakeVideoWriter.instances = []

        processed = classifier.process_message(
            var, FakeModel(), FakeRabbitMQ(), FakeVault(), {'payload': {'key': 'video'}, 'source': 'vault'}
        )

        self.assertTrue(processed)
        self.assertTrue(capture.released)
        self.assertTrue(FakeVideoWriter.instances[0].released)
        cv2.destroyAllWindows.assert_called_once()

    def test_process_message_resets_tracker_state_between_messages(self):
        capture = FakeVideoCapture(fps=30, frame_count=1)
        classifier, _ = import_classifier_with_fakes(capture)
        model = FakeModel()

        processed = classifier.process_message(
            FakeVar(), model, FakeRabbitMQ(), FakeVault(), {'payload': {'key': 'video'}, 'source': 'vault'}
        )

        self.assertTrue(processed)
        self.assertEqual(model.predictor.trackers[0].reset_calls, 1)
        self.assertEqual(model.predictor.vid_path, [None])

    def test_process_message_resets_tracker_state_on_sequential_messages(self):
        capture = FakeVideoCapture(fps=30, frame_count=2)
        classifier, _ = import_classifier_with_fakes(capture)
        model = FakeModel()
        rabbitmq = FakeRabbitMQ()

        processed_first = classifier.process_message(
            FakeVar(), model, rabbitmq, FakeVault(), {'payload': {'key': 'video-a'}, 'source': 'vault'}
        )

        second_capture = FakeVideoCapture(fps=30, frame_count=2)
        classifier.cv2.VideoCapture = lambda path: second_capture
        processed_second = classifier.process_message(
            FakeVar(), model, rabbitmq, FakeVault(), {'payload': {'key': 'video-b'}, 'source': 'vault'}
        )

        self.assertTrue(processed_first)
        self.assertTrue(processed_second)
        self.assertEqual(model.predictor.trackers[0].reset_calls, 2)
        self.assertEqual(model.predictor.vid_path, [None])


class MainLoopHardeningTest(unittest.TestCase):
    def test_main_retries_model_load_without_exiting_worker(self):
        capture = FakeVideoCapture(fps=30, frame_count=1)
        classifier, _ = import_classifier_with_fakes(capture)
        fake_rabbitmq = FakeRabbitMQ()

        classifier.VariableClass = FakeVar
        classifier.RabbitMQ = lambda **kwargs: fake_rabbitmq
        classifier.KerberosVault = lambda **kwargs: FakeVault()

        load_model = mock.Mock(side_effect=[RuntimeError('load failed'), FakeModel()])
        process_message = mock.Mock(side_effect=KeyboardInterrupt)

        with mock.patch.object(classifier, 'load_model', load_model), \
                mock.patch.object(classifier, 'process_message', process_message), \
                mock.patch.object(classifier.time, 'sleep') as sleep_mock:
            with self.assertRaises(KeyboardInterrupt):
                classifier.main()

        self.assertEqual(load_model.call_count, 2)
        sleep_mock.assert_called_once_with(5)
        self.assertGreaterEqual(fake_rabbitmq.process_data_events_calls, 2)
        process_message.assert_called_once()

    def test_main_continues_after_process_message_exception(self):
        capture = FakeVideoCapture(fps=30, frame_count=1)
        classifier, _ = import_classifier_with_fakes(capture)
        fake_rabbitmq = FakeRabbitMQ()

        classifier.VariableClass = FakeVar
        classifier.RabbitMQ = lambda **kwargs: fake_rabbitmq
        classifier.KerberosVault = lambda **kwargs: FakeVault()

        process_message = mock.Mock(side_effect=[RuntimeError('boom'), KeyboardInterrupt])

        with mock.patch.object(classifier, 'load_model', return_value=FakeModel()), \
                mock.patch.object(classifier, 'process_message', process_message):
            with self.assertRaises(KeyboardInterrupt):
                classifier.main()

        self.assertEqual(process_message.call_count, 2)
        self.assertGreaterEqual(fake_rabbitmq.process_data_events_calls, 3)


if __name__ == '__main__':
    unittest.main()
