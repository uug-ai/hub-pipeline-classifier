from functools import partial

from ultralytics.trackers.bot_sort import BOTSORT
from ultralytics.trackers.track import on_predict_postprocess_end, on_predict_start

from utils.TritonReIDEncoder import TritonReIDEncoder


class TritonReIDTracking:
    """Attach remote appearance extraction after BoT-SORT initializes without local weights."""

    def __init__(self, var):
        self.encoder = TritonReIDEncoder(
            var.TRITON_REID_URL,
            image_size=var.TRITON_REID_IMAGE_SIZE,
            batch_size=var.TRITON_REID_BATCH_SIZE,
            timeout=var.TRITON_REID_TIMEOUT,
        )

    def __call__(self, predictor):
        on_predict_start(predictor, persist=True)
        for tracker in predictor.trackers:
            if not isinstance(tracker, BOTSORT):
                raise ValueError('Remote ReID requires the generated BoT-SORT configuration')
            tracker.encoder = self.encoder
            tracker.args.with_reid = True


def track_with_triton_reid(var, model, **options):
    if not getattr(model, 'triton_reid_registered', False):
        callback = TritonReIDTracking(var)
        model.add_callback('on_predict_start', callback)
        model.add_callback('on_predict_postprocess_end', partial(on_predict_postprocess_end, persist=True))
        model.triton_reid_registered = True
    options.pop('persist', None)
    return model.predict(mode='track', batch=1, device='cpu', **options)