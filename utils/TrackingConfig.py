from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml


@contextmanager
def tracking_config(var):
    """Create a worker-local BoT-SORT config only when appearance matching is enabled."""
    if not var.TRACKER_REID_ENABLED:
        if var.TRACKER_CONFIG in {'bytetrack.yaml', 'botsort.yaml'}:
            yield str(Path(__file__).resolve().parent.parent / 'trackers' / var.TRACKER_CONFIG)
        else:
            yield var.TRACKER_CONFIG
        return

    backend = getattr(var, 'TRACKER_REID_BACKEND', 'local')
    if backend not in {'local', 'triton'}:
        raise ValueError('TRACKER_REID_BACKEND must be local or triton')
    model = 'triton' if backend == 'triton' else var.TRACKER_REID_MODEL.strip()
    if backend == 'local' and (not model or model == 'auto'):
        raise ValueError('TRACKER_REID_MODEL must name an explicit appearance model, not auto')
    for name in ('TRACKER_PROXIMITY_THRESH', 'TRACKER_APPEARANCE_THRESH'):
        if not 0 <= getattr(var, name) <= 1:
            raise ValueError(f'{name} must be between 0 and 1')
    if var.TRACKER_BUFFER < 1:
        raise ValueError('TRACKER_BUFFER must be at least 1')
    if var.TRACKER_GMC_METHOD not in {'none', 'sparseOptFlow', 'orb', 'sift', 'ecc'}:
        raise ValueError('Unsupported TRACKER_GMC_METHOD')

    config = {
        'tracker_type': 'botsort',
        'track_high_thresh': 0.5,
        'track_low_thresh': 0.1,
        'new_track_thresh': 0.6,
        'track_buffer': var.TRACKER_BUFFER,
        'match_thresh': 0.8,
        'fuse_score': True,
        'gmc_method': var.TRACKER_GMC_METHOD,
        'proximity_thresh': var.TRACKER_PROXIMITY_THRESH,
        'appearance_thresh': var.TRACKER_APPEARANCE_THRESH,
        'with_reid': backend == 'local',
        'model': model,
    }
    with TemporaryDirectory(prefix='classifier-tracker-') as directory:
        path = Path(directory) / 'botsort-reid.yaml'
        with path.open('w', encoding='utf-8') as config_file:
            yaml.safe_dump(config, config_file)
        yield str(path)