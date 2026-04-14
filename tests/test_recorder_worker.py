import queue
import numpy as np
from threading import Event
from pipeline import SENTINEL


def test_make_burst_returns_array_when_long_enough():
    from recorder_worker import RecorderWorker
    audio_q = queue.Queue()
    recording_stopped = Event()
    worker = RecorderWorker(audio_q, recording_stopped)

    recording = list(np.ones(16000, dtype=np.int16))  # 1s at 16000Hz
    result = worker._make_burst(recording, sample_rate=16000, min_duration_ms=100)
    assert result is not None
    assert result.dtype == np.int16
    assert len(result) == 16000


def test_make_burst_returns_none_when_too_short():
    from recorder_worker import RecorderWorker
    audio_q = queue.Queue()
    recording_stopped = Event()
    worker = RecorderWorker(audio_q, recording_stopped)

    short = list(np.ones(800, dtype=np.int16))  # 50ms at 16000Hz — below 100ms min
    result = worker._make_burst(short, sample_rate=16000, min_duration_ms=100)
    assert result is None
