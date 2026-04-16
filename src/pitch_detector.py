import aubio
import numpy as np


class PitchDetector:
    """Streaming pitch detector using aubio's yinfft.

    Feed int16 frames of exactly ``hop_size`` samples via :meth:`detect`.
    Returns Hz when the frame is confidently voiced within ``[min_hz, max_hz]``,
    otherwise ``None``.
    """

    def __init__(self, sample_rate=16000, hop_size=480, buf_size=2048,
                 min_hz=70.0, max_hz=350.0, silence_db=-40.0, tolerance=0.85):
        self._min_hz = min_hz
        self._max_hz = max_hz
        self._pitch = aubio.pitch('yinfft', buf_size, hop_size, sample_rate)
        self._pitch.set_unit('Hz')
        self._pitch.set_tolerance(tolerance)
        self._pitch.set_silence(silence_db)

    def detect(self, frame_int16):
        audio = frame_int16.astype(np.float32) / 32768.0
        hz = float(self._pitch(audio)[0])
        if hz < self._min_hz or hz > self._max_hz:
            return None
        return hz
