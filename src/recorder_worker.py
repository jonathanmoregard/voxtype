import queue as _queue
import numpy as np
import sounddevice as sd
import webrtcvad
from collections import deque
from threading import Event
from PyQt5.QtCore import QThread, pyqtSignal

from pipeline import SENTINEL
from pitch_detector import PitchDetector
from utils import ConfigManager


class RecorderWorker(QThread):
    statusSignal = pyqtSignal(str)
    pitchSignal = pyqtSignal(float)  # Hz, emitted during speech

    def __init__(self, audio_q, recording_stopped: Event):
        super().__init__()
        self._audio_q = audio_q
        self._recording_stopped = recording_stopped
        self._stop_event = Event()

    def stop(self):
        self._stop_event.set()

    def _make_burst(self, recording: list, sample_rate: int, min_duration_ms: int):
        """Convert a frame list to a numpy burst array, or None if too short."""
        audio = np.array(recording, dtype=np.int16)
        duration_ms = (len(audio) / sample_rate) * 1000
        if duration_ms < min_duration_ms:
            return None
        return audio

    def run(self):
        self.statusSignal.emit('recording')
        recording_options = ConfigManager.get_config_section('recording_options')
        sample_rate = recording_options.get('sample_rate') or 16000
        frame_duration_ms = 30
        frame_size = int(sample_rate * frame_duration_ms / 1000)
        silence_duration_ms = recording_options.get('silence_duration') or 900
        silence_frames = int(silence_duration_ms / frame_duration_ms)
        min_duration_ms = recording_options.get('min_duration') or 100
        initial_skip = int(0.15 * sample_rate / frame_size)

        vad = webrtcvad.Vad(2)
        pitch_enabled = bool(ConfigManager.get_config_value('misc', 'pitch_detection_enabled'))
        pitch_detector = PitchDetector(sample_rate=sample_rate, hop_size=frame_size) if pitch_enabled else None
        audio_buffer = deque(maxlen=frame_size)
        data_ready = Event()

        def audio_callback(indata, frames, time, status):
            if status:
                ConfigManager.console_print(f'Audio callback status: {status}')
            audio_buffer.extend(indata[:, 0])
            data_ready.set()

        recording = []
        speech_detected = False
        silent_frame_count = 0
        frames_to_skip = initial_skip
        pitch_history = deque(maxlen=5)

        with sd.InputStream(samplerate=sample_rate, channels=1, dtype='int16',
                            blocksize=frame_size,
                            device=recording_options.get('sound_device'),
                            callback=audio_callback):
            while not self._stop_event.is_set():
                triggered = data_ready.wait(timeout=0.1)
                if not triggered:
                    continue
                data_ready.clear()

                if len(audio_buffer) < frame_size:
                    continue

                frame = np.array(list(audio_buffer), dtype=np.int16)
                audio_buffer.clear()
                recording.extend(frame)

                if frames_to_skip > 0:
                    frames_to_skip -= 1
                    continue

                if vad.is_speech(frame.tobytes(), sample_rate):
                    silent_frame_count = 0
                    speech_detected = True
                    if pitch_detector is not None:
                        pitch = pitch_detector.detect(frame)
                        if pitch is not None:
                            pitch_history.append(pitch)
                            if len(pitch_history) >= 2:
                                self.pitchSignal.emit(float(np.median(list(pitch_history))))
                else:
                    silent_frame_count += 1

                if speech_detected and silent_frame_count > silence_frames:
                    burst = self._make_burst(recording, sample_rate, min_duration_ms)
                    if burst is not None:
                        self._audio_q.put(burst)
                    # reset for next burst
                    recording = []
                    speech_detected = False
                    silent_frame_count = 0
                    frames_to_skip = initial_skip

        # Flush partial burst on stop
        if recording:
            burst = self._make_burst(recording, sample_rate, min_duration_ms)
            if burst is not None:
                self._audio_q.put(burst)

        self._audio_q.put(SENTINEL)
        self._recording_stopped.set()
        self.statusSignal.emit('idle')
        ConfigManager.console_print('RecorderWorker stopped.')
