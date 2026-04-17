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

    def _make_burst(self, recording: list, sample_rate: int, min_duration_ms: int,
                    trim_frames: int = 0, frame_size: int = 0):
        """Convert a frame list to a numpy burst array, or None if too short."""
        audio = np.array(recording, dtype=np.int16)
        if trim_frames > 0 and frame_size > 0:
            trim_samples = trim_frames * frame_size
            audio = audio[:-trim_samples] if trim_samples < len(audio) else audio[:0]
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
        audio_buffer = deque(maxlen=frame_size * 20)  # hold ~600ms, absorbs main-loop lag
        data_ready = Event()
        callback_count = [0]
        drain_count = [0]

        def audio_callback(indata, frames, time, status):
            if status:
                ConfigManager.console_print(f'Audio callback status: {status}')
            callback_count[0] += 1
            prev_len = len(audio_buffer)
            audio_buffer.extend(indata[:, 0])
            # deque with maxlen silently drops from the left on overflow
            appended = len(indata[:, 0])
            if prev_len + appended > audio_buffer.maxlen:
                dropped = (prev_len + appended) - audio_buffer.maxlen
                ConfigManager.console_print(
                    f'[recorder] AUDIO BUFFER OVERFLOW: dropped {dropped} samples '
                    f'(callbacks={callback_count[0]}, drains={drain_count[0]})'
                )
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

                # Drain all full frames currently queued (not just the latest one)
                while len(audio_buffer) >= frame_size:
                    drain_count[0] += 1
                    frame_samples = [audio_buffer.popleft() for _ in range(frame_size)]
                    frame = np.array(frame_samples, dtype=np.int16)
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
                        burst = self._make_burst(recording, sample_rate, min_duration_ms,
                                                 trim_frames=silent_frame_count, frame_size=frame_size)
                        if burst is not None:
                            duration_ms = (len(burst) / sample_rate) * 1000
                            ConfigManager.console_print(
                                f'[recorder] burst EMIT (silence-trigger): '
                                f'{duration_ms:.0f}ms, speech_detected=True'
                            )
                            self._audio_q.put(burst)
                        else:
                            ConfigManager.console_print(
                                f'[recorder] burst DROPPED (silence-trigger, too short): '
                                f'{(len(recording) / sample_rate) * 1000:.0f}ms'
                            )
                        # reset for next burst
                        recording = []
                        speech_detected = False
                        silent_frame_count = 0
                        frames_to_skip = initial_skip

        # Drain any residual samples in audio_buffer before flushing
        if len(audio_buffer) > 0:
            residual = np.array(list(audio_buffer), dtype=np.int16)
            recording.extend(residual)
            ConfigManager.console_print(
                f'[recorder] drained {len(residual)} residual samples on stop'
            )

        # Flush partial burst on stop
        ConfigManager.console_print(
            f'[recorder] stop: recording={len(recording)} samples, '
            f'speech_detected={speech_detected}, '
            f'callbacks={callback_count[0]}, drains={drain_count[0]}'
        )
        if recording and speech_detected:
            burst = self._make_burst(recording, sample_rate, min_duration_ms)
            if burst is not None:
                duration_ms = (len(burst) / sample_rate) * 1000
                ConfigManager.console_print(
                    f'[recorder] burst EMIT (flush-on-stop): '
                    f'{duration_ms:.0f}ms, speech_detected={speech_detected}'
                )
                self._audio_q.put(burst)
            else:
                ConfigManager.console_print(
                    f'[recorder] burst DROPPED (flush, too short): '
                    f'{(len(recording) / sample_rate) * 1000:.0f}ms'
                )
        elif recording and not speech_detected:
            ConfigManager.console_print(
                f'[recorder] burst DROPPED (flush, no speech detected): '
                f'{(len(recording) / sample_rate) * 1000:.0f}ms'
            )

        self._audio_q.put(SENTINEL)
        self._recording_stopped.set()
        self.statusSignal.emit('idle')
        ConfigManager.console_print('RecorderWorker stopped.')
