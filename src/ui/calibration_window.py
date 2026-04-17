import numpy as np
import sounddevice as sd
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout,
                              QPushButton, QLabel, QProgressBar)

from pitch_detector import PitchDetector
from utils import ConfigManager

RECORD_SECONDS = 3
SAMPLE_RATE = 16000
FRAME_SIZE = 480  # 30ms


class PitchRecorder(QThread):
    progress = pyqtSignal(int)       # 0-100
    result = pyqtSignal(object)      # float Hz or None

    def run(self):
        frames = []
        total_frames = int(RECORD_SECONDS * SAMPLE_RATE / FRAME_SIZE)

        def callback(indata, frame_count, time_info, status):
            frames.append(indata[:, 0].copy())

        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='int16',
                            blocksize=FRAME_SIZE, callback=callback):
            for i in range(total_frames):
                sd.sleep(int(FRAME_SIZE / SAMPLE_RATE * 1000))
                self.progress.emit(int((i + 1) / total_frames * 100))

        detector = PitchDetector(sample_rate=SAMPLE_RATE, hop_size=FRAME_SIZE)
        pitches = [p for f in frames for p in [detector.detect(f)] if p]
        self.result.emit(float(np.median(pitches)) if pitches else None)


class CalibrationWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Pitch Calibration')
        self.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)
        self.setFixedWidth(320)
        self._recorder = None
        self._target = ConfigManager.get_config_value('misc', 'pitch_target')
        self._unwanted = ConfigManager.get_config_value('misc', 'pitch_unwanted')
        self._offset = ConfigManager.get_config_value('misc', 'pitch_threshold_offset') or 0.0
        self._build_ui()

    def show(self):
        self._offset = ConfigManager.get_config_value('misc', 'pitch_threshold_offset') or 0.0
        self._refresh_threshold_label()
        super().show()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(20, 20, 20, 20)

        layout.addWidget(QLabel('<b>Speak for 3 seconds in each voice.</b>'))

        # Target
        target_row = QHBoxLayout()
        self._target_btn = QPushButton('Record target voice')
        self._target_btn.clicked.connect(lambda: self._start('target'))
        self._target_val = QLabel(f'{int(self._target)} Hz' if self._target else '—')
        target_row.addWidget(self._target_btn)
        target_row.addWidget(self._target_val)
        layout.addLayout(target_row)

        # Unwanted
        unwanted_row = QHBoxLayout()
        self._unwanted_btn = QPushButton('Record unwanted voice')
        self._unwanted_btn.clicked.connect(lambda: self._start('unwanted'))
        self._unwanted_val = QLabel(f'{int(self._unwanted)} Hz' if self._unwanted else '—')
        unwanted_row.addWidget(self._unwanted_btn)
        unwanted_row.addWidget(self._unwanted_val)
        layout.addLayout(unwanted_row)

        self._progress = QProgressBar()
        self._progress.setVisible(False)
        layout.addWidget(self._progress)

        self._threshold_label = QLabel(self._threshold_text())
        self._threshold_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self._threshold_label)

        # Threshold offset adjustment
        offset_row = QHBoxLayout()
        down_btn = QPushButton('↓')
        down_btn.setFixedWidth(40)
        down_btn.clicked.connect(lambda: self._adjust_offset(-1.0))
        self._offset_label = QLabel(self._offset_text())
        self._offset_label.setAlignment(Qt.AlignCenter)
        up_btn = QPushButton('↑')
        up_btn.setFixedWidth(40)
        up_btn.clicked.connect(lambda: self._adjust_offset(1.0))
        offset_row.addWidget(down_btn)
        offset_row.addWidget(self._offset_label)
        offset_row.addWidget(up_btn)
        layout.addLayout(offset_row)

        self._status = QLabel('')
        self._status.setAlignment(Qt.AlignCenter)
        layout.addWidget(self._status)

        save_btn = QPushButton('Save')
        save_btn.clicked.connect(self._save)
        layout.addWidget(save_btn)

    def _start(self, kind):
        self._recording_kind = kind
        self._set_buttons(False)
        self._progress.setValue(0)
        self._progress.setVisible(True)
        self._status.setText('Speak now...')

        self._recorder = PitchRecorder()
        self._recorder.progress.connect(self._progress.setValue)
        self._recorder.result.connect(self._on_result)
        self._recorder.start()

    def _threshold_text(self):
        if self._target is not None and self._unwanted is not None:
            base = (self._target + self._unwanted) / 2
            return f'Threshold: {base + self._offset:.0f} Hz'
        return 'Threshold: —'

    def _offset_text(self):
        sign = '+' if self._offset >= 0 else ''
        return f'Offset: {sign}{self._offset:.0f} Hz'

    def _refresh_threshold_label(self):
        self._threshold_label.setText(self._threshold_text())
        if hasattr(self, '_offset_label'):
            self._offset_label.setText(self._offset_text())

    def _adjust_offset(self, delta: float):
        self._offset += delta
        ConfigManager.set_config_value(self._offset, 'misc', 'pitch_threshold_offset')
        ConfigManager.save_config()
        self._refresh_threshold_label()

    @pyqtSlot(object)
    def _on_result(self, hz):
        self._progress.setVisible(False)
        self._set_buttons(True)
        if hz is None:
            self._status.setText('No pitch detected. Try again.')
            return
        if self._recording_kind == 'target':
            self._target = hz
            self._target_val.setText(f'{int(hz)} Hz')
        else:
            self._unwanted = hz
            self._unwanted_val.setText(f'{int(hz)} Hz')
        self._refresh_threshold_label()
        self._status.setText(f'Recorded: {int(hz)} Hz')

    def _set_buttons(self, enabled):
        self._target_btn.setEnabled(enabled)
        self._unwanted_btn.setEnabled(enabled)

    def _save(self):
        if self._target is not None:
            ConfigManager.set_config_value(self._target, 'misc', 'pitch_target')
        if self._unwanted is not None:
            ConfigManager.set_config_value(self._unwanted, 'misc', 'pitch_unwanted')
        ConfigManager.set_config_value(self._offset, 'misc', 'pitch_threshold_offset')
        ConfigManager.save_config()
        self._status.setText('Saved.')
