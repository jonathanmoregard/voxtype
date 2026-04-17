import os
import sys
import time
import tempfile
import fcntl
from audioplayer import AudioPlayer
from pynput.keyboard import Controller
from PyQt5.QtCore import QObject, QProcess, QMetaObject, Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication, QSystemTrayIcon, QMenu, QAction

import queue
from threading import Event
from key_listener import KeyListener
from recorder_worker import RecorderWorker
from transcriber_worker import TranscriberWorker
from typer_worker import TyperWorker
from ui.status_window import StatusWindow
from ui.calibration_window import CalibrationWindow
from transcription import create_local_model
from input_simulation import InputSimulator
from utils import ConfigManager

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(SRC_DIR, '..', 'assets')
LOCK_FILE = os.path.join(tempfile.gettempdir(), 'whisper-writer.lock')


def acquire_lock():
    fh = open(LOCK_FILE, 'w')
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fh.write(str(os.getpid()))
        fh.flush()
        return fh
    except OSError:
        fh.close()
        return None


class WhisperWriterApp(QObject):
    def __init__(self):
        super().__init__()
        self.app = QApplication(sys.argv)
        self.app.setWindowIcon(QIcon(os.path.join(ASSETS_DIR, 'ww-logo.png')))
        self.app.setQuitOnLastWindowClosed(False)

        ConfigManager.initialize()
        self.calibration_window = CalibrationWindow()
        self.initialize_components()

    def initialize_components(self):
        self.input_simulator = InputSimulator()

        self.key_listener = KeyListener()
        self.key_listener.add_callback("on_activate", self.on_activation)
        self.key_listener.add_callback("on_deactivate", self.on_deactivation)
        self.key_listener.add_callback("on_pitch_up", lambda: self._on_pitch_adjust(1.0))
        self.key_listener.add_callback("on_pitch_down", lambda: self._on_pitch_adjust(-1.0))

        self._recorder = None
        self._transcriber = None
        self._typer = None
        self.local_model = None

        if not ConfigManager.get_config_value('misc', 'hide_status_window'):
            self.status_window = StatusWindow()

        self.create_tray_icon()

        model_options = ConfigManager.get_config_section('model_options')
        self.local_model = create_local_model() if not model_options.get('use_api') else None

        self.key_listener.start()

    def create_tray_icon(self):
        self.tray_icon = QSystemTrayIcon(QIcon(os.path.join(ASSETS_DIR, 'ww-logo.png')), self.app)

        tray_menu = QMenu()

        if ConfigManager.get_config_value('misc', 'pitch_detection_enabled'):
            calibrate_action = QAction('Calibrate Pitch', self.app)
            calibrate_action.triggered.connect(self.calibration_window.show)
            tray_menu.addAction(calibrate_action)

        exit_action = QAction('Exit', self.app)
        exit_action.triggered.connect(self.exit_app)
        tray_menu.addAction(exit_action)

        self.tray_icon.setContextMenu(tray_menu)
        self.tray_icon.show()

    def cleanup(self):
        self.stop_pipeline()
        if getattr(self, 'key_listener', None):
            self.key_listener.stop()
        if getattr(self, 'input_simulator', None):
            self.input_simulator.cleanup()

    def exit_app(self):
        self.cleanup()
        QApplication.quit()

    def _on_pitch_adjust(self, delta: float):
        if not (self._recorder and self._recorder.isRunning()):
            return
        current = ConfigManager.get_config_value('misc', 'pitch_threshold_offset') or 0.0
        ConfigManager.set_config_value(current + delta, 'misc', 'pitch_threshold_offset')
        ConfigManager.save_config()
        if hasattr(self, 'status_window'):
            QMetaObject.invokeMethod(self.status_window, 'updateThresholdOffset', Qt.QueuedConnection)

    def on_activation(self):
        use_api = ConfigManager.get_config_value('model_options', 'use_api')
        if not hasattr(self, '_recorder') or (not use_api and self.local_model is None):
            return
        if self._recorder and self._recorder.isRunning():
            recording_mode = ConfigManager.get_config_value('recording_options', 'recording_mode')
            if recording_mode in ('press_to_toggle', 'continuous'):
                self.stop_pipeline()
            return
        self.start_pipeline()

    def on_deactivation(self):
        if ConfigManager.get_config_value('recording_options', 'recording_mode') == 'hold_to_record':
            self.stop_pipeline()

    def start_pipeline(self):
        if self._recorder and self._recorder.isRunning():
            return

        audio_q = queue.Queue()
        text_q = queue.Queue()
        recording_stopped = Event()

        self._recorder = RecorderWorker(audio_q, recording_stopped)
        self._transcriber = TranscriberWorker(audio_q, text_q, self.local_model)
        self._typer = TyperWorker(text_q, self.input_simulator, recording_stopped)

        if not ConfigManager.get_config_value('misc', 'hide_status_window'):
            self._recorder.statusSignal.connect(self._on_status)
            self._recorder.pitchSignal.connect(self.status_window.updatePitch)
            self._typer.statusSignal.connect(self._on_status)
            self.status_window.closeSignal.connect(self.stop_pipeline)

        self._typer.finished.connect(self._on_pipeline_finished)

        self._recorder.start()
        self._transcriber.start()
        self._typer.start()

    def stop_pipeline(self):
        if self._recorder and self._recorder.isRunning():
            self._recorder.stop()

    def _on_status(self, status):
        if hasattr(self, 'status_window'):
            if status == 'idle':
                self.status_window.hide()
            else:
                self.status_window.show()
            self.status_window.updateStatus(status)

    def _on_pipeline_finished(self):
        if ConfigManager.get_config_value('misc', 'noise_on_completion'):
            AudioPlayer(os.path.join(ASSETS_DIR, 'beep.wav')).play(block=True)

        if ConfigManager.get_config_value('recording_options', 'recording_mode') == 'continuous':
            self.start_pipeline()
        else:
            self.key_listener.start()

    def run(self):
        sys.exit(self.app.exec_())


if __name__ == '__main__':
    lock = acquire_lock()
    if lock is None:
        print('Another instance is already running. Exiting.')
        sys.exit(1)

    app = WhisperWriterApp()
    app.run()
