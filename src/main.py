import os
import sys
import time
from audioplayer import AudioPlayer
from pynput.keyboard import Controller
from PyQt5.QtCore import QObject, QProcess
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication, QSystemTrayIcon, QMenu, QAction, QMessageBox

import queue
from threading import Event
from key_listener import KeyListener
from recorder_worker import RecorderWorker
from transcriber_worker import TranscriberWorker
from typer_worker import TyperWorker
from ui.settings_window import SettingsWindow
from ui.status_window import StatusWindow
from ui.calibration_window import CalibrationWindow
from transcription import create_local_model
from input_simulation import InputSimulator
from utils import ConfigManager


class WhisperWriterApp(QObject):
    def __init__(self):
        """
        Initialize the application, opening settings window if no configuration file is found.
        """
        super().__init__()
        self.app = QApplication(sys.argv)
        self.app.setWindowIcon(QIcon(os.path.join('assets', 'ww-logo.png')))
        self.app.setQuitOnLastWindowClosed(False)

        ConfigManager.initialize()

        self.settings_window = SettingsWindow()
        self.settings_window.settings_closed.connect(self.on_settings_closed)
        self.settings_window.settings_saved.connect(self.restart_app)
        self.calibration_window = CalibrationWindow()

        if ConfigManager.config_file_exists():
            self.initialize_components()
        else:
            print('No valid configuration file found. Opening settings window...')
            self.settings_window.show()

    def initialize_components(self):
        """
        Initialize the components of the application.
        """
        self.input_simulator = InputSimulator()

        self.key_listener = KeyListener()
        self.key_listener.add_callback("on_activate", self.on_activation)
        self.key_listener.add_callback("on_deactivate", self.on_deactivation)

        model_options = ConfigManager.get_config_section('model_options')
        model_path = model_options.get('local', {}).get('model_path')
        self.local_model = create_local_model() if not model_options.get('use_api') else None

        self._recorder = None
        self._transcriber = None
        self._typer = None

        if not ConfigManager.get_config_value('misc', 'hide_status_window'):
            self.status_window = StatusWindow()

        self.create_tray_icon()
        self.key_listener.start()

    def create_tray_icon(self):
        """
        Create the system tray icon and its context menu.
        """
        self.tray_icon = QSystemTrayIcon(QIcon(os.path.join('assets', 'ww-logo.png')), self.app)

        tray_menu = QMenu()

        settings_action = QAction('Open Settings', self.app)
        settings_action.triggered.connect(self.settings_window.show)
        tray_menu.addAction(settings_action)

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
        if self.key_listener:
            self.key_listener.stop()
        if self.input_simulator:
            self.input_simulator.cleanup()

    def exit_app(self):
        """
        Exit the application.
        """
        self.cleanup()
        QApplication.quit()

    def restart_app(self):
        """Restart the application to apply the new settings."""
        self.cleanup()
        QApplication.quit()
        QProcess.startDetached(sys.executable, sys.argv)

    def on_settings_closed(self):
        """
        If settings is closed without saving on first run, initialize the components with default values.
        """
        if not os.path.exists(os.path.join('src', 'config.yaml')):
            QMessageBox.information(
                self.settings_window,
                'Using Default Values',
                'Settings closed without saving. Default values are being used.'
            )
            self.initialize_components()

    def on_activation(self):
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
            self._recorder.statusSignal.connect(self.status_window.updateStatus)
            self._recorder.pitchSignal.connect(self.status_window.updatePitch)
            self._typer.statusSignal.connect(self.status_window.updateStatus)
            self.status_window.closeSignal.connect(self.stop_pipeline)

        self._typer.finished.connect(self._on_pipeline_finished)

        self._recorder.start()
        self._transcriber.start()
        self._typer.start()

    def stop_pipeline(self):
        if self._recorder and self._recorder.isRunning():
            self._recorder.stop()

    def _on_pipeline_finished(self):
        if ConfigManager.get_config_value('misc', 'noise_on_completion'):
            AudioPlayer(os.path.join('assets', 'beep.wav')).play(block=True)

        if ConfigManager.get_config_value('recording_options', 'recording_mode') == 'continuous':
            self.start_pipeline()
        else:
            self.key_listener.start()

    def run(self):
        """
        Start the application.
        """
        sys.exit(self.app.exec_())


if __name__ == '__main__':
    app = WhisperWriterApp()
    app.run()
