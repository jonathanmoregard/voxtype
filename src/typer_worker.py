from threading import Event
from PyQt5.QtCore import QThread, pyqtSignal

from pipeline import queue_to_generator, SENTINEL


class TyperWorker(QThread):
    statusSignal = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, text_q, input_simulator, recording_stopped: Event):
        super().__init__()
        self._text_q = text_q
        self._input_simulator = input_simulator
        self._recording_stopped = recording_stopped

    def run(self):
        self._recording_stopped.wait()
        first = True
        for text in queue_to_generator(self._text_q, sentinel=SENTINEL):
            if first:
                self.statusSignal.emit('typing')
                first = False
            self._input_simulator.typewrite(text)
        self.statusSignal.emit('idle')
        self.finished.emit()
