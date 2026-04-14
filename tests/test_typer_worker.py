import queue
import sys
import time
from threading import Event
from unittest.mock import MagicMock

from PyQt5.QtCore import QCoreApplication
from PyQt5.QtWidgets import QApplication

from pipeline import SENTINEL

app = QApplication.instance() or QApplication(sys.argv)


def test_typer_waits_for_recording_stopped_before_typing():
    from typer_worker import TyperWorker

    text_q = queue.Queue()
    recording_stopped = Event()
    mock_simulator = MagicMock()

    text_q.put('hello ')
    text_q.put(SENTINEL)

    typed = []
    mock_simulator.typewrite.side_effect = typed.append

    worker = TyperWorker(text_q, mock_simulator, recording_stopped)
    worker.start()
    time.sleep(0.1)
    assert typed == [], "Should not have typed before recording_stopped"

    recording_stopped.set()
    worker.wait(2000)
    assert typed == ['hello ']


def test_typer_emits_typing_then_idle():
    from typer_worker import TyperWorker

    text_q = queue.Queue()
    recording_stopped = Event()
    mock_simulator = MagicMock()
    recording_stopped.set()

    text_q.put('word ')
    text_q.put(SENTINEL)

    statuses = []
    worker = TyperWorker(text_q, mock_simulator, recording_stopped)
    worker.statusSignal.connect(statuses.append)
    worker.start()
    worker.wait(2000)
    QCoreApplication.processEvents()

    assert 'typing' in statuses
    assert statuses[-1] == 'idle'


def test_typer_idle_with_empty_queue():
    from typer_worker import TyperWorker

    text_q = queue.Queue()
    recording_stopped = Event()
    mock_simulator = MagicMock()
    recording_stopped.set()

    text_q.put(SENTINEL)

    statuses = []
    worker = TyperWorker(text_q, mock_simulator, recording_stopped)
    worker.statusSignal.connect(statuses.append)
    worker.start()
    worker.wait(2000)
    QCoreApplication.processEvents()

    mock_simulator.typewrite.assert_not_called()
    assert statuses == ['idle']
