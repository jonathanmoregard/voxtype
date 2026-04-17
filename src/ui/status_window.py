import sys
import os
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QFont, QPixmap, QPainter, QBrush, QColor, QPen
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel

from utils import ConfigManager

_GREEN  = QColor(76, 175, 80, 230)
_ORANGE = QColor(255, 152, 0, 230)
_RED    = QColor(244, 67, 54, 230)


def _get_threshold():
    target   = ConfigManager.get_config_value('misc', 'pitch_target')
    unwanted = ConfigManager.get_config_value('misc', 'pitch_unwanted')
    if target is not None and unwanted is not None:
        base = (target + unwanted) / 2
        offset = ConfigManager.get_config_value('misc', 'pitch_threshold_offset') or 0.0
        return base + offset
    return None


def _pitch_to_color(hz):
    threshold = _get_threshold()
    if threshold is not None:
        return _GREEN if hz < threshold else _RED
    # fallback defaults
    if hz < 130:
        return _GREEN
    elif hz < 180:
        return _ORANGE
    return _RED


class StatusWindow(QWidget):
    closeSignal = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setFixedSize(120, 145)
        self._border_color = QColor(60, 60, 60, 180)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(4)

        self.icon_label = QLabel()
        self.icon_label.setAlignment(Qt.AlignCenter)
        microphone_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'assets', 'microphone.png')
        pixmap = QPixmap(microphone_path).scaled(80, 80, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.icon_label.setPixmap(pixmap)
        layout.addWidget(self.icon_label)

        self._threshold_label = QLabel('')
        self._threshold_label.setAlignment(Qt.AlignCenter)
        font = QFont()
        font.setPointSize(9)
        self._threshold_label.setFont(font)
        self._threshold_label.setStyleSheet('color: rgba(40,40,40,200);')
        layout.addWidget(self._threshold_label)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(QBrush(QColor(255, 255, 255, 160)))
        painter.setPen(QPen(self._border_color, 3))
        painter.drawRoundedRect(1, 1, self.width() - 2, self.height() - 2, 36, 36)

    def show(self):
        screen = QApplication.primaryScreen()
        geo = screen.geometry()
        x = (geo.width() - self.width()) // 2
        y = geo.height() - self.height() - 120
        self.move(x, y)
        super().show()

    def closeEvent(self, event):
        self.closeSignal.emit()
        super().closeEvent(event)

    def _refresh_threshold_label(self):
        threshold = _get_threshold()
        if threshold is not None:
            self._threshold_label.setText(f'{threshold:.0f} Hz')
        else:
            self._threshold_label.setText('')

    @pyqtSlot(float)
    def updatePitch(self, hz):
        self._border_color = _pitch_to_color(hz)
        self.update()

    @pyqtSlot()
    def updateThresholdOffset(self):
        self._refresh_threshold_label()

    @pyqtSlot(str)
    def updateStatus(self, status):
        if status == 'recording':
            self._border_color = QColor(60, 60, 60, 180)
            self._refresh_threshold_label()
            self.show()
        elif status in ('typing', 'idle', 'error', 'cancel'):
            self.hide()
