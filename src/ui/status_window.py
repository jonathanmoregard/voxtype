import sys
import os
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QPixmap, QPainter, QBrush, QColor, QPen
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel


class StatusWindow(QWidget):
    closeSignal = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setFixedSize(120, 120)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)

        self.icon_label = QLabel()
        self.icon_label.setAlignment(Qt.AlignCenter)
        microphone_path = os.path.join('assets', 'microphone.png')
        pixmap = QPixmap(microphone_path).scaled(80, 80, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.icon_label.setPixmap(pixmap)
        layout.addWidget(self.icon_label)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(QBrush(QColor(255, 255, 255, 160)))
        painter.setPen(QPen(QColor(60, 60, 60, 180), 2))
        painter.drawRoundedRect(1, 1, self.width() - 2, self.height() - 2, 28, 28)

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

    @pyqtSlot(str)
    def updateStatus(self, status):
        if status == 'recording':
            self.show()
        elif status in ('typing', 'idle', 'error', 'cancel'):
            self.hide()
