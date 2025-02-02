

import sys
import cv2
import numpy as np
from PySide6.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout, QHBoxLayout, QPushButton
from PySide6.QtGui import QPixmap, QImage, QPainter, QBrush, QPalette
from PySide6.QtCore import QTimer, Qt

class ConcentrationApp(QWidget):
    def __init__(self):
        super().__init__()

        self.initUI()
        self.cap = cv2.VideoCapture(0)  # Webcam aç
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(100)

    def initUI(self):
        self.setWindowTitle("Konsantrasyon Analizi")
        self.setGeometry(100, 100, 800, 450)

        # **Arka plan resmi ayarla**
        self.setAutoFillBackground(True)
        palette = self.palette()
        background = QPixmap("ert.jpg")  # **Buraya arka plan resminin yolunu ekleyin**
        palette.setBrush(QPalette.Window, QBrush(background))
        self.setPalette(palette)

        # **Sol panel (Skor, Durum, Yorum)**
        self.score_label = QLabel("SKOR: 100")
        self.status_label = QLabel("DURUM: Yüksek")
        self.comment_label = QLabel("YORUM: Odaklanmış")

        self.score_label.setStyleSheet("color: black; font-size: 16px;")
        self.status_label.setStyleSheet("color: black; font-size: 16px;")
        self.comment_label.setStyleSheet("color: black; font-size: 16px;")

        left_layout = QVBoxLayout()
        left_layout.addWidget(self.score_label)
        left_layout.addWidget(self.status_label)
        left_layout.addWidget(self.comment_label)
        left_layout.addStretch()

        # **Orta alan (Kamera)**
        self.video_label = QLabel()
        self.video_label.setFixedSize(300, 200)
        self.video_label.setStyleSheet("border: 2px solid gray;")
        
        center_layout = QVBoxLayout()
        center_layout.addWidget(self.video_label)
        center_layout.setAlignment(Qt.AlignCenter)

        # **Sağ alt köşe (Direksiyon Butonu)**
        self.exit_button = QPushButton()
        self.exit_button.setFixedSize(100, 100)
        self.exit_button.setStyleSheet("border: none; background: transparent;")
        self.exit_button.setIcon(QPixmap("erty.webp"))  # **Buraya direksiyon resminin yolunu ekleyin**
        self.exit_button.setIconSize(self.exit_button.size())
        self.exit_button.clicked.connect(self.close)

        right_layout = QVBoxLayout()
        right_layout.addWidget(self.exit_button, alignment=Qt.AlignBottom | Qt.AlignRight)

        # **Ana Layout**
        main_layout = QHBoxLayout()
        main_layout.addLayout(left_layout)
        main_layout.addLayout(center_layout)
        main_layout.addLayout(right_layout)

        self.setLayout(main_layout)

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(q_img))

    def closeEvent(self, event):
        self.cap.release()
        event.accept()

def main():
    app = QApplication(sys.argv)
    window = ConcentrationApp()
    window.showFullScreen()  # **Tam ekran aç**
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
