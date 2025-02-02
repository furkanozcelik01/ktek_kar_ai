import sys
import cv2
import mediapipe as mp
from scipy.spatial import distance
import numpy as np
from PySide6.QtWidgets import QApplication, QMainWindow, QLabel
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import QTimer, Qt
from main_ui import Ui_MainWindow
import res_rc


# Ekran çözünürlüğü için global sınıf
class ScreenConfig:
    # Varsayılan tasarım çözünürlüğü
    DESIGN_WIDTH = 1920
    DESIGN_HEIGHT = 1080

    @classmethod
    def scale(cls, size):
        """Boyutu ekran oranına göre ölçeklendir"""
        scale_factor = min(cls.current_width / cls.DESIGN_WIDTH,
                           cls.current_height / cls.DESIGN_HEIGHT)
        return int(size * scale_factor)

    @classmethod
    def scale_point(cls, x, y):
        """Koordinat noktasını ekran oranına göre ölçeklendir"""
        scale_factor = min(cls.current_width / cls.DESIGN_WIDTH,
                           cls.current_height / cls.DESIGN_HEIGHT)
        return (int(x * scale_factor), int(y * scale_factor))


# Mediapipe sabitleri
mp_face_mesh = mp.solutions.face_mesh
EAR_THRESHOLD = 0.2
LEFT_EYE_INDEX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDEX = [362, 385, 387, 263, 373, 380]
HEAD_POSE_INDEX = [33, 263, 1, 61, 291, 199]


class ConcentrationApp(QMainWindow):
    def __init__(self):
        super().__init__()

        # Ekran boyutlarını al ve global değişkenleri ayarla
        screen = QApplication.primaryScreen().geometry()
        ScreenConfig.current_width = screen.width()
        ScreenConfig.current_height = screen.height()

        # UI dosyasını yükle
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.resize_ui_components()

        # Kamera frame'i için QLabel oluştur
        self.camera_label = QLabel(self.ui.ekran)
        self.camera_label.setGeometry(self.ui.ekran.rect())

        # Face Mesh başlat
        self.face_mesh = mp_face_mesh.FaceMesh(
            refine_landmarks=True,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Kamera ayarları
        self.cap = cv2.VideoCapture(0)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # 30ms refresh rate

        # Başlangıç değerleri
        self.score = 100
        self.status = "Normal"
        self.update_labels()

        # Push Button'a tıklama olayını bağla
        self.ui.pushButton.clicked.connect(self.close)

    def resize_ui_components(self):
        """UI bileşenlerinin boyutlarını ekran boyutuna göre ölçeklendir"""
        # Arkaplan etiketini tam ekrana ayarla ve tek resim olarak göster
        self.ui.arkaplan.setGeometry(0, 0, ScreenConfig.current_width, ScreenConfig.current_height)
        self.ui.arkaplan.setScaledContents(True)

        # Ana pencereyi tam ekran yap
        self.setFixedSize(ScreenConfig.current_width, ScreenConfig.current_height)

        # Merkezi widget'ı tam ekran yap
        self.ui.centralwidget.setGeometry(0, 0, ScreenConfig.current_width, ScreenConfig.current_height)

        # Kamera ekranını yeniden boyutlandır
        ekran_x, ekran_y = ScreenConfig.scale_point(1200, 120)
        ekran_width = ScreenConfig.scale(501)
        ekran_height = ScreenConfig.scale(261)
        self.ui.ekran.setGeometry(ekran_x, ekran_y, ekran_width, ekran_height)

        # Logo boyutunu ayarla
        logo_width = ScreenConfig.scale(181)
        logo_height = ScreenConfig.scale(151)
        logo_x = ekran_x + (ekran_width - logo_width) // 2
        logo_y = ekran_y + ekran_height + ScreenConfig.scale(50)
        self.ui.info_icon.setGeometry(logo_x, logo_y, logo_width, logo_height)

        # Etiketleri yeniden boyutlandır
        label_width = ScreenConfig.scale(200)
        label_height = ScreenConfig.scale(40)
        value_width = ScreenConfig.scale(300)

        base_x, base_y = ScreenConfig.scale_point(130, 360)
        spacing = ScreenConfig.scale(60)

        # Hız etiketi
        self.ui.hiz_label.setGeometry(base_x, base_y, label_width, label_height)
        self.ui.hiz_label_deger.setGeometry(
            base_x + label_width + ScreenConfig.scale(20),
            base_y, value_width, label_height
        )

        # Skor etiketi
        self.ui.skor_label.setGeometry(base_x, base_y + spacing, label_width, label_height)
        self.ui.skor_label_deger.setGeometry(
            base_x + label_width + ScreenConfig.scale(20),
            base_y + spacing, value_width, label_height
        )

        # Durum etiketi
        self.ui.durum_label.setGeometry(base_x, base_y + spacing * 2, label_width, label_height)
        self.ui.durum_label_deger.setGeometry(
            base_x + label_width + ScreenConfig.scale(20),
            base_y + spacing * 2, value_width, label_height
        )

        # Font boyutlarını ayarla
        font_size = ScreenConfig.scale(26)
        for label in [self.ui.hiz_label, self.ui.skor_label, self.ui.durum_label,
                      self.ui.hiz_label_deger, self.ui.skor_label_deger, self.ui.durum_label_deger]:
            font = label.font()
            font.setPointSize(font_size)
            label.setFont(font)

    def calculate_ear(self, eye_landmarks):
        """Eye Aspect Ratio (EAR) hesaplar."""
        A = distance.euclidean(eye_landmarks[1], eye_landmarks[5])
        B = distance.euclidean(eye_landmarks[2], eye_landmarks[4])
        C = distance.euclidean(eye_landmarks[0], eye_landmarks[3])
        return (A + B) / (2.0 * C)

    def get_eye_landmarks(self, face_landmarks, frame, eye_indices):
        """Verilen göz için landmark koordinatlarını çıkarır."""
        return [
            (int(face_landmarks.landmark[i].x * frame.shape[1]),
             int(face_landmarks.landmark[i].y * frame.shape[0]))
            for i in eye_indices
        ]

    def analyze_head_pose(self, face_landmarks, frame):
        """Baş pozisyonunu analiz eder ve açıları döndürür."""
        face_2d = []
        face_3d = []

        for idx, lm in enumerate(face_landmarks.landmark):
            if idx in HEAD_POSE_INDEX:
                x, y = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
                face_2d.append([x, y])
                face_3d.append([x, y, lm.z])

        if len(face_2d) == len(HEAD_POSE_INDEX):
            face_2d = np.array(face_2d, dtype=np.float64)
            face_3d = np.array(face_3d, dtype=np.float64)

            focal_length = 1 * frame.shape[1]
            cam_matrix = np.array([[focal_length, 0, frame.shape[1] / 2],
                                   [0, focal_length, frame.shape[0] / 2],
                                   [0, 0, 1]])
            distortion_matrix = np.zeros((4, 1), dtype=np.float64)

            success, rotation_vec, translation_vec = cv2.solvePnP(
                face_3d, face_2d, cam_matrix, distortion_matrix)
            if success:
                rmat, jac = cv2.Rodrigues(rotation_vec)
                angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
                return angles[0] * 360, angles[1] * 360
        return None, None

    def update_score(self, ear, x_angle, y_angle):
        """Puanı göz kırpma ve baş pozisyonuna göre günceller."""
        if ear < EAR_THRESHOLD:
            self.score -= 1  # Göz kapalı
        else:
            self.score += 1  # Göz açık

        if y_angle is not None:
            if abs(y_angle) > 10:  # Sağa veya sola bakıyor
                self.score -= 2
            elif abs(x_angle) > 10:  # Yukarı veya aşağı bakıyor
                self.score -= 2
            else:  # Öne bakıyor
                self.score += 2

        self.score = max(0, min(self.score, 100))  # Puanı 0-100 arasında sınırla

    def update_status(self):
        """Puana göre durumu günceller"""

        if self.score >= 67:
            self.status = "Yüksek Konsantrasyon"

            self.ui.info_icon.setPixmap(QPixmap(":/images/star.png"))
        elif 33 <= self.score < 67:
            self.status = "Orta Konsantrasyon"
            self.ui.info_icon.setPixmap(QPixmap(":/images/info.png"))
        else:
            self.status = "Düşük Konsantrasyon"
            self.ui.info_icon.setPixmap(QPixmap(":/images/warning.png"))

    def update_labels(self):
        """UI etiketlerini günceller"""
        self.ui.skor_label_deger.setText(str(self.score))
        self.ui.durum_label_deger.setText(self.status)

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # Görüntüyü işle
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Göz analizi
                    left_eye = self.get_eye_landmarks(face_landmarks, frame, LEFT_EYE_INDEX)
                    right_eye = self.get_eye_landmarks(face_landmarks, frame, RIGHT_EYE_INDEX)
                    avg_ear = (self.calculate_ear(left_eye) + self.calculate_ear(right_eye)) / 2.0

                    # Baş pozisyonu analizi
                    x_angle, y_angle = self.analyze_head_pose(face_landmarks, frame)

                    # Skor ve durum güncelleme
                    self.update_score(avg_ear, x_angle, y_angle)
                    self.update_status()
                    self.update_labels()

            # Frame'i UI'a yerleştir
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            q_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            scaled_pixmap = QPixmap.fromImage(q_img).scaled(
                self.ui.ekran.size(),
                Qt.KeepAspectRatio,  # En boy oranını koru
                Qt.SmoothTransformation  # Daha yumuşak görüntü için
            )
            self.camera_label.setPixmap(scaled_pixmap)

    def closeEvent(self, event):
        self.cap.release()
        self.face_mesh.close()
        event.accept()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()


def main():
    app = QApplication(sys.argv)
    window = ConcentrationApp()
    window.showFullScreen()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
