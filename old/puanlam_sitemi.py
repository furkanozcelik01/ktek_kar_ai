import cv2
import mediapipe as mp
from scipy.spatial import distance
import time
import numpy as np

# Mediapipe modülleri
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# EAR (Eye Aspect Ratio) fonksiyonu
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# EAR eşik değeri
EAR_THRESHOLD = 0.2

# Mediapipe Face Mesh başlat
face_mesh = mp_face_mesh.FaceMesh(
    refine_landmarks=True, max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5
)

# Landmark indeksleri
LEFT_EYE_INDEX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDEX = [362, 385, 387, 263, 373, 380]
HEAD_POSE_INDEX = [33, 263, 1, 61, 291, 199]

# Kamera başlat
cap = cv2.VideoCapture("http://192.168.1.105:4747/video")

# Puanlama parametreleri
score = 100
start_time = time.time()
pose_penalty_start = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Görüntüyü RGB'ye çevir
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Sol ve sağ göz landmark'larını al
            left_eye = [
                (int(face_landmarks.landmark[i].x * frame.shape[1]), int(face_landmarks.landmark[i].y * frame.shape[0]))
                for i in LEFT_EYE_INDEX]
            right_eye = [
                (int(face_landmarks.landmark[i].x * frame.shape[1]), int(face_landmarks.landmark[i].y * frame.shape[0]))
                for i in RIGHT_EYE_INDEX]

            # EAR hesapla
            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            avg_ear = (left_ear + right_ear) / 2.0

            # Göz durumu kontrol et
            if avg_ear < EAR_THRESHOLD:
                score -= 1  # Göz kapalı ise puan azalt
            else:
                score += 1  # Göz açık ise puan artır

            # Baş pozisyonunu al
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

                success, rotation_vec, translation_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, distortion_matrix)
                if success:
                    rmat, jac = cv2.Rodrigues(rotation_vec)
                    angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

                    x = angles[0] * 360
                    y = angles[1] * 360

                    # Baş pozisyonuna göre puan kontrolü
                     # Burada y eksenine göre puanlama yapılıyor
            if y < -10:
                text = "Looking Left"
                score -= 1  # Puan düşüşü
            elif y > 10:
                text = "Looking Right"
                score -= 1  # Puan düşüşü
            elif x < -10:
                text = "Looking Down"
                score -= 1  # Puan düşüşü
            elif x > 10:
                text = "Looking Up"
                score -= 1  # Puan düşüşü
            else:
                text = "Forward"
                score += 1  # Puan artışı
    # Puanı 0 ile 100 arasında tut
    score = max(0, min(score, 100))

    # Renk kodlama: puana göre
    if score >= 67:
        color = (0, 255, 0)  # Yeşil (Yüksek puan - Konsantrasyon iyi)
        text = "Yuksek Konsantrasyon"
    elif 33 <= score < 67:
        color = (0, 255, 255)  # Sarı (Orta puan - Konsantrasyon orta)
        text = "Orta Konsantrasyon"
    else:
        color = (0, 0, 255)  # Kırmızı (Düşük puan - Konsantrasyon düşük)
        text = "Dusuk Konsantrasyon"

    # Puan ve renk bilgisi çerçeveye yazdır
    cv2.putText(frame, f"Score: {score}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.putText(frame, text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Görüntüyü göster
    cv2.imshow("Konsantrasyon Analizi", frame)

    # Her dakika puanı sıfırla
    if time.time() - start_time > 60:
        start_time = time.time()
        score = 100

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
