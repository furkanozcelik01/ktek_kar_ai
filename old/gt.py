import cv2
import mediapipe as mp
from scipy.spatial import distance
import time

# Mediapipe yüz mesh modülleri
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils


# Göz açıklığını ölçmek için EAR (Eye Aspect Ratio) fonksiyonu
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

# Göz landmark'ları
LEFT_EYE_INDEX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDEX = [362, 385, 387, 263, 373, 380]

# Puanlama parametreleri
score = 100
start_time = time.time()

# Kamera başlat
cap = cv2.VideoCapture("http://192.168.1.105:4747/video")

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

            # Puanı 0 ile 100 arasında tut
            score = max(0, min(score, 100))

            # Göz çevresine çizgi çiz
            for point in left_eye + right_eye:
                cv2.circle(frame, point, 2, (255, 0, 0), -1)

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
    cv2.imshow("Goz Durumu ve Puanlama", frame)

    # Her dakika puanı sıfırla
    if time.time() - start_time > 60:
        start_time = time.time()
        score = 100

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
