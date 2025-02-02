import cv2
import mediapipe as mp
import numpy as np

# MediaPipe FaceMesh modülü
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

def classify_emotion(landmarks):
    # Göz açıklığı oranları
    left_eye_ratio = np.linalg.norm(landmarks[159] - landmarks[145]) / np.linalg.norm(landmarks[33] - landmarks[133])
    right_eye_ratio = np.linalg.norm(landmarks[386] - landmarks[374]) / np.linalg.norm(landmarks[263] - landmarks[362])


    def classify_emotion(landmarks, thresholds=None):
        """
        Yüz ifadelerini sınıflandırır.

        Parametreler:
        - landmarks (numpy array): Yüz noktalarının koordinatlarını içerir.
        - thresholds (dict): Sınıflandırma için kullanılacak eşik değerlerini içerir.

        Dönüş:
        - str: Belirlenen yüz ifadesi.
        """
        if thresholds is None:
            thresholds = {
                "mouth_open_ratio": 0.6,
                "eye_open_ratio": 0.3,
                "brow_distance": 0.1,
                "sad_eye_ratio": 0.2,
                "mouth_width_ratio": 0.4,
            }

        # Göz açıklığı oranları
        left_eye_ratio = np.linalg.norm(landmarks[159] - landmarks[145]) / max(
            np.linalg.norm(landmarks[33] - landmarks[133]), 1e-6)
        right_eye_ratio = np.linalg.norm(landmarks[386] - landmarks[374]) / max(
            np.linalg.norm(landmarks[263] - landmarks[362]), 1e-6)

        # Ağız açıklığı oranları
        mouth_open_ratio = np.linalg.norm(landmarks[13] - landmarks[14]) / max(
            np.linalg.norm(landmarks[78] - landmarks[308]), 1e-6)
        mouth_width_ratio = np.linalg.norm(landmarks[291] - landmarks[61]) / max(
            np.linalg.norm(landmarks[78] - landmarks[308]), 1e-6)

        # Kaşların arası mesafe (kaş çatma durumu)
        brow_distance = np.linalg.norm(landmarks[55] - landmarks[285])

        # Yüz ifadelerini sınıflandırma
        if mouth_open_ratio > thresholds["mouth_open_ratio"] and \
                (left_eye_ratio > thresholds["eye_open_ratio"] or right_eye_ratio > thresholds["eye_open_ratio"]):
            return "Mutlu"
        elif mouth_open_ratio < thresholds["mouth_open_ratio"] / 1.5 and brow_distance < thresholds["brow_distance"]:
            return "Sinirli"
        elif left_eye_ratio < thresholds["sad_eye_ratio"] and right_eye_ratio < thresholds["sad_eye_ratio"]:
            return "Üzgün"
        elif mouth_width_ratio > thresholds["mouth_width_ratio"]:
            return "Şaşkın"
        else:
            return "Nötr"

    # Ağız açıklığı oranları
    mouth_open_ratio = np.linalg.norm(landmarks[13] - landmarks[14]) / np.linalg.norm(landmarks[78] - landmarks[308])
    mouth_width_ratio = np.linalg.norm(landmarks[291] - landmarks[61]) / np.linalg.norm(landmarks[78] - landmarks[308])

    # Kaşların arası mesafe (kaş çatma durumu)
    brow_distance = np.linalg.norm(landmarks[55] - landmarks[285])

    # Yüz ifadelerini sınıflandırma
    if mouth_open_ratio > 0.6 and (left_eye_ratio > 0.3 or right_eye_ratio > 0.3):
        return "Mutlu"
    elif mouth_open_ratio < 0.4 and brow_distance < 0.1:
        return "Sinirli"
    elif left_eye_ratio < 0.2 and right_eye_ratio < 0.2:
        return "Üzgün"
    elif mouth_width_ratio > 0.4:
        return "Şaşkın"
    else:
        return "Nötr"


# Video yakalama
cap = cv2.VideoCapture("http://192.168.1.105:4747/video")

# MediaPipe FaceMesh başlatma
with mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # BGR'den RGB'ye çevir
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # İşleme
        results = face_mesh.process(rgb_frame)

        # Yüz tespiti varsa
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Noktaları çıkar
                landmarks = np.array([(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark])

                # Yüz ifadelerini sınıflandır
                emotion = classify_emotion(landmarks)

                # İfadeyi ekranda göster
                cv2.putText(frame, emotion, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                # Yüzü çiz
                mp_drawing.draw_landmarks(
                    frame,
                    face_landmarks,
                    mp_face_mesh.FACEMESH_TESSELATION,  # Güncel bağlantılar
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1)
                )

        # Çıktıyı göster
        cv2.imshow('Emotion Detection', frame)

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
