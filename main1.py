import cv2
import mediapipe as mp
from scipy.spatial import distance
import time
import numpy as np

# Mediapipe modülleri ve sabitler
mp_face_mesh = mp.solutions.face_mesh
EAR_THRESHOLD = 0.2
LEFT_EYE_INDEX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDEX = [362, 385, 387, 263, 373, 380]
HEAD_POSE_INDEX = [33, 263, 1, 61, 291, 199]


def initialize_face_mesh():
    """Mediapipe Face Mesh modülünü başlatır."""
    return mp_face_mesh.FaceMesh(
        refine_landmarks=True, max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5
    )


def calculate_ear(eye_landmarks):
    """Eye Aspect Ratio (EAR) hesaplar."""
    A = distance.euclidean(eye_landmarks[1], eye_landmarks[5])
    B = distance.euclidean(eye_landmarks[2], eye_landmarks[4])
    C = distance.euclidean(eye_landmarks[0], eye_landmarks[3])
    return (A + B) / (2.0 * C)


def get_eye_landmarks(face_landmarks, frame, eye_indices):
    """Verilen göz için landmark koordinatlarını çıkarır."""
    return [
        (int(face_landmarks.landmark[i].x * frame.shape[1]), int(face_landmarks.landmark[i].y * frame.shape[0]))
        for i in eye_indices
    ]


def analyze_head_pose(face_landmarks, frame):
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

        success, rotation_vec, translation_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, distortion_matrix)
        if success:
            rmat, jac = cv2.Rodrigues(rotation_vec)
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
            return angles[0] * 360, angles[1] * 360
    return None, None


def update_score(ear, x_angle, y_angle, current_score):
    """Puanı göz kırpma ve baş pozisyonuna göre günceller."""
    if ear < EAR_THRESHOLD:
        current_score -= 1  # Göz kapalı
    else:
        current_score += 1  # Göz açık

    if y_angle is not None:
        if y_angle < -10:  # Sağa bakıyor
            current_score -= 2
        elif y_angle > 10:  # Sola bakıyor
            current_score -= 2
        elif x_angle < -10:  # Aşağı bakıyor
            current_score -= 2
        elif x_angle > 10:  # Yukarı bakıyor
            current_score -= 2
        else:  # Öne bakıyor
            current_score += 2

    return max(0, min(current_score, 100))  # Puanı 0-100 arasında sınırla


def display_frame_info(frame, score):
    """Görüntüye puan ve durum bilgisini ekler."""
    if score >= 67:
        color = (0, 255, 0)
        text = "Yuksek Konsantrasyon"
    elif 33 <= score < 67:
        color = (0, 255, 255)
        text = "Orta Konsantrasyon"
    else:
        color = (0, 0, 255)
        text = "Dusuk Konsantrasyon"

    cv2.putText(frame, f"Score: {score}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.putText(frame, text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)


def process_frame(frame, face_mesh, score):
    """Bir kareyi işler ve puanı günceller."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            left_eye = get_eye_landmarks(face_landmarks, frame, LEFT_EYE_INDEX)
            right_eye = get_eye_landmarks(face_landmarks, frame, RIGHT_EYE_INDEX)
            avg_ear = (calculate_ear(left_eye) + calculate_ear(right_eye)) / 2.0

            x_angle, y_angle = analyze_head_pose(face_landmarks, frame)
            score = update_score(avg_ear, x_angle, y_angle, score)

    display_frame_info(frame, score)
    return score


def main():
    """Ana program akışı."""
    face_mesh = initialize_face_mesh()
    cap = cv2.VideoCapture("http://192.168.1.107:4747/video")
    score = 100
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        score = process_frame(frame, face_mesh, score)

        # Her dakika puanı sıfırla
        if time.time() - start_time > 60:
            start_time = time.time()
            score = 100

        cv2.imshow("Konsantrasyon Analizi", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
