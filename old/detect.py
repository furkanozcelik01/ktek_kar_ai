import sys
import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from old.utils import visualize

COUNTER, FPS = 0, 0
START_TIME = time.time()
DETECTION_RESULT = None

def detect_face_orientation(face_detection_result, frame_width, frame_height, image):
    if face_detection_result.detections:
        for detection in face_detection_result.detections:
            # Bounding box merkezini bul
            bounding_box = detection.bounding_box
            center_x = bounding_box.origin_x + bounding_box.width / 2
            center_y = bounding_box.origin_y + bounding_box.height / 2

            # Merkez değerini çerçeve genişliğiyle normalize et
            normalized_center_x = center_x / frame_width

            # Bounding box'ı görüntüye çiz
            start_point = (int(bounding_box.origin_x), int(bounding_box.origin_y))
            end_point = (int(bounding_box.origin_x + bounding_box.width),
                         int(bounding_box.origin_y + bounding_box.height))
            cv2.rectangle(image, start_point, end_point, (255, 0, 0), 2)

            # Yüzün sola mı sağa mı döndüğünü kontrol et
            if normalized_center_x < 0.4:  # Çerçevenin sol tarafı
                print("Yüz sola dönük")
            elif normalized_center_x > 0.6:  # Çerçevenin sağ tarafı
                print("Yüz sağa dönük")
            else:
                print("Yüz düz bakıyor")

            # Merkeze işaret koy
            cv2.circle(image, (int(center_x), int(center_y)), 5, (0, 255, 0), -1)

def run(model: str, min_detection_confidence: float,
        min_suppression_threshold: float, camera_id: str, width: int,
        height: int) -> None:
    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    row_size = 50  # pixels
    left_margin = 24  # pixels
    text_color = (0, 0, 0)  # black
    font_size = 1
    font_thickness = 1
    fps_avg_frame_count = 10

    def save_result(result: vision.FaceDetectorResult, unused_output_image: mp.Image,
                    timestamp_ms: int):
        global FPS, COUNTER, START_TIME, DETECTION_RESULT

        # Calculate the FPS
        if COUNTER % fps_avg_frame_count == 0:
            FPS = fps_avg_frame_count / (time.time() - START_TIME)
            START_TIME = time.time()

        DETECTION_RESULT = result
        COUNTER += 1

        # Yüz yönünü algıla ve bounding box'ı görselleştir
        detect_face_orientation(result, width, height, current_frame)

    # Initialize the face detection model
    base_options = python.BaseOptions(model_asset_path=model)
    options = vision.FaceDetectorOptions(base_options=base_options,
                                         running_mode=vision.RunningMode.LIVE_STREAM,
                                         min_detection_confidence=min_detection_confidence,
                                         min_suppression_threshold=min_suppression_threshold,
                                         result_callback=save_result)
    detector = vision.FaceDetector.create_from_options(options)

    # Continuously capture images from the camera and run inference
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            sys.exit(
                'ERROR: Unable to read from webcam. Please verify your webcam settings.'
            )

        image = cv2.flip(image, 1)

        # Convert the image from BGR to RGB as required by the TFLite model.
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        # Run face detection using the model.
        detector.detect_async(mp_image, time.time_ns() // 1_000_000)

        # Show the FPS
        fps_text = 'FPS = {:.1f}'.format(FPS)
        text_location = (left_margin, row_size)
        current_frame = image
        cv2.putText(current_frame, fps_text, text_location, cv2.FONT_HERSHEY_DUPLEX,
                    font_size, text_color, font_thickness, cv2.LINE_AA)

        if DETECTION_RESULT:
            current_frame = visualize(current_frame, DETECTION_RESULT)

        cv2.imshow('face_detection', current_frame)

        # Stop the program if the ESC key is pressed.
        if cv2.waitKey(1) == 27:
            break

    detector.close()
    cap.release()
    cv2.destroyAllWindows()

def main():
    run('detector.tflite', 0.5, 0.5, "http://192.168.1.105:4747/video", 1280, 720)

if __name__ == '__main__':
    main()
