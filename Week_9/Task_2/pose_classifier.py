import sys
import cv2
import mediapipe as mp

def main():
    # Komut satırından resim yolunu al
    if len(sys.argv) < 2:
        print("Kullanım: python pose_classifier.py <resim_yolu>")
        sys.exit(1)
        
    image_path = sys.argv[1]
    
    # MediaPipe Pose modelini başlat
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

    # Resmi OpenCV ile oku
    image = cv2.imread(image_path)
    if image is None:
        print("None")
        sys.exit(1)

    # OpenCV görüntüyü BGR okur, MediaPipe RGB bekler
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resmi analiz et
    results = pose.process(image_rgb)

    # Eğer insan pozu bulunamazsa
    if not results.pose_landmarks:
        print("None")
        sys.exit(0)

    landmarks = results.pose_landmarks.landmark

    # Omuz ve bilek y koordinatlarını al (y=0 resmin en üstüdür, bu yüzden küçük y değeri daha yukardadır)
    left_shoulder_y = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
    left_wrist_y = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y

    right_shoulder_y = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y
    right_wrist_y = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y

    # Mantık: Bilek, omuzdan daha yukarıdaysa (y değeri daha küçükse) kol havadadır
    left_up = left_wrist_y < left_shoulder_y
    right_up = right_wrist_y < right_shoulder_y

    # İstenen çıktıları yazdır
    if left_up and right_up:
        print("both")
    elif left_up:
        print("left")
    elif right_up:
        print("right")
    else:
        print("None")

if __name__ == "__main__":
    main()