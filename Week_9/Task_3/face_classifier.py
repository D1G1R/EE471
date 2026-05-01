import sys
import cv2
import mediapipe as mp

def main():
    # Komut satırından argüman olarak resim yolunu al
    if len(sys.argv) < 2:
        print("Kullanım: python face_classifier.py <resim_yolu>")
        sys.exit(1)
        
    image_path = sys.argv[1]
    
    # MediaPipe Face Mesh modelini başlat
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

    # Resmi oku ve formatını dönüştür
    image = cv2.imread(image_path)
    if image is None:
        print("Resim bulunamadı.")
        sys.exit(1)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)

    # Yüz bulunamazsa çık
    if not results.multi_face_landmarks:
        print("Yüz bulunamadı")
        sys.exit(0)

    # İlk tespit edilen yüzün referans noktalarını (landmarks) al
    landmarks = results.multi_face_landmarks[0].landmark

    # MediaPipe Face Mesh indeksleri:
    # 1: Burun ucu
    # 234: Yüzün sol kenarı (resimdeki sol)
    # 454: Yüzün sağ kenarı (resimdeki sağ)
    nose_x = landmarks[1].x
    left_side_x = landmarks[454].x  
    right_side_x = landmarks[234].x 

    # Burun ucunun sol ve sağ kenarlara olan yatay mesafesini hesapla
    dist_to_left = abs(nose_x - left_side_x)
    dist_to_right = abs(right_side_x - nose_x)

    # Matematiksel oran: Sol mesafe / Sağ mesafe
    if dist_to_right == 0:  # Sıfıra bölünme hatasını engellemek için
        dist_to_right = 0.0001
        
    ratio = dist_to_left / dist_to_right

    if ratio < 0.7:
        print("left")
    elif ratio > 1.4:
        print("right")
    else:
        print("straight")

if __name__ == "__main__":
    main()