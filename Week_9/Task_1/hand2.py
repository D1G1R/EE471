import urllib.request
import os
import cv2
import mediapipe as mp
import time
import numpy as np  # Siyah bilgi penceresi oluşturmak için eklendi
from mediapipe.framework.formats import landmark_pb2

# 1. Model İndirme (Aynı)
model_path = 'hand_landmarker.task'
if not os.path.exists(model_path):
    print("Hand Landmarker modeli indiriliyor...")
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    urllib.request.urlretrieve(url, model_path)
    print("Model başarıyla indirildi!")

# 2. MediaPipe Tasks API Tanımlamaları
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

latest_result = None

# 3. Callback Fonksiyonu
def print_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_result
    latest_result = result

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands=2, 
    result_callback=print_result)

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

print("\nKamera açılıyor... (Çıkmak için kamera penceresindeyken ESC tuşuna basın)")

# 4. Landmarker ve Kamera Döngüsü
with HandLandmarker.create_from_options(options) as landmarker:
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Kameradan görüntü alınamadı.")
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int(time.time() * 1000)
        
        landmarker.detect_async(mp_image, timestamp_ms)
        
        # Eğer el tespit edildiyse hem çizim yap hem de bilgi penceresini doldur
        if latest_result and latest_result.hand_landmarks and latest_result.handedness:
            # --- 1. KAMERA ÜZERİNE ÇİZİM KISMI ---
            for hand_landmarks in latest_result.hand_landmarks:
                hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                hand_landmarks_proto.landmark.extend([
                    landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) 
                    for landmark in hand_landmarks
                ])
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks_proto,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
                )

            # --- 2. CANLI VERİ PENCERESİ KISMI ---
            # 500x450 boyutunda siyah bir arkaplan oluşturuyoruz
            info_window = np.zeros((500, 450, 3), dtype=np.uint8)
            
            # Ekrana yazdıracağımız verileri çekiyoruz (Sadece ilk eli ve bilek noktasını alıyoruz)
            handedness_data = latest_result.handedness[0][0]
            landmark_0 = latest_result.hand_landmarks[0][0]
            
            # İstediğin formattaki metin satırları
            text_lines = [
                "HandLandmarkerResult:",
                "  Handedness:",
                f"    score: {handedness_data.score:.5f}",
                f"    category: {handedness_data.category_name}",
                "  Landmarks (Wrist - #0):",
                f"    x: {landmark_0.x:.6f}",
                f"    y: {landmark_0.y:.6f}",
                f"    z: {landmark_0.z:.6f}",
                "    ... (21 landmarks for a hand)"
            ]
            
            # Metinleri satır satır siyah pencereye yazdır
            for i, line in enumerate(text_lines):
                cv2.putText(info_window, line, (15, 40 + i*35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
            cv2.imshow('Live Data Dashboard', info_window)
            
        else:
            # Eğer kamera açık ama el görünmüyorsa veri penceresini simsiyah yap
            cv2.imshow('Live Data Dashboard', np.zeros((500, 450, 3), dtype=np.uint8))
        
        cv2.imshow('Task 1 - Hand Landmark', frame)
        
        if cv2.waitKey(5) & 0xFF == 27:
            break
            
    cap.release()
    cv2.destroyAllWindows()