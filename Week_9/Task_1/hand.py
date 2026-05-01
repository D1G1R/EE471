import urllib.request
import os
import cv2
import mediapipe as mp
import time
from mediapipe.framework.formats import landmark_pb2

# 1. Hand Landmarker .task Modelini İndir
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

# Asenkron işlemden dönecek sonuçları tutacağımız global değişken
latest_result = None

# 3. Callback Fonksiyonu (Linkteki yapının kalbi)
# Her kare analiz edildiğinde bu fonksiyon arka planda otomatik tetiklenir
def print_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_result
    latest_result = result

# Options ayarlarını linkteki gibi LIVE_STREAM modunda yapıyoruz
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)

# Çizim araçları
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

print("\nKamera açılıyor... (Çıkmak için kamera penceresindeyken ESC tuşuna basın)")

# 4. Landmarker'ı Başlat ve Kamera Döngüsüne Gir
with HandLandmarker.create_from_options(options) as landmarker:
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Kameradan görüntü alınamadı.")
            break

        # Görüntüyü ayna efektiyle çevir ve RGB formatına sok
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # OpenCV görüntüsünü MediaPipe'ın beklediği Image nesnesine dönüştür
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Linkteki yapı gereği asenkron tespit için milisaniye cinsinden zaman damgası vermeliyiz
        timestamp_ms = int(time.time() * 1000)
        
        # Analizi asenkron olarak tetikle (Sonucu yukarıdaki callback fonksiyonuna atacak)
        landmarker.detect_async(mp_image, timestamp_ms)
        
        # Eğer callback'e veri ulaştıysa (el tespit edildiyse), noktaları ekrana çiz
        if latest_result and latest_result.hand_landmarks:
            for hand_landmarks in latest_result.hand_landmarks:
                # Tasks API'nin yeni veri tipini eski çizim aracının anlayacağı protobuffer formatına çeviriyoruz
                hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                hand_landmarks_proto.landmark.extend([
                    landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) 
                    for landmark in hand_landmarks
                ])
                
                # Eklemleri ve bağlantıları çiz
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks_proto,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
                )
        
        # Çizim yapılmış son kareyi ekranda göster
        cv2.imshow('Task 1 - Tasks API Hand Landmark', frame)
        
        if cv2.waitKey(5) & 0xFF == 27:
            break
            
    cap.release()
    cv2.destroyAllWindows()