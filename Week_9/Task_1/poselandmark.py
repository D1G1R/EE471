import urllib.request
import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

# 1. Modeli indir (Klasörde yoksa Google'dan çeker)
model_path = 'pose_landmarker_heavy.task'
if not os.path.exists(model_path):
    print("Pose Landmarker modeli indiriliyor...")
    url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
    urllib.request.urlretrieve(url, model_path)
    print("Model indirildi!")

# 2. Test için örnek bir resim kullan
image_path = 'pose-2.jpg' 
if not os.path.exists(image_path):
    print("Örnek test resmi indiriliyor...")
    img_url = "https://storage.googleapis.com/mediapipe-assets/woman_doing_yoga.jpg"
    urllib.request.urlretrieve(img_url, image_path)
    print("Test resmi indirildi!")

# 3. MediaPipe Pose Landmarker ayarları 
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    output_segmentation_masks=False)

# Yardımcı çizim fonksiyonu
def draw_landmarks_on_image(rgb_image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  # Her bir tespit edilen poz için döngü oluştur
  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]

    # Cizim için dönüştür
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])
    
    # Yer işaretlerini (nokta ve çizgiler) çiz
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      pose_landmarks_proto,
      solutions.pose.POSE_CONNECTIONS,
      solutions.drawing_styles.get_default_pose_landmarks_style())
      
  return annotated_image

# 4. Landmarker'ı ayarlarla oluştur
with vision.PoseLandmarker.create_from_options(options) as landmarker:

    # 5. Resmi MediaPipe formatına dönüştür
    if os.path.exists(image_path):
        mp_image = mp.Image.create_from_file(image_path)
        
        # 6. Pose algılamasını gerçekleştir
        print("Pose tespiti yapılıyor...\n")
        detection_result = landmarker.detect(mp_image)
        
        # 7. Tespit edilen noktaları görselleştir
        annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), detection_result)
        
        # Resim Rgb olduğundan OpenCV için BGR formatına çevir
        annotated_image_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)

        cv2.imshow("Task - Static Image Pose Landmarker", annotated_image_bgr)
        
        print("\nÇıkmak için resim penceresi üzerindeyken klavyeden herhangi bir tuşa basın...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    else:
        print(f"{image_path} bulunamadı.")