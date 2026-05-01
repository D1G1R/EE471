import urllib.request
import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

# 1. Modeli indir (Klasörde yoksa Google'dan çeker)
model_path = 'face_landmarker.task'
if not os.path.exists(model_path):
    print("Face Landmarker modeli indiriliyor...")
    url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    urllib.request.urlretrieve(url, model_path)
    print("Model indirildi!")

# 2. Test için örnek bir resim kullan
image_path = 'kanye.png'
if not os.path.exists(image_path):
    print(f"Uyarı: {image_path} resmi bulunamadı, lutfen klasore bir yuz resmi ekleyin.")

# 3. MediaPipe Face Landmarker ayarları
base_options = python.BaseOptions(model_asset_path=model_path)

# Sitedeki örneğe uygun olarak blendshape ve transformation matrisleri true veriyoruz
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=True)

# 4. Landmarker'ı ayarlarla oluştur
with vision.FaceLandmarker.create_from_options(options) as landmarker:

    if os.path.exists(image_path):
        # 5. Resmi MediaPipe formatına dönüştür
        mp_image = mp.Image.create_from_file(image_path)
        
        # 6. Yüz tespitini gerçekleştir
        face_landmarker_result = landmarker.detect(mp_image)
        
        # 7. Çıktıyı tam istenilen formatta terminale yazdır:
        print("FaceLandmarkerResult:")
        
        # Face Landmarks (görsel kirlilik olmaması için sadece ilk iki noktayı ve sayıyı yazdırıyoruz - doc öyle gösteriyor)
        if face_landmarker_result.face_landmarks:
            print("  face_landmarks:")
            for face_idx, face_landmarks in enumerate(face_landmarker_result.face_landmarks):
                # İlk iki landmak'ı örneklemek için gösteriyoruz (dokümantasyondaki gibi)
                for i in range(2):
                    if i < len(face_landmarks):
                        landmark = face_landmarks[i]
                        print(f"    NormalizedLandmark #{i}:")
                        print(f"      x: {landmark.x}")
                        print(f"      y: {landmark.y}")
                        print(f"      z: {landmark.z}")
                print(f"    ... ({len(face_landmarks)} landmarks for each face)")
        
        # Face Blendshapes
        if face_landmarker_result.face_blendshapes:
            print("  face_blendshapes:")
            for face_idx, blendshapes in enumerate(face_landmarker_result.face_blendshapes):
                # 4 adet blendshape örneği
                for i in range(min(4, len(blendshapes))):
                    shape = blendshapes[i]
                    print(f"    {shape.category_name}: {shape.score}")
                print(f"    ... ({len(blendshapes)} blendshapes for each face)")
                
        # Facial Transformation Matrixes
        if face_landmarker_result.facial_transformation_matrixes:
            print("  facial_transformation_matrixes:")
            for face_idx, matrix_list in enumerate(face_landmarker_result.facial_transformation_matrixes):
                # Dokümandaki gibi ilk iki satırı gösterilim
                print(f"    [{matrix_list[0,0]:.8e}, {matrix_list[0,1]:.8e}, {matrix_list[0,2]:.8e}, {matrix_list[0,3]:.8e}]")
                print(f"    [{matrix_list[1,0]:.8e}, {matrix_list[1,1]:.8e}, {matrix_list[1,2]:.8e}, {matrix_list[1,3]:.8e}]")
                print("    ...")
                
        # 8. Çıktıyı OpenCV ile görselleştir:
        # mp_image RGB formatındadır, OpenCV için BGR'ye çevirip kopyalıyoruz
        annotated_image = np.copy(mp_image.numpy_view())
        annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)

        if face_landmarker_result.face_landmarks:
            for face_landmarks in face_landmarker_result.face_landmarks:
                face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                face_landmarks_proto.landmark.extend([
                    landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
                ])

                # Yüz ağını (tesselation) çiz
                mp.solutions.drawing_utils.draw_landmarks(
                    image=annotated_image,
                    landmark_list=face_landmarks_proto,
                    connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style())
                
                # Yüz kontürlerini çiz
                mp.solutions.drawing_utils.draw_landmarks(
                    image=annotated_image,
                    landmark_list=face_landmarks_proto,
                    connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_contours_style())
                
                # Göz bebeklerini (irises) çiz
                mp.solutions.drawing_utils.draw_landmarks(
                    image=annotated_image,
                    landmark_list=face_landmarks_proto,
                    connections=mp.solutions.face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_iris_connections_style())
        
        cv2.imshow("Face Landmarker", annotated_image)
        # Resmi görebilmek için herhangi bir tuşa basılmasını bekle
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    else:
        print("Hedef resim bulunamadığı için işlem iptal edildi.")