import urllib.request
import os
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 1. Modeli indir (Klasörde yoksa Google'dan çeker)
model_path = 'efficientnet_lite0.tflite'
if not os.path.exists(model_path):
    print("Model indiriliyor...")
    url = "https://storage.googleapis.com/mediapipe-models/image_classifier/efficientnet_lite0/float32/1/efficientnet_lite0.tflite"
    urllib.request.urlretrieve(url, model_path)
    print("Model indirildi!")

# 2. Test için örnek bir resim indir (Eğer elinde yoksa)
image_path = 'burger.jpg'
if not os.path.exists(image_path):
    print("Test resmi indiriliyor...")
    img_url = "https://storage.googleapis.com/mediapipe-tasks/image_classifier/burger.jpg" 
    urllib.request.urlretrieve(img_url, image_path)
    print("Test resmi indirildi!")

# 3. MediaPipe Image Classifier ayarları (Linkteki IMAGE modu)
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.ImageClassifierOptions(
    base_options=base_options,
    max_results=3) # En olası ilk 3 tahmini göstersin
classifier = vision.ImageClassifier.create_from_options(options)

# 4. Resmi MediaPipe'ın istediği formata dönüştürerek yükle
mp_image = mp.Image.create_from_file(image_path)

# 5. Sınıflandırmayı gerçekleştir
print("Sınıflandırma yapılıyor...\n")
classification_result = classifier.classify(mp_image)

# 6. Sonuçları terminale yazdır
print("--- SINIFLANDIRMA SONUÇLARI ---")
for category in classification_result.classifications[0].categories:
    print(f"Kategori: {category.category_name}, Skor: {category.score:.4f}")

# 7. Hoca için ekran kaydı (Deliverable) alabilmen adına resmi ekranda göster
img_cv2 = cv2.imread(image_path)
top_category = classification_result.classifications[0].categories[0]
text = f"{top_category.category_name} ({top_category.score:.2f})"

# En yüksek ihtimalli sonucu resmin sol üstüne yaz
cv2.putText(img_cv2, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
cv2.imshow("Task 1 - Static Image Classification", img_cv2)

print("\nÇıkmak için resim penceresi üzerindeyken klavyeden herhangi bir tuşa basın...")
cv2.waitKey(0)
cv2.destroyAllWindows()