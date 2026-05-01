import urllib.request
import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Model download
model_path = 'deeplabv3.tflite'
if not os.path.exists(model_path):
    print("Resim Segmentasyonu modeli indiriliyor...")
    url = "https://storage.googleapis.com/mediapipe-models/image_segmenter/deeplab_v3/float32/1/deeplab_v3.tflite"
    urllib.request.urlretrieve(url, model_path)
    print("Model indirildi!")

# 2. Test için örnek bir resim indir (Eğer elinde yoksa)
image_path = 'pose-3.jpg'  # Örnek resim adı
if not os.path.exists(image_path):
    print("Örnek test resmi indiriliyor...")
    img_url = "https://storage.googleapis.com/mediapipe-assets/cat.jpg" # Örnek kedi resmi
    try:
        urllib.request.urlretrieve(img_url, image_path)
        print("Test resmi indirildi!")
    except Exception as e:
        print(f"Resim indirilemedi ({e}), alternatif bir resimle değiştirebilirsiniz.")

# 3. MediaPipe Image Segmenter ayarları 
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.ImageSegmenterOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    output_category_mask=True)

# 4. Segmenter'i ayarlar ile oluştur
with vision.ImageSegmenter.create_from_options(options) as segmenter:

    # 5. Resmi MediaPipe formatına dönüştür
    if os.path.exists(image_path):
        mp_image = mp.Image.create_from_file(image_path)
        
        # 6. Segmentasyonu gerçekleştir
        print("Segmentasyon yapılıyor...\n")
        segmentation_result = segmenter.segment(mp_image)
        
        # Category maskesi (Her piksel hangi nesne sınıfına ait olduğunu tutar)
        category_mask = segmentation_result.category_mask
        mask_array = category_mask.numpy_view()
        
        # 7. Sonuçları görselleştir (OpenCV ile)
        img_cv2 = cv2.imread(image_path)
        
        # Dokümantasyondaki uyarının tam uygulaması: 
        # "kontrastı artırmak ve görselleştirmeyi iyileştirmek için piksel değerleri 10 ile çarpılır."
        visualized_mask = (mask_array * 10).astype(np.uint8)
        
        # Maskeyi 3 kanallı (BGR) formata çevir
        mask_bgr = cv2.cvtColor(visualized_mask, cv2.COLOR_GRAY2BGR)
        
        # Orijinal resim ile kategori maskesini yan yana (sitedeki gibi) göster
        output_image = np.hstack((img_cv2, mask_bgr))
        
        cv2.imshow("Orijinal Resim (Sol) ve Kategori Maskesi (Sag)", output_image)
        
        print("\nÇıkmak için resim penceresi üzerindeyken klavyeden herhangi bir tuşa basın...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    else:
        print(f"{image_path} bulunamadı. Lütfen geçerli bir resim ekleyiniz.")