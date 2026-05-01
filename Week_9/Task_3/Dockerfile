# Temel Python imajını kullan
FROM python:3.9-slim

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Çalışma dizinini ayarla
WORKDIR /app

# Gereksinimleri kopyala ve kur
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Tüm kodları ve test resimlerini kopyala
COPY . .

# Konteyner çalıştığında argüman alarak bu scripti çalıştır
ENTRYPOINT ["python", "face_classifier.py"]