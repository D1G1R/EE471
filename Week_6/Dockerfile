FROM python:3.11.5-slim-bullseye

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    ca-certificates \
    libasound2 \
    libssl1.1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Yorumu kaldır ve chmod ekle
RUN mkdir -p /app/static /app/uploads && chmod -R 777 /app/static /app/uploads

EXPOSE 8080

CMD ["python", "-m", "flask", "run", "--host=0.0.0.0", "--port=8080"]