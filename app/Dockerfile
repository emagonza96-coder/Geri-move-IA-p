FROM python:3.12-slim

# Dependencias del sistema para OpenCV y MediaPipe
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libx11-6 \
    libxcb1 \
    libfontconfig1 \
    libice6 \
    v4l-utils \
    ffmpeg \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Instalar dependencias Python
COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Descargar modelo PoseLandmarker (full) durante el build para que ya esté cacheado
RUN mkdir -p models && \
    wget -q -O models/pose_landmarker_full.task \
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"

COPY app ./app

ENTRYPOINT ["python", "-m", "app.main"]
