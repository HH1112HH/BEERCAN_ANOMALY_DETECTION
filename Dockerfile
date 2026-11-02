FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY backend_asgi.py camera_service.py classifier.py conveyor_controller.py detector.py main.py patch_core.py ./
COPY static ./static

EXPOSE 5004
CMD ["uvicorn", "backend_asgi:app", "--host", "0.0.0.0", "--port", "5004"]
