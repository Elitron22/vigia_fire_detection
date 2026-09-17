# syntax=docker/dockerfile:1.7
FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    YOLO_CONFIG_DIR=/tmp/Ultralytics \
    TFM_PROJECT_ROOT=/app \
    TFM_APP_DEVICE=cpu

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-app.txt /tmp/requirements-app.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install \
        --index-url https://download.pytorch.org/whl/cpu \
        torch==2.12.1+cpu torchvision==0.27.1+cpu \
    && python -m pip install --requirement /tmp/requirements-app.txt \
    && rm /tmp/requirements-app.txt

COPY fire_app/ fire_app/
COPY tools/run_detection_app.py tools/run_detection_app.py
COPY tests/ tests/
COPY configs/app.release.yaml configs/app.release.yaml
COPY artifacts/14_final_model_freeze/ artifacts/14_final_model_freeze/
COPY deployment/rpi5/model/yolo26s_640_ncnn_model/ deployment/rpi5/model/yolo26s_640_ncnn_model/

EXPOSE 8000
CMD ["python", "tools/run_detection_app.py", "--host", "0.0.0.0", "--port", "8000"]
