"""Inferencia YOLO reutilizable por imagen, vídeo y cámara."""

from __future__ import annotations

from dataclasses import dataclass
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from ultralytics import YOLO

from .alerting import AlertEvent, TemporalAlertEngine


EXPECTED_CLASS_NAMES = {0: "smoke", 1: "fire"}


def validate_class_mapping(names: dict[object, object]) -> dict[int, str]:
    """Evita desplegar por error un modelo con clases distintas."""
    normalized = {int(key): str(value).lower() for key, value in names.items()}
    if normalized != EXPECTED_CLASS_NAMES:
        raise ValueError(
            f"Mapeo de clases incompatible: {normalized}; se esperaba {EXPECTED_CLASS_NAMES}."
        )
    return normalized


def operating_keep_indices(
    names: dict[object, object],
    class_ids: list[int] | np.ndarray,
    scores: list[float] | np.ndarray,
    class_confidence: dict[str, float],
) -> list[int]:
    """Conserva únicamente cajas que cumplen el umbral final de su clase."""
    normalized = {int(key): str(value).lower() for key, value in names.items()}
    return [
        index
        for index, (class_id, score) in enumerate(zip(class_ids, scores))
        if float(score) >= float(class_confidence.get(normalized[int(class_id)], 1.0))
    ]


@dataclass
class InferenceResult:
    annotated_bgr: np.ndarray
    detections: list[dict[str, object]]
    processing_ms: float


@dataclass
class VideoResult:
    frames_read: int
    frames_inferred: int
    fps: float
    duration_seconds: float
    events: list[AlertEvent]
    detections_total: int
    output_path: str
    media_type: str


class ModelManager:
    """Carga los pesos bajo demanda y serializa el acceso a cada modelo."""

    def __init__(self) -> None:
        self._models: dict[str, YOLO] = {}
        self._locks: dict[str, threading.Lock] = {}
        self._cache_lock = threading.Lock()

    def _model_and_lock(self, weights: Path) -> tuple[YOLO, threading.Lock]:
        key = str(Path(weights).resolve())
        with self._cache_lock:
            if key not in self._models:
                model = YOLO(key)
                validate_class_mapping(model.names)
                self._models[key] = model
                self._locks[key] = threading.Lock()
            return self._models[key], self._locks[key]

    def predict_frame(
        self,
        frame_bgr: np.ndarray,
        weights: Path,
        *,
        confidence: float,
        nms_iou: float,
        imgsz: int,
        device: str,
        class_confidence: dict[str, float] | None = None,
    ) -> InferenceResult:
        model, lock = self._model_and_lock(weights)
        started = time.perf_counter()
        with lock:
            prediction = model.predict(
                source=frame_bgr,
                conf=confidence,
                iou=nms_iou,
                imgsz=imgsz,
                device=device,
                # Reproduce el modo FP32 usado al fijar métricas y umbrales.
                # En YOLO26 este argumento también evita cambiar de rama de
                # inferencia respecto al pipeline de evaluación del TFM.
                quantize="fp32",
                rect=False,
                max_det=300,
                agnostic_nms=False,
                verbose=False,
            )[0]
            if prediction.boxes is not None and class_confidence:
                class_ids = prediction.boxes.cls.detach().cpu().numpy().astype(int)
                scores = prediction.boxes.conf.detach().cpu().numpy()
                keep = operating_keep_indices(prediction.names, class_ids, scores, class_confidence)
                prediction.boxes = prediction.boxes[keep]
            annotated = prediction.plot()
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        detections: list[dict[str, object]] = []
        if prediction.boxes is not None:
            boxes = prediction.boxes.xyxy.detach().cpu().numpy()
            confidences = prediction.boxes.conf.detach().cpu().numpy()
            classes = prediction.boxes.cls.detach().cpu().numpy().astype(int)
            for xyxy, score, class_id in zip(boxes, confidences, classes):
                detections.append(
                    {
                        "class_id": int(class_id),
                        "class_name": str(prediction.names[int(class_id)]),
                        "confidence": float(score),
                        "xyxy": [round(float(value), 2) for value in xyxy],
                    }
                )
        return InferenceResult(annotated_bgr=annotated, detections=detections, processing_ms=elapsed_ms)


def decode_image(content: bytes) -> np.ndarray:
    array = np.frombuffer(content, dtype=np.uint8)
    frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("El contenido no es una imagen válida o su formato no está soportado.")
    return frame


def encode_jpeg(frame_bgr: np.ndarray, quality: int = 88) -> bytes:
    ok, encoded = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("No se pudo codificar el fotograma anotado.")
    return encoded.tobytes()


def process_video(
    manager: ModelManager,
    input_path: Path,
    output_path: Path,
    weights: Path,
    *,
    confidence: float,
    nms_iou: float,
    imgsz: int,
    device: str,
    stride: int,
    alert_engine: TemporalAlertEngine,
    class_confidence: dict[str, float] | None = None,
    on_event: Callable[[AlertEvent, bytes], None] | None = None,
) -> VideoResult:
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise ValueError("No se pudo abrir el vídeo subido.")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        fps = 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        capture.release()
        raise ValueError("El vídeo no informa de unas dimensiones válidas.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    staging_path = output_path.parent / f"{output_path.stem}.staging.avi"
    writer = cv2.VideoWriter(str(staging_path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError("No se pudo crear el vídeo temporal MJPEG de salida.")

    frames_read = 0
    frames_inferred = 0
    detections_total = 0
    events: list[AlertEvent] = []
    last_annotated: np.ndarray | None = None
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            timestamp = frames_read / fps
            if frames_read % max(1, stride) == 0:
                result = manager.predict_frame(
                    frame,
                    weights,
                    confidence=confidence,
                    nms_iou=nms_iou,
                    imgsz=imgsz,
                    device=device,
                    class_confidence=class_confidence,
                )
                last_annotated = result.annotated_bgr
                frames_inferred += 1
                detections_total += len(result.detections)
                new_events = alert_engine.update(timestamp, result.detections)
                events.extend(new_events)
                if on_event:
                    snapshot = encode_jpeg(result.annotated_bgr)
                    for event in new_events:
                        on_event(event, snapshot)
            writer.write(last_annotated if last_annotated is not None else frame)
            frames_read += 1
    finally:
        capture.release()
        writer.release()

    if frames_read == 0:
        staging_path.unlink(missing_ok=True)
        raise ValueError("El vídeo no contiene fotogramas decodificables.")

    actual_output = output_path
    media_type = "video/mp4"
    ffmpeg = shutil.which("ffmpeg")
    transcoded = False
    if ffmpeg:
        completed = subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(staging_path),
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(output_path),
            ],
            capture_output=True,
            text=True,
            timeout=max(60.0, frames_read / fps * 4.0),
            check=False,
        )
        transcoded = completed.returncode == 0 and output_path.is_file() and output_path.stat().st_size > 0
    if transcoded:
        staging_path.unlink(missing_ok=True)
    else:
        output_path.unlink(missing_ok=True)
        actual_output = output_path.with_suffix(".avi")
        actual_output.unlink(missing_ok=True)
        staging_path.replace(actual_output)
        media_type = "video/x-msvideo"
    return VideoResult(
        frames_read=frames_read,
        frames_inferred=frames_inferred,
        fps=fps,
        duration_seconds=frames_read / fps,
        events=events,
        detections_total=detections_total,
        output_path=str(actual_output),
        media_type=media_type,
    )
