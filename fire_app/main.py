"""API FastAPI y flujo web para la demostración del detector."""

from __future__ import annotations

import base64
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import re
import threading
import time
import uuid

from fastapi import FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from .alerting import AlertEvent, EventStore, TemporalAlertEngine
from .config import ROOT, AppSettings, load_settings
from .inference import ModelManager, decode_image, encode_jpeg, process_video
from .model_registry import ModelRegistry, RegisteredModel
from .telegram import TelegramNotifier


STATIC_DIR = Path(__file__).resolve().parent / "static"


def _probability(value: float, name: str) -> float:
    if not 0.0 <= value <= 1.0:
        raise HTTPException(status_code=422, detail=f"{name} debe estar entre 0 y 1.")
    return float(value)


def _image_size(value: int) -> int:
    if value < 320 or value > 1280 or value % 32:
        raise HTTPException(status_code=422, detail="imgsz debe ser múltiplo de 32 entre 320 y 1280.")
    return value


def _safe_filename(name: str, fallback: str) -> str:
    basename = Path(name).name
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", basename).strip("._")
    return cleaned[:120] or fallback


def _run_id(prefix: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}_{prefix}_{uuid.uuid4().hex[:8]}"


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def create_app(settings: AppSettings | None = None) -> FastAPI:
    config = settings or load_settings()
    registry = ModelRegistry(ROOT)
    manager = ModelManager()
    notifier = TelegramNotifier(config.telegram)
    event_store = EventStore(config.event_log)
    config.output_directory.mkdir(parents=True, exist_ok=True)

    app = FastAPI(
        title="Detector de incendios forestales — TFM",
        version="0.1.0",
        description="Inferencia de humo y fuego en imágenes, vídeos y cámara.",
    )
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    def available_models() -> list[RegisteredModel]:
        models = registry.discover()
        configured = set(config.inference.selectable_models)
        if configured:
            models = [model for model in models if model.experiment_id in configured]
        return models

    def inference_imgsz(model: RegisteredModel) -> int:
        return config.inference.selectable_models.get(
            model.experiment_id,
            model.trained_imgsz or config.inference.imgsz,
        )

    def operating_thresholds(
        smoke_threshold: float | None,
        fire_threshold: float | None,
        model: RegisteredModel,
    ) -> dict[str, float]:
        defaults = config.alerts.model_minimum_confidence.get(
            model.experiment_id,
            config.alerts.minimum_confidence,
        )
        return {
            "smoke": _probability(
                defaults["smoke"] if smoke_threshold is None else smoke_threshold,
                "smoke_threshold",
            ),
            "fire": _probability(
                defaults["fire"] if fire_threshold is None else fire_threshold,
                "fire_threshold",
            ),
        }

    def resolve_model(experiment_id: str | None) -> RegisteredModel:
        selectable = set(config.inference.selectable_models) or {config.inference.default_experiment_id}
        if not config.inference.allow_runtime_overrides and experiment_id not in {None, *selectable}:
            raise HTTPException(status_code=409, detail="El despliegue tiene bloqueado el modelo final.")
        try:
            model = registry.resolve(experiment_id, config.inference.default_experiment_id)
            if config.inference.selectable_models and model.experiment_id not in selectable:
                raise HTTPException(status_code=409, detail="El modelo no forma parte del despliegue final.")
            return model
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    def runtime_parameters(
        confidence: float | None,
        nms_iou: float | None,
        imgsz: int | None,
        model: RegisteredModel,
    ) -> tuple[float, float, int]:
        expected = (
            config.inference.confidence,
            config.inference.nms_iou,
            inference_imgsz(model),
        )
        if not config.inference.allow_runtime_overrides:
            supplied = (confidence, nms_iou, imgsz)
            if any(value is not None and value != target for value, target in zip(supplied, expected)):
                raise HTTPException(
                    status_code=409,
                    detail="El despliegue final bloquea confianza, IoU y resolución.",
                )
            return expected
        return (
            _probability(config.inference.confidence if confidence is None else confidence, "confidence"),
            _probability(config.inference.nms_iou if nms_iou is None else nms_iou, "nms_iou"),
            _image_size(config.inference.imgsz if imgsz is None else imgsz),
        )

    def source_event_saver(max_notifications: int = 1):
        """Crea un guardado por fuente con un límite de avisos a Telegram.

        Los eventos se conservan todos en el registro, pero una imagen, un vídeo
        o una sesión de cámara solo puede originar un mensaje. El cierre y los
        posibles re-disparos siguen siendo auditables sin duplicar avisos.
        """
        sent_notifications = 0
        lock = threading.Lock()

        def save(event: AlertEvent, jpeg_bytes: bytes) -> None:
            nonlocal sent_notifications
            notification: dict[str, object] | None = None
            if event.event_type == "triggered":
                with lock:
                    should_notify = sent_notifications < max_notifications
                    if should_notify:
                        sent_notifications += 1
                if should_notify:
                    notification = notifier.send_alert(event, jpeg_bytes)
                else:
                    notification = {
                        "status": "suppressed",
                        "message": "Aviso duplicado suprimido para esta fuente.",
                    }
            event_store.append(event, notification)

        return save

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/health")
    async def health() -> dict[str, object]:
        models = available_models()
        selected = registry.resolve(None, config.inference.default_experiment_id) if models else None
        return {
            "status": "ok",
            "models_available": len(models),
            "device": config.inference.device,
            "architecture": platform.machine(),
            "selected_model": selected.public_dict(selected=True) if selected else None,
            "telegram": notifier.public_status(),
        }

    @app.get("/api/models")
    async def models() -> dict[str, object]:
        available = available_models()
        selected_id = config.inference.default_experiment_id
        if available and all(model.experiment_id != selected_id for model in available):
            selected_id = available[0].experiment_id
        return {
            "default_experiment_id": selected_id,
            "models": [
                {
                    **model.public_dict(selected=model.experiment_id == selected_id),
                    "inference_imgsz": inference_imgsz(model),
                    "minimum_confidence": config.alerts.model_minimum_confidence.get(
                        model.experiment_id,
                        config.alerts.minimum_confidence,
                    ),
                }
                for model in available
            ],
        }

    @app.get("/api/config")
    async def public_config() -> dict[str, object]:
        return {
            "inference": asdict(config.inference),
            "alerts": asdict(config.alerts),
            "telegram": notifier.public_status(),
            "max_upload_mib": round(config.max_upload_bytes / (1024 * 1024)),
        }

    @app.get("/api/events")
    async def events(limit: int = Query(50, ge=1, le=500)) -> dict[str, object]:
        return {"events": event_store.recent(limit)}

    @app.post("/api/telegram/test")
    async def telegram_test() -> dict[str, object]:
        return await run_in_threadpool(notifier.send_test)

    @app.post("/api/analyze/image")
    async def analyze_image(
        request: Request,
        experiment_id: str | None = None,
        confidence: float | None = Query(None),
        nms_iou: float | None = Query(None),
        imgsz: int | None = Query(None),
        smoke_threshold: float | None = Query(None),
        fire_threshold: float | None = Query(None),
    ) -> dict[str, object]:
        content = await request.body()
        if not content:
            raise HTTPException(status_code=400, detail="No se recibió ninguna imagen.")
        if len(content) > config.max_upload_bytes:
            raise HTTPException(status_code=413, detail="La imagen supera el tamaño máximo permitido.")
        model = resolve_model(experiment_id)
        confidence, nms_iou, imgsz = runtime_parameters(confidence, nms_iou, imgsz, model)
        thresholds = operating_thresholds(smoke_threshold, fire_threshold, model)
        confidence = min(confidence, *thresholds.values())
        try:
            frame = decode_image(content)
            result = await run_in_threadpool(
                manager.predict_frame,
                frame,
                model.weights,
                confidence=confidence,
                nms_iou=nms_iou,
                imgsz=imgsz,
                device=config.inference.device,
                class_confidence=thresholds,
            )
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        run_id = _run_id("image")
        output_dir = config.output_directory / "images" / run_id
        annotated_jpeg = encode_jpeg(result.annotated_bgr)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "annotated.jpg").write_bytes(annotated_jpeg)
        metadata = {
            "schema_version": 1,
            "run_id": run_id,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source_filename": _safe_filename(request.headers.get("x-filename", "imagen"), "imagen"),
            "experiment_id": model.experiment_id,
            "parameters": {
                "confidence": confidence,
                "nms_iou": nms_iou,
                "imgsz": imgsz,
                "device": config.inference.device,
                "minimum_confidence": thresholds,
            },
            "processing_ms": result.processing_ms,
            "detections": result.detections,
        }
        alert_events: list[AlertEvent] = []
        if config.alerts.enabled and result.detections:
            candidates = [
                detection
                for detection in result.detections
                if str(detection.get("class_name", "")) in thresholds
                and float(detection.get("confidence", 0.0))
                >= thresholds[str(detection.get("class_name", ""))]
            ]
            if candidates:
                # Se prioriza fuego frente a humo y, dentro de cada clase, la
                # detección de mayor confianza.
                strongest = max(
                    candidates,
                    key=lambda item: (
                        str(item.get("class_name", "")) == "fire",
                        float(item.get("confidence", 0.0)),
                    ),
                )
                event = AlertEvent.create(
                    "triggered",
                    str(strongest["class_name"]),
                    0.0,
                    float(strongest["confidence"]),
                    str(metadata["source_filename"]),
                    "active",
                )
                await run_in_threadpool(source_event_saver(), event, annotated_jpeg)
                alert_events.append(event)
        metadata["events"] = [event.to_dict() for event in alert_events]
        _write_json(output_dir / "metadata.json", metadata)
        return {
            **metadata,
            "annotated_image": "data:image/jpeg;base64," + base64.b64encode(annotated_jpeg).decode("ascii"),
        }

    @app.post("/api/analyze/video")
    async def analyze_video(
        request: Request,
        experiment_id: str | None = None,
        confidence: float | None = Query(None),
        nms_iou: float | None = Query(None),
        imgsz: int | None = Query(None),
        smoke_threshold: float | None = Query(None),
        fire_threshold: float | None = Query(None),
        hold_seconds: float = Query(config.alerts.hold_seconds, ge=0.0, le=300.0),
    ) -> FileResponse:
        content = await request.body()
        if not content:
            raise HTTPException(status_code=400, detail="No se recibió ningún vídeo.")
        if len(content) > config.max_upload_bytes:
            raise HTTPException(status_code=413, detail="El vídeo supera el tamaño máximo permitido.")
        model = resolve_model(experiment_id)
        confidence, nms_iou, imgsz = runtime_parameters(confidence, nms_iou, imgsz, model)
        thresholds = operating_thresholds(smoke_threshold, fire_threshold, model)
        confidence = min(confidence, *thresholds.values())
        run_id = _run_id("video")
        output_dir = config.output_directory / "videos" / run_id
        output_dir.mkdir(parents=True, exist_ok=True)
        source_name = _safe_filename(request.headers.get("x-filename", "entrada.mp4"), "entrada.mp4")
        suffix = Path(source_name).suffix.lower() or ".mp4"
        if suffix not in {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}:
            suffix = ".video"
        input_path = output_dir / f"input{suffix}"
        output_path = output_dir / "annotated.mp4"
        input_path.write_bytes(content)
        alert_settings = replace(config.alerts, hold_seconds=hold_seconds, minimum_confidence=thresholds)
        engine = TemporalAlertEngine(alert_settings, source_id=source_name)
        save_source_event = source_event_saver()
        try:
            summary = await run_in_threadpool(
                process_video,
                manager,
                input_path,
                output_path,
                model.weights,
                confidence=confidence,
                nms_iou=nms_iou,
                imgsz=imgsz,
                device=config.inference.device,
                stride=config.inference.video_stride,
                alert_engine=engine,
                class_confidence=thresholds,
                on_event=save_source_event,
            )
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        metadata = {
            "schema_version": 1,
            "run_id": run_id,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source_filename": source_name,
            "experiment_id": model.experiment_id,
            "parameters": {
                "confidence": confidence,
                "nms_iou": nms_iou,
                "imgsz": imgsz,
                "device": config.inference.device,
                "hold_seconds": hold_seconds,
                "minimum_confidence": thresholds,
            },
            "summary": {**asdict(summary), "events": [event.to_dict() for event in summary.events]},
        }
        _write_json(output_dir / "metadata.json", metadata)
        final_video = Path(summary.output_path)
        return FileResponse(
            final_video,
            media_type=summary.media_type,
            filename=f"{Path(source_name).stem}_detectado{final_video.suffix}",
            headers={
                "X-TFM-Run-Id": run_id,
                "X-TFM-Event-Count": str(len(summary.events)),
                "X-TFM-Detection-Summary": json.dumps(
                    summary.detections_by_class,
                    ensure_ascii=True,
                    separators=(",", ":"),
                ),
                "X-TFM-Processing-Ms-Per-Frame": f"{summary.processing_ms_per_frame:.3f}",
            },
        )

    @app.websocket("/api/live")
    async def live(websocket: WebSocket) -> None:
        await websocket.accept()
        try:
            params = websocket.query_params
            model = resolve_model(params.get("experiment_id"))
            confidence, nms_iou, imgsz = runtime_parameters(
                float(params["confidence"]) if "confidence" in params else None,
                float(params["nms_iou"]) if "nms_iou" in params else None,
                int(params["imgsz"]) if "imgsz" in params else None,
                model,
            )
            thresholds = operating_thresholds(
                float(params["smoke_threshold"]) if "smoke_threshold" in params else None,
                float(params["fire_threshold"]) if "fire_threshold" in params else None,
                model,
            )
            confidence = min(confidence, *thresholds.values())
            hold_seconds = float(params.get("hold_seconds", config.alerts.hold_seconds))
            if not 0 <= hold_seconds <= 300:
                raise ValueError("hold_seconds debe estar entre 0 y 300.")
            source_id = f"camara_{uuid.uuid4().hex[:8]}"
            engine = TemporalAlertEngine(
                replace(config.alerts, hold_seconds=hold_seconds, minimum_confidence=thresholds),
                source_id,
            )
            save_source_event = source_event_saver()
            started = time.monotonic()
            await websocket.send_json({"type": "ready", "source_id": source_id, "experiment_id": model.experiment_id})
            while True:
                content = await websocket.receive_bytes()
                if len(content) > 12 * 1024 * 1024:
                    await websocket.send_json({"type": "error", "message": "El fotograma supera 12 MiB."})
                    continue
                frame = decode_image(content)
                result = await run_in_threadpool(
                    manager.predict_frame,
                    frame,
                    model.weights,
                    confidence=confidence,
                    nms_iou=nms_iou,
                    imgsz=imgsz,
                    device=config.inference.device,
                    class_confidence=thresholds,
                )
                timestamp = time.monotonic() - started
                new_events = engine.update(timestamp, result.detections)
                annotated_jpeg = encode_jpeg(result.annotated_bgr, quality=82)
                for event in new_events:
                    await run_in_threadpool(save_source_event, event, annotated_jpeg)
                await websocket.send_json(
                    {
                        "type": "prediction",
                        "timestamp_seconds": timestamp,
                        "processing_ms": result.processing_ms,
                        "detections": result.detections,
                        "alert_state": engine.snapshot(),
                        "events": [event.to_dict() for event in new_events],
                        "annotated_image": "data:image/jpeg;base64," + base64.b64encode(annotated_jpeg).decode("ascii"),
                    }
                )
        except WebSocketDisconnect:
            return
        except (ValueError, KeyError, FileNotFoundError, RuntimeError) as exc:
            await websocket.send_json({"type": "error", "message": str(exc)})
            await websocket.close(code=1008)

    return app


app = create_app()
