"""Configuración validada para la aplicación de detección."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class InferenceSettings:
    default_experiment_id: str
    selectable_models: dict[str, int]
    device: str
    imgsz: int
    confidence: float
    nms_iou: float
    video_stride: int
    allow_runtime_overrides: bool


@dataclass(frozen=True)
class AlertSettings:
    enabled: bool
    hold_seconds: float
    clear_seconds: float
    cooldown_seconds: float
    maximum_gap_seconds: float
    minimum_confidence: dict[str, float]
    model_minimum_confidence: dict[str, dict[str, float]] = field(default_factory=dict)


@dataclass(frozen=True)
class TelegramSettings:
    mode: str
    timeout_seconds: float


@dataclass(frozen=True)
class AppSettings:
    host: str
    port: int
    max_upload_bytes: int
    inference: InferenceSettings
    alerts: AlertSettings
    telegram: TelegramSettings
    output_directory: Path
    event_log: Path


def _bounded_probability(value: Any, name: str) -> float:
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} debe estar entre 0 y 1.")
    return number


def load_settings(path: Path | None = None) -> AppSettings:
    """Carga la configuración YAML y aplica anulaciones seguras por entorno."""
    source = Path(path or os.environ.get("TFM_APP_CONFIG", ROOT / "configs" / "app.yaml"))
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    if raw.get("schema_version") != 1:
        raise ValueError("La configuración de la app requiere schema_version=1.")

    server = raw["server"]
    inference = raw["inference"]
    alerts = raw["alerts"]
    telegram = raw["telegram"]
    storage = raw["storage"]
    device = os.environ.get("TFM_APP_DEVICE", str(inference["device"]))
    telegram_mode = os.environ.get("TFM_TELEGRAM_MODE", str(telegram["mode"]))
    if telegram_mode not in {"disabled", "dry_run", "live"}:
        raise ValueError("telegram.mode debe ser disabled, dry_run o live.")

    confidence = _bounded_probability(inference["confidence"], "inference.confidence")
    nms_iou = _bounded_probability(inference["nms_iou"], "inference.nms_iou")
    selectable_models = {
        str(experiment_id): int(imgsz)
        for experiment_id, imgsz in inference.get("selectable_models", {}).items()
    }
    if any(imgsz < 320 or imgsz > 1280 or imgsz % 32 for imgsz in selectable_models.values()):
        raise ValueError("Las resoluciones de inference.selectable_models deben ser múltiplos de 32 entre 320 y 1280.")
    if selectable_models and str(inference["default_experiment_id"]) not in selectable_models:
        raise ValueError("inference.default_experiment_id debe estar incluido en inference.selectable_models.")
    class_confidence = {
        str(name): _bounded_probability(value, f"alerts.minimum_confidence.{name}")
        for name, value in alerts["minimum_confidence"].items()
    }
    if set(class_confidence) != {"smoke", "fire"}:
        raise ValueError("Se requieren umbrales de alerta para smoke y fire.")
    model_class_confidence: dict[str, dict[str, float]] = {}
    for experiment_id, values in alerts.get("model_minimum_confidence", {}).items():
        profile = {
            str(name): _bounded_probability(value, f"alerts.model_minimum_confidence.{experiment_id}.{name}")
            for name, value in values.items()
        }
        if set(profile) != {"smoke", "fire"}:
            raise ValueError(f"El perfil {experiment_id} requiere umbrales para smoke y fire.")
        model_class_confidence[str(experiment_id)] = profile

    output_directory = ROOT / storage["output_directory"]
    event_log = ROOT / storage["event_log"]
    return AppSettings(
        host=str(server["host"]),
        port=int(server["port"]),
        max_upload_bytes=int(server["max_upload_mib"]) * 1024 * 1024,
        inference=InferenceSettings(
            default_experiment_id=str(inference["default_experiment_id"]),
            selectable_models=selectable_models,
            device=device,
            imgsz=int(inference["imgsz"]),
            confidence=confidence,
            nms_iou=nms_iou,
            video_stride=max(1, int(inference["video_stride"])),
            allow_runtime_overrides=bool(inference.get("allow_runtime_overrides", True)),
        ),
        alerts=AlertSettings(
            enabled=bool(alerts["enabled"]),
            hold_seconds=max(0.0, float(alerts["hold_seconds"])),
            clear_seconds=max(0.0, float(alerts["clear_seconds"])),
            cooldown_seconds=max(0.0, float(alerts["cooldown_seconds"])),
            maximum_gap_seconds=max(0.0, float(alerts["maximum_gap_seconds"])),
            minimum_confidence=class_confidence,
            model_minimum_confidence=model_class_confidence,
        ),
        telegram=TelegramSettings(
            mode=telegram_mode,
            timeout_seconds=max(0.1, float(telegram["timeout_seconds"])),
        ),
        output_directory=output_directory,
        event_log=event_log,
    )
