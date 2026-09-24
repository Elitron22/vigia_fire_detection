"""Benchmark temporal reproducible para los candidatos D-Fire.

La inferencia se ejecuta una sola vez por modelo a confianza mínima y se guarda
en una caché inmutable ligada a pesos, manifiesto, resolución y código. Los
umbrales y reglas temporales se evalúan después offline. El script no consulta
el conjunto de test de imágenes.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import random
import shutil
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tfm_pipeline as pipeline


DEFAULT_CONFIG = ROOT / "configs" / "operational_video_benchmark.yaml"
CLASS_NAMES = {0: "smoke", 1: "fire"}
FRAME_COLUMNS = [
    "model_key", "video_id", "sample_index", "source_frame_index",
    "timestamp_s", "source_timestamp_s", "max_smoke_confidence",
    "max_fire_confidence", "smoke_detections", "fire_detections",
    "preprocess_ms", "inference_ms", "postprocess_ms", "predict_wall_ms",
]
BOX_COLUMNS = [
    "model_key", "video_id", "sample_index", "timestamp_s", "class_id",
    "class_name", "confidence", "x1", "y1", "x2", "y2",
]
INFERENCE_CACHE_SCHEMA = 1


def fingerprint(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def project_relative(path: Path) -> str:
    return Path(path).resolve().relative_to(ROOT.resolve()).as_posix()


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("Solo se admite schema_version=1.")
    if not config.get("test_locked"):
        raise ValueError("El benchmark operativo debe mantener test_locked=true.")
    if float(config["sample_fps"]) <= 0:
        raise ValueError("sample_fps debe ser positivo.")
    if not 0 < float(config["prediction_confidence"]) < 1:
        raise ValueError("prediction_confidence no válido.")
    if len(config.get("models", {})) != 2:
        raise ValueError("El protocolo requiere exactamente dos modelos.")
    for model_key, model in config["models"].items():
        if not model.get("profiles"):
            raise ValueError(f"{model_key}: faltan perfiles.")
        for profile_key, profile in model["profiles"].items():
            for field in ("smoke_threshold", "fire_threshold"):
                value = float(profile[field])
                if not config["prediction_confidence"] <= value <= 1:
                    raise ValueError(f"{model_key}/{profile_key}: {field} no válido.")
    for rule_key, rule in config.get("temporal_rules", {}).items():
        window = int(rule["window_frames"])
        hits = int(rule["minimum_hits"])
        clear = int(rule["clear_after_misses"])
        if not (1 <= hits <= window and clear >= 1):
            raise ValueError(f"Regla temporal no válida: {rule_key}.")
    return config


def load_corpus(config: dict[str, Any]) -> tuple[Path, list[dict[str, Any]]]:
    manifest_path = ROOT / config["corpus_manifest"]
    records = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not records or len({row["id"] for row in records}) != len(records):
        raise ValueError("El manifiesto de vídeo está vacío o contiene ID duplicados.")
    allowed_labels = {"positive", "negative"}
    for row in records:
        if row["event_label"] not in allowed_labels:
            raise ValueError(f"Etiqueta no válida en {row['id']}.")
        path = ROOT / row["relative_path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        if "test" in Path(row["relative_path"]).parts:
            raise ValueError("El benchmark no admite rutas del conjunto de test.")
    return manifest_path, records


def resolve_experiment(model: dict[str, Any]) -> tuple[Path, dict[str, Any], Path]:
    experiment_root = ROOT / "artifacts" / "experiments" / model["experiment_id"]
    descriptor_path = experiment_root / "experiment.json"
    descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
    if descriptor.get("status") != "complete":
        raise ValueError(f"Experimento incompleto: {model['experiment_id']}.")
    weights = ROOT / descriptor["best_model_rel"]
    if not weights.is_file():
        raise FileNotFoundError(weights)
    return experiment_root, descriptor, weights


def cache_signature(
    config: dict[str, Any], model_key: str, model: dict[str, Any], weights: Path,
    manifest_path: Path, ultralytics_version: str,
) -> tuple[str, dict[str, Any]]:
    signature = {
        "schema_version": 1,
        "model_key": model_key,
        "experiment_id": model["experiment_id"],
        "weights_sha256": pipeline.sha256_file(weights),
        "manifest_sha256": pipeline.sha256_file(manifest_path),
        "sample_fps": float(config["sample_fps"]),
        "imgsz": int(model["imgsz"]),
        "prediction_confidence": float(config["prediction_confidence"]),
        "nms_iou": float(config["nms_iou"]),
        "inference_cache_schema": INFERENCE_CACHE_SCHEMA,
        "ultralytics_version": ultralytics_version,
    }
    return fingerprint(signature)[:16], signature


def _atomic_csv_gzip(table: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    table.to_csv(temporary, index=False, compression="gzip")
    temporary.replace(path)


def _first_frame(records: list[dict[str, Any]]):
    import cv2

    capture = cv2.VideoCapture(str(ROOT / records[0]["relative_path"]))
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"No se pudo leer {records[0]['id']} para el calentamiento.")
    return frame


def infer_model(
    config: dict[str, Any], model_key: str, model_config: dict[str, Any],
    records: list[dict[str, Any]], manifest_path: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    import cv2
    import torch
    import ultralytics
    from ultralytics import YOLO

    experiment_root, descriptor, weights = resolve_experiment(model_config)
    cache_id, signature = cache_signature(
        config, model_key, model_config, weights, manifest_path, ultralytics.__version__
    )
    cache_dir = experiment_root / "evaluation" / "operational_video_cache" / cache_id
    frame_path = cache_dir / "frame_scores.csv.gz"
    box_path = cache_dir / "box_predictions.csv.gz"
    metadata_path = cache_dir / "cache_metadata.json"

    if metadata_path.is_file() and frame_path.is_file() and box_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("status") == "complete" and metadata.get("signature") == signature:
            frame_scores = pd.read_csv(frame_path)
            expected_ids = {row["id"] for row in records}
            if set(frame_scores.video_id) == expected_ids and len(frame_scores) > 0:
                print(f"[cache] {model_key}: {len(frame_scores)} fotogramas", flush=True)
                return frame_scores, metadata

    cache_dir.mkdir(parents=True, exist_ok=True)
    random.seed(int(config["seed"]))
    np.random.seed(int(config["seed"]))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(config["seed"]))

    detector = YOLO(str(weights))
    pipeline.validate_class_mapping(detector.names)
    device = config["device"] if torch.cuda.is_available() else "cpu"
    warmup = _first_frame(records)
    for _ in range(int(config["warmup_iterations"])):
        detector.predict(
            warmup, imgsz=int(model_config["imgsz"]), conf=float(config["prediction_confidence"]),
            iou=float(config["nms_iou"]), device=device, verbose=False,
        )

    frame_rows: list[dict[str, Any]] = []
    box_rows: list[dict[str, Any]] = []
    video_runs: list[dict[str, Any]] = []
    sample_interval = 1.0 / float(config["sample_fps"])

    for video_number, record in enumerate(records, start=1):
        path = ROOT / record["relative_path"]
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise RuntimeError(f"No se pudo abrir {path}.")
        native_fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(native_fps) or native_fps <= 0:
            native_fps = float(record["fps"])
        if native_fps + 1e-9 < float(config["sample_fps"]):
            raise ValueError(f"{record['id']}: FPS nativos inferiores al muestreo solicitado.")

        next_sample_s = 0.0
        source_frame_index = 0
        sample_index = 0
        decoded_frames = 0
        started = time.perf_counter()
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            decoded_frames += 1
            source_timestamp_s = source_frame_index / native_fps
            tolerance = 0.5 / native_fps
            if source_timestamp_s + tolerance >= next_sample_s:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                prediction_started = time.perf_counter()
                result = detector.predict(
                    frame, imgsz=int(model_config["imgsz"]),
                    conf=float(config["prediction_confidence"]), iou=float(config["nms_iou"]),
                    device=device, verbose=False,
                )[0]
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                predict_wall_ms = (time.perf_counter() - prediction_started) * 1000

                confidences = {0: 0.0, 1: 0.0}
                counts = {0: 0, 1: 0}
                if result.boxes is not None and len(result.boxes):
                    xyxy = result.boxes.xyxy.detach().cpu().numpy()
                    confs = result.boxes.conf.detach().cpu().numpy()
                    classes = result.boxes.cls.detach().cpu().numpy().astype(int)
                    for coordinates, confidence, class_id in zip(xyxy, confs, classes):
                        if class_id not in CLASS_NAMES:
                            raise ValueError(f"Clase inesperada {class_id} en {model_key}.")
                        confidence = float(confidence)
                        confidences[class_id] = max(confidences[class_id], confidence)
                        counts[class_id] += 1
                        box_rows.append({
                            "model_key": model_key, "video_id": record["id"],
                            "sample_index": sample_index, "timestamp_s": next_sample_s,
                            "class_id": class_id, "class_name": CLASS_NAMES[class_id],
                            "confidence": confidence,
                            **dict(zip(("x1", "y1", "x2", "y2"), map(float, coordinates))),
                        })
                speed = result.speed or {}
                frame_rows.append({
                    "model_key": model_key, "video_id": record["id"],
                    "sample_index": sample_index, "source_frame_index": source_frame_index,
                    "timestamp_s": next_sample_s, "source_timestamp_s": source_timestamp_s,
                    "max_smoke_confidence": confidences[0],
                    "max_fire_confidence": confidences[1],
                    "smoke_detections": counts[0], "fire_detections": counts[1],
                    "preprocess_ms": float(speed.get("preprocess", np.nan)),
                    "inference_ms": float(speed.get("inference", np.nan)),
                    "postprocess_ms": float(speed.get("postprocess", np.nan)),
                    "predict_wall_ms": predict_wall_ms,
                })
                sample_index += 1
                next_sample_s += sample_interval
            source_frame_index += 1
        capture.release()
        elapsed = time.perf_counter() - started
        expected_samples = max(1, int(math.ceil(float(record["duration_seconds"]) * config["sample_fps"])))
        # Algunos contenedores WebM/Ogg declaran una duracion ligeramente mayor
        # que la secuencia que OpenCV consigue decodificar (VFR, audio o cola del
        # contenedor). A 5 fps, dos segundos siguen detectando truncamientos reales
        # sin rechazar esas diferencias de metadatos.
        sample_tolerance = max(2, int(math.ceil(float(config["sample_fps"]) * 2.0)))
        if abs(sample_index - expected_samples) > sample_tolerance:
            raise AssertionError(
                f"{record['id']}: {sample_index} muestras; se esperaban aproximadamente "
                f"{expected_samples} (tolerancia {sample_tolerance})."
            )
        video_runs.append({
            "video_id": record["id"], "native_fps": native_fps,
            "decoded_frames": decoded_frames, "sampled_frames": sample_index,
            "wall_seconds": elapsed,
            "processing_fps": sample_index / elapsed if elapsed else np.nan,
        })
        print(
            f"[{video_number}/{len(records)}] {model_key} · {record['id']}: "
            f"{sample_index} muestras, {sample_index / elapsed:.1f} fps de proceso",
            flush=True,
        )

    frames = pd.DataFrame(frame_rows, columns=FRAME_COLUMNS)
    boxes = pd.DataFrame(box_rows, columns=BOX_COLUMNS)
    _atomic_csv_gzip(frames, frame_path)
    _atomic_csv_gzip(boxes, box_path)
    metadata = {
        "schema_version": 1, "status": "complete",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "signature": signature, "cache_id": cache_id,
        "experiment_id": descriptor["experiment_id"],
        "frame_scores_rel": project_relative(frame_path),
        "box_predictions_rel": project_relative(box_path),
        "sampled_frames": len(frames), "detections": len(boxes),
        "videos": video_runs,
    }
    pipeline.write_json_atomic(metadata_path, metadata)
    return frames, metadata


def get_frame_scores(
    config: dict[str, Any], model_key: str, model_config: dict[str, Any],
    records: list[dict[str, Any]], manifest_path: Path, offline: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not offline:
        return infer_model(config, model_key, model_config, records, manifest_path)

    try:
        import ultralytics
    except ImportError as error:
        raise RuntimeError("El modo offline requiere conocer la versión de Ultralytics de la caché.") from error
    experiment_root, _, weights = resolve_experiment(model_config)
    cache_id, signature = cache_signature(
        config, model_key, model_config, weights, manifest_path, ultralytics.__version__
    )
    cache_dir = experiment_root / "evaluation" / "operational_video_cache" / cache_id
    metadata_path = cache_dir / "cache_metadata.json"
    frame_path = cache_dir / "frame_scores.csv.gz"
    if not metadata_path.is_file() or not frame_path.is_file():
        raise FileNotFoundError(f"No existe caché válida para {model_key}: {cache_dir}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("status") != "complete" or metadata.get("signature") != signature:
        raise ValueError(f"Caché incompatible para {model_key}.")
    frames = pd.read_csv(frame_path)
    print(f"[offline] {model_key}: {len(frames)} fotogramas", flush=True)
    return frames, metadata


def temporal_states(
    raw_hits: Iterable[bool], window_frames: int, minimum_hits: int,
    clear_after_misses: int,
) -> np.ndarray:
    """Aplica activación M-de-N y cierre tras K ausencias consecutivas."""
    history: deque[bool] = deque(maxlen=int(window_frames))
    active = False
    consecutive_misses = 0
    states: list[bool] = []
    for raw_hit in map(bool, raw_hits):
        history.append(raw_hit)
        if active:
            if raw_hit:
                consecutive_misses = 0
            else:
                consecutive_misses += 1
                if consecutive_misses >= int(clear_after_misses):
                    active = False
                    consecutive_misses = 0
        elif len(history) >= int(minimum_hits) and sum(history) >= int(minimum_hits):
            active = True
            consecutive_misses = 0
        states.append(active)
    return np.asarray(states, dtype=bool)


def states_to_episodes(states: np.ndarray, timestamps: np.ndarray, interval: float) -> list[dict[str, float]]:
    episodes: list[dict[str, float]] = []
    start: float | None = None
    for state, timestamp in zip(states, timestamps):
        if state and start is None:
            start = float(timestamp)
        elif not state and start is not None:
            episodes.append({"start_s": start, "end_s": float(timestamp), "duration_s": float(timestamp) - start})
            start = None
    if start is not None:
        end = float(timestamps[-1] + interval)
        episodes.append({"start_s": start, "end_s": end, "duration_s": end - start})
    return episodes


def evaluate_video_configuration(
    frames: pd.DataFrame, record: dict[str, Any], model_key: str,
    profile_key: str, profile: dict[str, Any], rule_key: str,
    rule: dict[str, Any], sample_fps: float,
) -> dict[str, Any]:
    table = frames[frames.video_id == record["id"]].sort_values("sample_index")
    if table.empty:
        raise AssertionError(f"Faltan fotogramas de {record['id']} para {model_key}.")
    smoke_raw = table.max_smoke_confidence.to_numpy() >= float(profile["smoke_threshold"])
    fire_raw = table.max_fire_confidence.to_numpy() >= float(profile["fire_threshold"])
    temporal_args = {
        "window_frames": int(rule["window_frames"]),
        "minimum_hits": int(rule["minimum_hits"]),
        "clear_after_misses": int(rule["clear_after_misses"]),
    }
    smoke_active = temporal_states(smoke_raw, **temporal_args)
    fire_active = temporal_states(fire_raw, **temporal_args)
    any_active = smoke_active | fire_active
    timestamps = table.timestamp_s.to_numpy(dtype=float)
    interval = 1.0 / float(sample_fps)
    episodes = states_to_episodes(any_active, timestamps, interval)

    event_label = record["event_label"]
    start = record.get("expected_alert_start_s")
    end = record.get("expected_alert_end_s")
    event_detected = np.nan
    time_to_detection_s = np.nan
    event_alert_coverage = np.nan
    expected_smoke_detected = np.nan
    expected_fire_detected = np.nan
    pre_event_alarm = False
    if event_label == "positive":
        start = float(start)
        end = float(record["duration_seconds"] if end is None else end)
        event_mask = (timestamps >= start - 1e-9) & (timestamps <= end + 1e-9)
        event_detected = bool(any_active[event_mask].any())
        if event_detected:
            first = float(timestamps[event_mask][np.flatnonzero(any_active[event_mask])[0]])
            time_to_detection_s = max(0.0, first - start)
        event_alert_coverage = float(any_active[event_mask].mean()) if event_mask.any() else np.nan
        expected = set(record.get("expected_classes", []))
        expected_smoke_detected = bool(smoke_active[event_mask].any()) if "smoke" in expected else np.nan
        expected_fire_detected = bool(fire_active[event_mask].any()) if "fire" in expected else np.nan
        pre_event_alarm = bool(any_active[timestamps < start - 1e-9].any())

    active_seconds = min(float(record["duration_seconds"]), float(any_active.sum()) * interval)
    false_episodes = len(episodes) if event_label == "negative" else 0
    false_active_seconds = active_seconds if event_label == "negative" else 0.0
    return {
        "configuration_id": f"{model_key}|{profile_key}|{rule_key}",
        "model_key": model_key, "profile_key": profile_key, "temporal_rule": rule_key,
        "video_id": record["id"], "evaluation_role": record["evaluation_role"],
        "event_label": event_label, "scenario_tags": ";".join(record["scenario_tags"]),
        "duration_seconds": float(record["duration_seconds"]),
        "sampled_frames": len(table),
        "smoke_threshold": float(profile["smoke_threshold"]),
        "fire_threshold": float(profile["fire_threshold"]),
        "window_frames": temporal_args["window_frames"],
        "minimum_hits": temporal_args["minimum_hits"],
        "clear_after_misses": temporal_args["clear_after_misses"],
        "raw_smoke_hit_frames": int(smoke_raw.sum()),
        "raw_fire_hit_frames": int(fire_raw.sum()),
        "smoke_alarm_frames": int(smoke_active.sum()),
        "fire_alarm_frames": int(fire_active.sum()),
        "any_alarm_frames": int(any_active.sum()),
        "alarm_episodes": len(episodes), "active_seconds": active_seconds,
        "event_detected": event_detected, "time_to_detection_s": time_to_detection_s,
        "event_alert_coverage": event_alert_coverage,
        "expected_smoke_detected": expected_smoke_detected,
        "expected_fire_detected": expected_fire_detected,
        "pre_event_alarm": pre_event_alarm,
        "false_alarm_episodes": false_episodes,
        "false_alarm_active_seconds": false_active_seconds,
        "has_false_alarm": bool(false_episodes > 0),
        "episode_intervals_json": json.dumps(episodes, ensure_ascii=False),
    }


def wilson_interval(successes: int, total: int, z: float = 1.959964) -> tuple[float, float]:
    if total <= 0:
        return np.nan, np.nan
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def bootstrap_interval(
    table: pd.DataFrame, statistic, iterations: int, rng: np.random.Generator,
) -> tuple[float, float]:
    if table.empty:
        return np.nan, np.nan
    values = []
    for _ in range(int(iterations)):
        sample = table.iloc[rng.integers(0, len(table), size=len(table))]
        value = float(statistic(sample))
        if math.isfinite(value):
            values.append(value)
    if not values:
        return np.nan, np.nan
    return tuple(map(float, np.percentile(values, [2.5, 97.5])))


def summarize_configurations(
    per_video: pd.DataFrame, config: dict[str, Any]
) -> pd.DataFrame:
    rows = []
    roles = ["all", *config["reporting_roles"]]
    rng = np.random.default_rng(int(config["seed"]))
    for configuration_id, configuration in per_video.groupby("configuration_id", sort=False):
        for role in roles:
            table = configuration if role == "all" else configuration[configuration.evaluation_role == role]
            if table.empty:
                continue
            positive = table[table.event_label == "positive"]
            negative = table[table.event_label == "negative"]
            detected = int((positive.event_detected == True).sum())  # noqa: E712
            recall = detected / len(positive) if len(positive) else np.nan
            recall_low, recall_high = wilson_interval(detected, len(positive))
            detected_ttd = positive.loc[positive.event_detected == True, "time_to_detection_s"].dropna()  # noqa: E712
            negative_hours = float(negative.duration_seconds.sum()) / 3600
            false_episodes = int(negative.false_alarm_episodes.sum())
            false_rate = false_episodes / negative_hours if negative_hours else np.nan
            false_low, false_high = bootstrap_interval(
                negative,
                lambda sample: sample.false_alarm_episodes.sum() / (sample.duration_seconds.sum() / 3600),
                int(config["bootstrap_iterations"]), rng,
            )
            ttd_low, ttd_high = bootstrap_interval(
                positive[positive.event_detected == True],  # noqa: E712
                lambda sample: sample.time_to_detection_s.median(),
                int(config["bootstrap_iterations"]), rng,
            )
            first = table.iloc[0]
            rows.append({
                "configuration_id": configuration_id, "model_key": first.model_key,
                "profile_key": first.profile_key, "temporal_rule": first.temporal_rule,
                "evaluation_role": role, "videos": len(table),
                "positive_videos": len(positive), "detected_events": detected,
                "event_recall": recall, "event_recall_ci95_low": recall_low,
                "event_recall_ci95_high": recall_high,
                "median_time_to_detection_s": float(detected_ttd.median()) if len(detected_ttd) else np.nan,
                "p90_time_to_detection_s": float(detected_ttd.quantile(.90)) if len(detected_ttd) else np.nan,
                "median_ttd_ci95_low": ttd_low, "median_ttd_ci95_high": ttd_high,
                "mean_event_alert_coverage": float(positive.event_alert_coverage.mean()) if len(positive) else np.nan,
                "negative_videos": len(negative), "negative_video_hours": negative_hours,
                "false_alarm_episodes": false_episodes,
                "false_alarms_per_hour": false_rate,
                "false_alarms_per_hour_ci95_low": false_low,
                "false_alarms_per_hour_ci95_high": false_high,
                "negative_videos_with_alarm": int(negative.has_false_alarm.sum()),
                "negative_video_alarm_rate": float(negative.has_false_alarm.mean()) if len(negative) else np.nan,
                "false_alarm_seconds_per_hour": (
                    float(negative.false_alarm_active_seconds.sum()) / negative_hours if negative_hours else np.nan
                ),
                "smoke_threshold": first.smoke_threshold, "fire_threshold": first.fire_threshold,
                "window_frames": first.window_frames, "minimum_hits": first.minimum_hits,
                "clear_after_misses": first.clear_after_misses,
            })
    return pd.DataFrame(rows)


def timing_summary(frame_scores: pd.DataFrame, cache_metadata: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for model_key, table in frame_scores.groupby("model_key", sort=False):
        metadata = cache_metadata[model_key]
        video_runs = metadata["videos"]
        total_wall = sum(float(row["wall_seconds"]) for row in video_runs)
        rows.append({
            "model_key": model_key, "sampled_frames": len(table),
            "preprocess_ms_median": float(table.preprocess_ms.median()),
            "inference_ms_median": float(table.inference_ms.median()),
            "inference_ms_p95": float(table.inference_ms.quantile(.95)),
            "postprocess_ms_median": float(table.postprocess_ms.median()),
            "predict_wall_ms_median": float(table.predict_wall_ms.median()),
            "predict_wall_ms_p95": float(table.predict_wall_ms.quantile(.95)),
            "end_to_end_processing_fps": len(table) / total_wall if total_wall else np.nan,
        })
    return pd.DataFrame(rows)


def pareto_selection(summary: pd.DataFrame) -> pd.DataFrame:
    table = summary[summary.evaluation_role == "all"].copy()
    table["pareto_efficient"] = False
    eligible = table.dropna(subset=["event_recall", "false_alarms_per_hour", "median_time_to_detection_s"])
    for index, row in eligible.iterrows():
        dominated = (
            (eligible.event_recall >= row.event_recall)
            & (eligible.false_alarms_per_hour <= row.false_alarms_per_hour)
            & (eligible.median_time_to_detection_s <= row.median_time_to_detection_s)
            & (
                (eligible.event_recall > row.event_recall)
                | (eligible.false_alarms_per_hour < row.false_alarms_per_hour)
                | (eligible.median_time_to_detection_s < row.median_time_to_detection_s)
            )
        ).any()
        table.loc[index, "pareto_efficient"] = not dominated
    table = table.sort_values(
        ["event_recall", "false_alarms_per_hour", "median_time_to_detection_s", "configuration_id"],
        ascending=[False, True, True, True], kind="stable",
    ).reset_index(drop=True)
    table.insert(0, "pilot_rank", np.arange(1, len(table) + 1))
    return table


def build_figures(
    selection: pd.DataFrame, timing: pd.DataFrame, config: dict[str, Any], output: Path,
) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    colors = {key: value["color"] for key, value in config["models"].items()}
    markers = {"instant": "o", "two_of_three": "s", "three_of_five": "^"}
    paths: list[Path] = []

    from matplotlib.lines import Line2D

    short_models = {"yolo26s_768": "YOLO26s 768→768", "yolov8s_768_eval640": "YOLOv8s 768→640"}
    short_rules = {"instant": "1/1", "two_of_three": "2/3", "three_of_five": "3/5"}

    fig, ax = plt.subplots(figsize=(10, 6.2))
    for (model_key, rule_key), table in selection.groupby(["model_key", "temporal_rule"], sort=False):
        ax.scatter(
            table.false_alarms_per_hour, table.median_time_to_detection_s,
            s=85, color=colors[model_key], marker=markers.get(rule_key, "o"), alpha=.82,
        )
    for row in selection[selection.pareto_efficient].itertuples():
        ax.annotate(
            f"{row.profile_key}\n{row.temporal_rule}",
            (row.false_alarms_per_hour, row.median_time_to_detection_s),
            xytext=(5, 6), textcoords="offset points", fontsize=8,
        )
    ax.set(
        title="Compromiso temporal del benchmark piloto",
        xlabel="Episodios de falsa alarma por hora negativa",
        ylabel="Mediana de tiempo hasta detección (s)",
    )
    ax.grid(color="#E3E6E8", linewidth=.8)
    model_handles = [
        Line2D([0], [0], marker="o", linestyle="none", markersize=8, color=colors[key], label=short_models[key])
        for key in config["models"]
    ]
    rule_handles = [
        Line2D([0], [0], marker=markers[key], linestyle="none", markersize=8,
               markerfacecolor="#666666", markeredgecolor="#666666", label=short_rules[key])
        for key in config["temporal_rules"]
    ]
    first_legend = ax.legend(handles=model_handles, frameon=False, fontsize=9, loc="upper right")
    ax.add_artist(first_legend)
    ax.legend(handles=rule_handles, title="Persistencia", frameon=False, fontsize=9, loc="center right")
    fig.tight_layout()
    path = figures / "01_false_alarms_vs_detection_time.png"
    fig.savefig(path, dpi=180); plt.close(fig); paths.append(path)

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9), constrained_layout=True)
    metrics = [
        ("false_alarms_per_hour", "Episodios falsos por hora", ".0f"),
        ("false_alarm_seconds_per_hour", "Segundos en falsa alarma por hora", ".0f"),
    ]
    for column, model_key in enumerate(config["models"]):
        model_table = selection[selection.model_key == model_key]
        profile_keys = list(config["models"][model_key]["profiles"])
        rule_keys = list(config["temporal_rules"])
        for row_index, (metric, metric_label, number_format) in enumerate(metrics):
            matrix = (
                model_table.pivot(index="profile_key", columns="temporal_rule", values=metric)
                .reindex(index=profile_keys, columns=rule_keys)
            )
            ax = axes[row_index, column]
            image = ax.imshow(matrix.values, cmap="YlOrBr", aspect="auto")
            for y in range(len(profile_keys)):
                for x in range(len(rule_keys)):
                    value = matrix.iloc[y, x]
                    ax.text(x, y, format(value, number_format), ha="center", va="center", color="#202124", fontsize=10)
            ax.set_xticks(range(len(rule_keys)), [short_rules[key] for key in rule_keys])
            ax.set_yticks(range(len(profile_keys)), [config["models"][model_key]["profiles"][key]["label"] for key in profile_keys])
            ax.set_xlabel("Regla temporal M/N")
            if column == 0:
                ax.set_ylabel("Perfil de confianza")
            ax.set_title(f"{short_models[model_key]} · {metric_label}", fontsize=11)
            fig.colorbar(image, ax=ax, fraction=.046, pad=.04)
    fig.suptitle("Comportamiento en los vídeos negativos del piloto", fontsize=15)
    path = figures / "02_configuration_comparison.png"
    fig.savefig(path, dpi=180); plt.close(fig); paths.append(path)

    fig, ax = plt.subplots(figsize=(8, 5))
    positions = np.arange(len(timing))
    width = .34
    ax.bar(positions - width / 2, timing.inference_ms_median, width, label="Mediana", color="#6F8FAF")
    ax.bar(positions + width / 2, timing.inference_ms_p95, width, label="P95", color="#D7A34A")
    ax.set_xticks(positions, [short_models[key] for key in timing.model_key])
    ax.set(title="Latencia de inferencia por fotograma", ylabel="Milisegundos")
    ax.grid(axis="y", color="#E3E6E8", linewidth=.8)
    ax.legend(frameon=False)
    fig.tight_layout()
    path = figures / "03_inference_latency.png"
    fig.savefig(path, dpi=180); plt.close(fig); paths.append(path)
    return paths


def markdown_table(table: pd.DataFrame, columns: list[str]) -> str:
    labels = {
        "pilot_rank": "Rango piloto", "model_key": "Modelo", "profile_key": "Perfil",
        "temporal_rule": "Regla", "event_recall": "Recall eventos",
        "median_time_to_detection_s": "TTD mediano", "false_alarms_per_hour": "FA/h",
        "negative_videos_with_alarm": "Vídeos neg. con alarma", "pareto_efficient": "Pareto",
    }
    header = "| " + " | ".join(labels.get(column, column) for column in columns) + " |"
    separator = "|" + "|".join("---" for _ in columns) + "|"
    rows = []
    for row in table[columns].itertuples(index=False, name=None):
        formatted = []
        for column, value in zip(columns, row):
            if pd.isna(value):
                formatted.append("—")
            elif column == "event_recall":
                formatted.append(f"{float(value):.1%}")
            elif column in {"median_time_to_detection_s", "false_alarms_per_hour"}:
                formatted.append(f"{float(value):.2f}")
            elif column == "pareto_efficient":
                formatted.append("sí" if bool(value) else "no")
            else:
                formatted.append(str(value))
        rows.append("| " + " | ".join(formatted) + " |")
    return "\n".join([header, separator, *rows])


def plain_markdown_table(table: pd.DataFrame) -> str:
    columns = list(table.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "|" + "|".join("---" for _ in columns) + "|"
    rows = []
    for values in table.itertuples(index=False, name=None):
        formatted = []
        for value in values:
            if pd.isna(value):
                formatted.append("—")
            elif isinstance(value, (float, np.floating)):
                formatted.append(f"{float(value):.3f}")
            else:
                formatted.append(str(value))
        rows.append("| " + " | ".join(formatted) + " |")
    return "\n".join([header, separator, *rows])


def write_summary(
    output: Path, selection: pd.DataFrame, records: list[dict[str, Any]],
    timing: pd.DataFrame, config: dict[str, Any], final_allowed: bool,
) -> Path:
    positives = sum(row["event_label"] == "positive" for row in records)
    negatives = sum(row["event_label"] == "negative" for row in records)
    independent_negatives = sum(
        row["event_label"] == "negative" and row["evaluation_role"] == "independent_operational"
        for row in records
    )
    negative_minutes = sum(
        float(row["duration_seconds"]) for row in records if row["event_label"] == "negative"
    ) / 60
    independent_negative_minutes = sum(
        float(row["duration_seconds"])
        for row in records
        if row["event_label"] == "negative"
        and row["evaluation_role"] == "independent_operational"
    ) / 60
    top = selection.head(10)
    columns = [
        "pilot_rank", "model_key", "profile_key", "temporal_rule", "event_recall",
        "median_time_to_detection_s", "false_alarms_per_hour",
        "negative_videos_with_alarm", "pareto_efficient",
    ]
    if final_allowed:
        status_text = "El corpus cumple los mínimos declarados para una selección operativa final."
    elif independent_negatives == 0:
        status_text = (
            "El resultado es **piloto**: no se declara un ganador final porque faltan "
            "negativos independientes y duración negativa suficiente."
        )
    else:
        status_text = (
            "El resultado es **piloto**: ya contiene negativos independientes, pero no "
            "alcanza la duración negativa mínima declarada."
        )
    lines = f"""# Benchmark operativo de vídeo D-Fire

## tl;dr

Se evaluaron **{len(selection)} configuraciones** sobre {len(records)} vídeos
({positives} positivos y {negatives} negativos), muestreados a
{float(config['sample_fps']):.1f} FPS. {status_text}

El conjunto de test de imágenes no se cargó. Las cifras de falsas alarmas por
hora combinan negativos externos con negativos diagnósticos de D-Fire y, por la
duración todavía limitada, no demuestran todavía generalización externa.

## Contexto y método

- Dos checkpoints congelados; no se reentrenó ningún modelo.
- Tres perfiles de confianza por modelo y tres reglas temporales.
- La persistencia se aplica por clase; la alerta global es humo O fuego.
- Una falsa alarma es un episodio temporal completo, no un fotograma.
- Intervalos del recall: Wilson 95 %. Intervalos de TTD y FA/h: bootstrap por vídeo.
- Los FPS de proceso incluyen la decodificación secuencial del vídeo; la latencia
  del modelo se informa aparte.

## Cobertura del corpus

- Duración negativa: **{negative_minutes:.2f} min**.
- Vídeos negativos independientes: **{independent_negatives}**.
- Duración negativa independiente: **{independent_negative_minutes:.2f} min**.
- Mínimo declarado para la evaluación final: **{float(config['minimum_final_negative_minutes']):.0f} min negativos**.

## Resultados piloto

{markdown_table(top, columns)}

La ordenación anterior prioriza recall de eventos, después menos falsas alarmas
por hora y finalmente menor tiempo hasta detección. Es una ayuda para revisar el
piloto, no una selección final.

## Rendimiento

{plain_markdown_table(timing)}

## Siguiente paso

Ampliar y congelar un holdout con negativos externos, revisar manualmente el
instante de inicio de cada evento y repetir el mismo pipeline sin cambiar pesos,
perfiles ni reglas. Solo entonces se puede escoger el sistema operativo y acudir
una única vez al test.

## Artefactos

- `per_video_metrics.csv`: métricas y episodios de cada vídeo/configuración.
- `benchmark_summary.csv`: agregados por rol con intervalos de incertidumbre.
- `pilot_selection.csv`: vista comparativa y frontera de Pareto.
- `timing_summary.csv`: latencia y throughput.
- `figures/`: comparaciones visuales.
- Las predicciones por fotograma y caja permanecen en cachés ligadas a cada experimento.
"""
    path = output / "RESUMEN_BENCHMARK_VIDEO.md"
    path.write_text(lines, encoding="utf-8")
    return path


def run(config_path: Path = DEFAULT_CONFIG, *, offline: bool = False) -> Path:
    config = load_config(config_path)
    manifest_path, records = load_corpus(config)
    run_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + fingerprint(config)[:8]
    output_parent = ROOT / config["output_parent"]
    output = output_parent / run_id
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(config_path, output / "config.yaml")
    run_summary: dict[str, Any] = {
        "schema_version": 1, "status": "running", "run_id": run_id,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "benchmark_id": config["benchmark_id"], "test_inference_executed": False,
        "corpus_manifest_rel": project_relative(manifest_path),
        "corpus_manifest_sha256": pipeline.sha256_file(manifest_path),
    }
    pipeline.write_json_atomic(output / "run_summary.json", run_summary)
    try:
        frame_tables = []
        cache_metadata: dict[str, dict[str, Any]] = {}
        for model_key, model_config in config["models"].items():
            frames, metadata = get_frame_scores(
                config, model_key, model_config, records, manifest_path, offline
            )
            frame_tables.append(frames)
            cache_metadata[model_key] = metadata
        frame_scores = pd.concat(frame_tables, ignore_index=True)

        rows = []
        for model_key, model_config in config["models"].items():
            model_frames = frame_scores[frame_scores.model_key == model_key]
            for profile_key, profile in model_config["profiles"].items():
                for rule_key, rule in config["temporal_rules"].items():
                    for record in records:
                        rows.append(evaluate_video_configuration(
                            model_frames, record, model_key, profile_key, profile,
                            rule_key, rule, float(config["sample_fps"]),
                        ))
        per_video = pd.DataFrame(rows)
        summary = summarize_configurations(per_video, config)
        timing = timing_summary(frame_scores, cache_metadata)
        selection = pareto_selection(summary)

        negative_minutes = sum(
            float(row["duration_seconds"]) for row in records if row["event_label"] == "negative"
        ) / 60
        independent_negatives = sum(
            row["event_label"] == "negative" and row["evaluation_role"] == "independent_operational"
            for row in records
        )
        final_allowed = (
            negative_minutes >= float(config["minimum_final_negative_minutes"])
            and (independent_negatives > 0 or not config["require_independent_negatives_for_final"])
        )

        per_video.to_csv(output / "per_video_metrics.csv", index=False)
        summary.to_csv(output / "benchmark_summary.csv", index=False)
        selection.to_csv(output / "pilot_selection.csv", index=False)
        timing.to_csv(output / "timing_summary.csv", index=False)
        pipeline.write_json_atomic(output / "cache_index.json", cache_metadata)
        figures = build_figures(selection, timing, config, output)
        summary_path = write_summary(output, selection, records, timing, config, final_allowed)

        code_dir = output / "code"
        code_dir.mkdir()
        for source in (
            Path(__file__), ROOT / "tools" / "verify_operational_video_benchmark.py",
            ROOT / "tools" / "build_operational_video_benchmark_notebook.py",
        ):
            if source.exists():
                shutil.copy2(source, code_dir / source.name)

        expected_configurations = sum(len(model["profiles"]) for model in config["models"].values()) * len(config["temporal_rules"])
        if len(selection) != expected_configurations:
            raise AssertionError(f"Configuraciones incompletas: {len(selection)}/{expected_configurations}.")
        if len(per_video) != expected_configurations * len(records):
            raise AssertionError("Cobertura por vídeo incompleta.")

        run_summary.update({
            "status": "complete", "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "offline": offline, "models": list(config["models"]),
            "configurations": expected_configurations, "videos": len(records),
            "positive_videos": sum(row["event_label"] == "positive" for row in records),
            "negative_videos": sum(row["event_label"] == "negative" for row in records),
            "negative_minutes": negative_minutes,
            "independent_negative_videos": independent_negatives,
            "final_selection_allowed": final_allowed,
            "summary_rel": project_relative(summary_path),
            "cache_ids": {key: value["cache_id"] for key, value in cache_metadata.items()},
            "output_hashes": {
                project_relative(path): pipeline.sha256_file(path)
                for path in output.rglob("*")
                if path.is_file() and path.name != "run_summary.json"
            },
            "figures": [project_relative(path) for path in figures],
        })
        pipeline.write_json_atomic(output / "run_summary.json", run_summary)
        pipeline.write_json_atomic(output_parent / "latest.json", {
            "run_id": run_id, "run_rel": project_relative(output),
            "summary_rel": project_relative(summary_path),
            "run_summary_sha256": pipeline.sha256_file(output / "run_summary.json"),
        })
        print(f"Benchmark completo: {output}", flush=True)
        print(selection[[
            "pilot_rank", "model_key", "profile_key", "temporal_rule", "event_recall",
            "median_time_to_detection_s", "false_alarms_per_hour", "pareto_efficient",
        ]].head(10).to_string(index=False), flush=True)
        return output
    except BaseException as error:
        run_summary.update(
            status="incomplete", completed_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
            error=f"{type(error).__name__}: {error}",
        )
        pipeline.write_json_atomic(output / "run_summary.json", run_summary)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--offline", action="store_true", help="Exige cachés; no carga la GPU.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args.config, offline=args.offline)


if __name__ == "__main__":
    main()
