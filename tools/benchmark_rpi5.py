"""Mide latencia extremo a extremo del modelo desplegado en Raspberry Pi 5."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import time

from ultralytics import YOLO, __version__ as ultralytics_version


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path(os.environ.get("TFM_APP_MODEL_PATH", "model/yolo26s_768_final.pt")))
    parser.add_argument("--images", type=Path, required=True, help="Imagen o directorio de imágenes de validación/manuales.")
    parser.add_argument("--imgsz", type=int, default=768)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--smoke-threshold", type=float, default=0.36)
    parser.add_argument("--fire-threshold", type=float, default=0.16)
    parser.add_argument("--output", type=Path, default=Path("artifacts/rpi5_benchmark.json"))
    return parser.parse_args()


def image_paths(value: Path) -> list[Path]:
    if value.is_file() and value.suffix.lower() in IMAGE_SUFFIXES:
        return [value]
    if value.is_dir():
        paths = sorted(path for path in value.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
        if paths:
            return paths
    raise FileNotFoundError(f"No se encontraron imágenes compatibles en {value}")


def temperature_celsius() -> float | None:
    path = Path("/sys/class/thermal/thermal_zone0/temp")
    return float(path.read_text().strip()) / 1000.0 if path.is_file() else None


def throttled_status() -> str | None:
    try:
        return subprocess.run(
            ["vcgencmd", "get_throttled"], capture_output=True, text=True, timeout=2, check=False
        ).stdout.strip() or None
    except (FileNotFoundError, subprocess.SubprocessError):
        return None


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return ordered[index]


def main() -> None:
    args = parse_args()
    model_path = args.model.expanduser().resolve()
    images = image_paths(args.images.expanduser().resolve())
    model = YOLO(str(model_path))
    thresholds = {"smoke": args.smoke_threshold, "fire": args.fire_threshold}
    if any(not 0.0 <= value <= 1.0 for value in thresholds.values()):
        raise ValueError("Los umbrales deben pertenecer al intervalo [0, 1].")
    confidence = min(thresholds.values())
    for index in range(max(0, args.warmup)):
        model.predict(str(images[index % len(images)]), imgsz=args.imgsz, conf=confidence, iou=0.70, device="cpu", quantize="fp32", rect=False, max_det=300, agnostic_nms=False, verbose=False)

    timings: list[float] = []
    raw_counts = {"smoke": 0, "fire": 0}
    started_temperature = temperature_celsius()
    for index in range(max(1, args.runs)):
        image = images[index % len(images)]
        started = time.perf_counter()
        result = model.predict(str(image), imgsz=args.imgsz, conf=confidence, iou=0.70, device="cpu", quantize="fp32", rect=False, max_det=300, agnostic_nms=False, verbose=False)[0]
        timings.append((time.perf_counter() - started) * 1000.0)
        if result.boxes is not None:
            for class_id, score in zip(result.boxes.cls.tolist(), result.boxes.conf.tolist()):
                name = str(result.names[int(class_id)])
                threshold = thresholds.get(name, confidence)
                if float(score) >= threshold:
                    raw_counts[name] = raw_counts.get(name, 0) + 1

    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "platform": {"machine": platform.machine(), "platform": platform.platform(), "python": platform.python_version()},
        "ultralytics_version": ultralytics_version,
        "model": str(model_path),
        "images": len(images),
        "warmup_runs": max(0, args.warmup),
        "measured_runs": max(1, args.runs),
        "imgsz": args.imgsz,
        "class_thresholds": thresholds,
        "latency_ms": {
            "mean": statistics.fmean(timings),
            "median": statistics.median(timings),
            "p95": percentile(timings, 0.95),
            "minimum": min(timings),
            "maximum": max(timings),
        },
        "throughput_fps_from_mean": 1000.0 / statistics.fmean(timings),
        "detections_after_final_thresholds": raw_counts,
        "temperature_celsius": {"before": started_temperature, "after": temperature_celsius()},
        "throttled": throttled_status(),
        "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
