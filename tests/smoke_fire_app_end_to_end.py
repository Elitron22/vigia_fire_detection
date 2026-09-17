"""Prueba funcional en CPU de imagen, vídeo y WebSocket.

No forma parte de la suite unitaria porque carga pesos reales. Se ejecuta de
forma explícita cuando se quiere verificar la aplicación completa.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile

import cv2
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fire_app.main import app  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image",
        type=Path,
        default=ROOT / "data" / "D-Fire" / "train" / "images" / "AoF00001.jpg",
    )
    return parser.parse_args()


def make_video(image_path: Path, output_path: Path) -> None:
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise FileNotFoundError(f"No se pudo abrir {image_path}")
    height, width = frame.shape[:2]
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), 4.0, (width, height))
    if not writer.isOpened():
        raise RuntimeError("No se pudo crear el vídeo temporal.")
    for _ in range(8):
        writer.write(frame)
    writer.release()


def main() -> None:
    image_path = parse_args().image.resolve()
    image_bytes = image_path.read_bytes()
    query = "imgsz=320&confidence=0.05&nms_iou=0.70&hold_seconds=0.5"
    report: dict[str, object] = {}
    with TestClient(app) as client:
        image_response = client.post(
            "/api/analyze/image?imgsz=320&confidence=0.05&nms_iou=0.70",
            content=image_bytes,
            headers={"X-Filename": image_path.name},
        )
        image_response.raise_for_status()
        image_result = image_response.json()
        report["image"] = {
            "status": image_response.status_code,
            "detections": len(image_result["detections"]),
            "run_id": image_result["run_id"],
        }

        with tempfile.TemporaryDirectory() as temporary:
            video_path = Path(temporary) / "smoke.mp4"
            make_video(image_path, video_path)
            video_response = client.post(
                f"/api/analyze/video?{query}",
                content=video_path.read_bytes(),
                headers={"X-Filename": video_path.name},
            )
            video_response.raise_for_status()
            report["video"] = {
                "status": video_response.status_code,
                "output_bytes": len(video_response.content),
                "events": int(video_response.headers["X-TFM-Event-Count"]),
                "run_id": video_response.headers["X-TFM-Run-Id"],
            }

        camera_query = query.replace("hold_seconds=0.5", "hold_seconds=0")
        with client.websocket_connect(f"/api/live?{camera_query}") as websocket:
            ready = websocket.receive_json()
            websocket.send_bytes(image_bytes)
            prediction = websocket.receive_json()
            if ready["type"] != "ready" or prediction["type"] != "prediction":
                raise AssertionError("El WebSocket no completó el protocolo esperado.")
            report["camera"] = {
                "status": prediction["type"],
                "detections": len(prediction["detections"]),
                "events": len(prediction["events"]),
            }
            if not prediction["events"]:
                raise AssertionError("La cámara no activó la alerta temporal de persistencia cero.")

    print(json.dumps({"status": "passed", **report}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
