"""Catalog and preview videos without running inference or using the GPU.

The script is intentionally independent from Ultralytics.  It probes every video
with OpenCV, computes a SHA-256 checksum, and creates contact sheets sampled at
fixed relative positions.  It is used both to curate the operational corpus and
to audit it later.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np


VIDEO_EXTENSIONS = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".ogv", ".webm"}


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_frame(capture: cv2.VideoCapture, frame_index: int) -> np.ndarray | None:
    capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_index))
    ok, frame = capture.read()
    return frame if ok else None


def fit_thumbnail(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    scale = min(width / frame.shape[1], height / frame.shape[0])
    resized = cv2.resize(
        frame,
        (max(1, round(frame.shape[1] * scale)), max(1, round(frame.shape[0] * scale))),
        interpolation=cv2.INTER_AREA,
    )
    x = (width - resized.shape[1]) // 2
    y = (height - resized.shape[0]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def probe(path: Path, sample_positions: list[float]) -> tuple[dict[str, object], list[np.ndarray]]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"No se pudo abrir el vídeo: {path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0.0
    frames: list[np.ndarray] = []
    for position in sample_positions:
        index = round(max(0.0, min(1.0, position)) * max(0, frame_count - 1))
        frame = read_frame(capture, index)
        if frame is not None:
            frames.append(frame)
    capture.release()

    return (
        {
            "filename": path.name,
            "relative_path": path.as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            "width": width,
            "height": height,
            "fps": round(fps, 6),
            "frame_count": frame_count,
            "duration_seconds": round(duration, 3),
        },
        frames,
    )


def label_thumbnail(frame: np.ndarray, label: str, position: float) -> np.ndarray:
    thumb = fit_thumbnail(frame, 256, 144)
    cv2.rectangle(thumb, (0, 0), (256, 24), (0, 0, 0), thickness=-1)
    cv2.putText(
        thumb,
        f"{label}  {position:.0%}",
        (6, 17),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return thumb


def write_contact_sheets(
    entries: list[tuple[Path, dict[str, object], list[np.ndarray]]],
    output_dir: Path,
    sample_positions: list[float],
    columns: int = 5,
) -> None:
    cells: list[np.ndarray] = []
    for path, _, frames in entries:
        for position, frame in zip(sample_positions, frames, strict=False):
            cells.append(label_thumbnail(frame, path.stem, position))

    per_sheet = columns * 8
    for sheet_index in range(math.ceil(len(cells) / per_sheet)):
        page = cells[sheet_index * per_sheet : (sheet_index + 1) * per_sheet]
        rows = math.ceil(len(page) / columns)
        blank = np.full((144, 256, 3), 245, dtype=np.uint8)
        page.extend([blank] * (rows * columns - len(page)))
        sheet = np.vstack(
            [np.hstack(page[row * columns : (row + 1) * columns]) for row in range(rows)]
        )
        target = output_dir / f"contact_sheet_{sheet_index + 1:02d}.jpg"
        if not cv2.imwrite(str(target), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88]):
            raise RuntimeError(f"No se pudo escribir {target}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=project_root() / "data" / "D-Fire" / "Videos",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root() / "artifacts" / "12_operational_video_corpus" / "source_audit",
    )
    parser.add_argument("--positions", default="0.1,0.5,0.9")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    positions = [float(value) for value in args.positions.split(",")]

    paths = sorted(
        (path for path in source_dir.rglob("*") if path.suffix.lower() in VIDEO_EXTENSIONS),
        key=lambda path: path.name.lower(),
    )
    if not paths:
        raise RuntimeError(f"No se encontraron vídeos en {source_dir}")

    entries: list[tuple[Path, dict[str, object], list[np.ndarray]]] = []
    for index, path in enumerate(paths, start=1):
        record, frames = probe(path, positions)
        record["relative_path"] = path.relative_to(project_root()).as_posix()
        entries.append((path, record, frames))
        print(f"[{index:03d}/{len(paths):03d}] {path.name}", flush=True)

    fieldnames = list(entries[0][1])
    with (output_dir / "source_video_catalog.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(record for _, record, _ in entries)
    (output_dir / "source_video_catalog.json").write_text(
        json.dumps([record for _, record, _ in entries], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_contact_sheets(entries, output_dir, positions)
    print(f"Catálogo: {output_dir / 'source_video_catalog.csv'}", flush=True)


if __name__ == "__main__":
    main()
