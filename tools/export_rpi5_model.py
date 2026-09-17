"""Exporta el YOLO26s final congelado a NCNN para Raspberry Pi 5."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from ultralytics import YOLO, __version__ as ultralytics_version


ROOT = Path(__file__).resolve().parents[1]
FREEZE_ROOT = ROOT / "artifacts" / "14_final_model_freeze" / "final"
DEFAULT_OUTPUT = ROOT / "deployment" / "rpi5" / "model" / "yolo26s_768_ncnn_model"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--imgsz", type=int, default=None, help="Resolución cuadrada de exportación; por defecto usa la congelada.")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def ensure_export_requirements() -> None:
    """Instala solo los conversores; evita que PNNX reemplace PyTorch/CUDA."""
    missing: list[str] = []
    if importlib.util.find_spec("ncnn") is None:
        missing.append("ncnn==1.0.20260526")
    if importlib.util.find_spec("pnnx") is None:
        missing.append("pnnx==20260526")
    if missing:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-deps", *missing],
            check=True,
        )


def main() -> None:
    args = parse_args()
    ensure_export_requirements()
    freeze = json.loads((FREEZE_ROOT / "freeze_manifest.json").read_text(encoding="utf-8"))
    imgsz = int(args.imgsz or freeze["inference_imgsz"])
    if imgsz <= 0 or imgsz % 32:
        raise ValueError("imgsz debe ser positivo y múltiplo de 32.")
    weights = FREEZE_ROOT / freeze["frozen_weights_rel"]
    actual_hash = sha256_file(weights)
    if actual_hash != freeze["weights_sha256"]:
        raise RuntimeError("El checkpoint congelado no supera la verificación SHA-256.")
    output = args.output.resolve()
    if output.exists():
        if not args.force:
            raise FileExistsError(f"Ya existe {output}; usa --force para sustituirlo.")
        if output != DEFAULT_OUTPUT.resolve() and ROOT not in output.parents:
            raise ValueError("Por seguridad, --force solo puede borrar una salida dentro del proyecto.")
        shutil.rmtree(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="tfm-rpi5-export-") as temporary:
        temporary_root = Path(temporary)
        temporary_weights = temporary_root / f"yolo26s_{imgsz}.pt"
        shutil.copy2(weights, temporary_weights)
        exported = Path(
            YOLO(str(temporary_weights)).export(
                format="ncnn",
                imgsz=imgsz,
                batch=1,
                device="cpu",
            )
        )
        shutil.copytree(exported, output)

    files = {
        path.relative_to(output).as_posix(): sha256_file(path)
        for path in sorted(output.rglob("*"))
        if path.is_file()
    }
    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "format": "ncnn",
        "source_model": f"YOLO26s 768→{imgsz}",
        "source_experiment_id": freeze["experiment_id"],
        "source_weights_sha256": actual_hash,
        "imgsz": imgsz,
        "smoke_threshold": float(freeze["smoke_threshold"]),
        "fire_threshold": float(freeze["fire_threshold"]),
        "ultralytics_version": ultralytics_version,
        "files": files,
    }
    (output / "deployment_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(output)


if __name__ == "__main__":
    main()
