"""Construye el paquete independiente de despliegue NCNN para Raspberry Pi 5."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import tempfile


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "deployment" / "rpi5" / "model" / "yolo26s_768_ncnn_model"
DEFAULT_OUTPUT = ROOT / "dist" / "tfm-fire-rpi5-ncnn.tar.gz"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = args.model.resolve()
    required = {
        "model.ncnn.param", "model.ncnn.bin", "metadata.yaml",
        "deployment_manifest.json", "deployment_calibration.json",
    }
    if not model.is_dir() or not required.issubset({path.name for path in model.iterdir()}):
        raise RuntimeError("El directorio NCNN no contiene todos los artefactos requeridos.")
    calibration = json.loads((model / "deployment_calibration.json").read_text(encoding="utf-8"))
    if calibration.get("status") != "passed" or calibration.get("source_split") != "val":
        raise RuntimeError("La calibración NCNN sobre validación no está aprobada.")
    if calibration.get("test_inference_executed") is not False:
        raise RuntimeError("No se construirá un bundle calibrado consultando test.")

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="tfm-rpi5-ncnn-bundle-") as temporary:
        bundle = Path(temporary) / "tfm-fire-rpi5-ncnn"
        bundle.mkdir()
        shutil.copytree(ROOT / "fire_app", bundle / "fire_app", ignore=shutil.ignore_patterns("__pycache__"))
        (bundle / "tools").mkdir()
        for name in ("run_detection_app.py", "benchmark_rpi5.py"):
            shutil.copy2(ROOT / "tools" / name, bundle / "tools" / name)
        (bundle / "configs").mkdir()
        shutil.copy2(ROOT / "configs" / "app.rpi5.ncnn.yaml", bundle / "configs" / "app.rpi5.ncnn.yaml")
        shutil.copytree(ROOT / "deployment" / "rpi5_ncnn", bundle / "deployment" / "rpi5_ncnn")
        (bundle / "model").mkdir()
        shutil.copytree(model, bundle / "model" / model.name, ignore=shutil.ignore_patterns("__pycache__"))
        (bundle / "artifacts").mkdir()

        inventory = {
            path.relative_to(bundle).as_posix(): sha256_file(path)
            for path in sorted(bundle.rglob("*")) if path.is_file()
        }
        manifest = {
            "schema_version": 1,
            "variant_id": "yolo26s_768_ncnn_rpi5",
            "role": "separate_edge_deployment_variant_not_final_test_model",
            "source_split_for_calibration": "val",
            "test_inference_executed": False,
            "files": inventory,
        }
        (bundle / "bundle_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        with tarfile.open(output, "w:gz") as archive:
            archive.add(bundle, arcname=bundle.name)
    print(output)


if __name__ == "__main__":
    main()
