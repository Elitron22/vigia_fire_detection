"""Construye un tar.gz mínimo para desplegar la aplicación en Raspberry Pi 5."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import tempfile


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "artifacts" / "14_final_model_freeze" / "final" / "weights" / "best.pt"
DEFAULT_OUTPUT = ROOT / "dist" / "tfm-fire-rpi5.tar.gz"
EXPECTED_SHA256 = "bfb54c4726519fb473bb7c4825577e51c05af2b0db5fd46bf1931a8a29285f67"


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
    if not model.is_file() or sha256_file(model) != EXPECTED_SHA256:
        raise RuntimeError("El bundle solo admite el checkpoint PyTorch final con su SHA-256 congelado.")
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="tfm-rpi5-bundle-") as temporary:
        bundle = Path(temporary) / "tfm-fire-rpi5"
        bundle.mkdir()
        shutil.copytree(ROOT / "fire_app", bundle / "fire_app", ignore=shutil.ignore_patterns("__pycache__"))
        (bundle / "tools").mkdir()
        for name in ("run_detection_app.py", "benchmark_rpi5.py"):
            shutil.copy2(ROOT / "tools" / name, bundle / "tools" / name)
        (bundle / "configs").mkdir()
        shutil.copy2(ROOT / "configs" / "app.rpi5.yaml", bundle / "configs" / "app.rpi5.yaml")
        shutil.copytree(ROOT / "deployment" / "rpi5", bundle / "deployment" / "rpi5", ignore=shutil.ignore_patterns("model"))
        (bundle / "model").mkdir()
        shutil.copy2(model, bundle / "model" / "yolo26s_768_final.pt")
        (bundle / "artifacts").mkdir()

        inventory = {
            path.relative_to(bundle).as_posix(): sha256_file(path)
            for path in sorted(bundle.rglob("*"))
            if path.is_file()
        }
        (bundle / "bundle_manifest.json").write_text(
            json.dumps({"schema_version": 1, "files": inventory}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        with tarfile.open(output, "w:gz") as archive:
            archive.add(bundle, arcname=bundle.name)
    print(output)


if __name__ == "__main__":
    main()
