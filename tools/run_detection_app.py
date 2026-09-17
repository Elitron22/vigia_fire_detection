"""Inicia la aplicación local de detección."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import uvicorn


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fire_app.config import load_settings  # noqa: E402


def parse_args() -> argparse.Namespace:
    settings = load_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=settings.host)
    parser.add_argument("--port", type=int, default=settings.port)
    parser.add_argument("--reload", action="store_true", help="Recarga el servidor al editar código.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    uvicorn.run("fire_app.main:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
