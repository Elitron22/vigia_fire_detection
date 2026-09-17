"""Comprobacion rapida y determinista del entorno del TFM."""

from __future__ import annotations

import argparse
import json
import platform
import sys

import cv2
import matplotlib
import numpy as np
import pandas as pd
import torch
import ultralytics
from ultralytics import YOLO


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--build",
        action="store_true",
        help="Solo comprueba imports; durante el build no hay una GPU disponible.",
    )
    args = parser.parse_args()

    report = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": __import__("torchvision").__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "ultralytics": ultralytics.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "opencv": cv2.__version__,
        "matplotlib": matplotlib.__version__,
        "cuda_available": torch.cuda.is_available(),
    }

    # Construye la arquitectura sin descargar pesos; valida la integracion
    # Ultralytics/PyTorch tanto durante el build como en tiempo de ejecucion.
    model = YOLO("yolov8s.yaml")

    if not args.build and torch.cuda.is_available():
        device = torch.device("cuda:0")
        model.model.to(device).eval()
        sample = torch.zeros((1, 3, 640, 640), device=device)
        with torch.inference_mode():
            model.model(sample)
        torch.cuda.synchronize()
        properties = torch.cuda.get_device_properties(0)
        report.update(
            {
                "gpu": torch.cuda.get_device_name(0),
                "compute_capability": list(torch.cuda.get_device_capability(0)),
                "gpu_memory_gib": round(properties.total_memory / 1024**3, 2),
                "yolov8s_cuda_forward": "ok",
            }
        )
    elif not args.build:
        report["warning"] = (
            "CUDA no esta disponible. La auditoria funciona, pero el entrenamiento "
            "del baseline debe realizarse con GPU o habilitarse conscientemente en CPU."
        )

    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
