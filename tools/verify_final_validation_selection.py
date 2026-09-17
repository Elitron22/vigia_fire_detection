"""Verifica la selección final de modelos limitada a validación."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tfm_pipeline as pipeline


def main() -> None:
    config = yaml.safe_load((ROOT / "configs" / "final_validation_selection.yaml").read_text(encoding="utf-8"))
    parent = ROOT / config["output_parent"]
    pointer = pipeline.read_json(parent / "latest.json")
    output = ROOT / pointer["run_rel"]
    summary = pipeline.read_json(output / "run_summary.json")
    if summary.get("status") != "complete" or summary.get("split") != "val":
        raise AssertionError("Ejecución final incompleta o split incorrecto")
    if summary.get("test_inference_executed") is not False or any(summary.get("source_test_flags", [True])):
        raise AssertionError("La selección ha consultado test o una fuente no bloqueada")
    if pipeline.sha256_file(output / "RESUMEN_SELECCION_VALIDACION.md") != pointer["summary_sha256"]:
        raise AssertionError("El resumen no coincide con latest.json")
    for relative, expected in summary["output_hashes"].items():
        if pipeline.sha256_file(output / relative) != expected:
            raise AssertionError(f"Hash incorrecto: {relative}")

    operating = pd.read_csv(output / "final_operating_points.csv")
    classes = pd.read_csv(output / "class_metrics.csv")
    sizes = pd.read_csv(output / "size_metrics.csv")
    standard = pd.read_csv(output / "standard_metrics.csv")
    errors = pd.read_csv(output / "error_comparison.csv")
    grid = pd.read_csv(output / "class_threshold_grid_01pct.csv")
    budgets = pd.read_csv(output / "alarm_budget_comparison.csv")
    if len(operating) != 2 or set(operating.candidate) != set(config["candidates"]):
        raise AssertionError("Candidatos finales incompletos")
    if len(classes) != 4 or len(sizes) != 12 or len(standard) != 6 or len(errors) != 2:
        raise AssertionError("Cobertura de métricas incompleta")
    if len(budgets) != 4 or set(np.round(budgets.alarm_budget, 2)) != {0.01, 0.02}:
        raise AssertionError("Comparación 1 % frente a 2 % incompleta")
    expected_grid = len(config["candidates"]) * len(config["threshold_search"]["smoke_thresholds"]) * len(config["threshold_search"]["fire_thresholds"])
    if len(grid) != expected_grid or grid[["candidate", "smoke_threshold", "fire_threshold"]].duplicated().any():
        raise AssertionError("Cuadrícula del 1 % incompleta o duplicada")
    budget = float(config["selection_policy"]["maximum_negative_alarm_rate"])
    if (operating.negative_alarm_rate > budget + 1e-12).any():
        raise AssertionError("Algún candidato supera el 1 % de alarmas negativas")
    if set(operating.negative_images) != {783} or set(operating.negative_images_with_alarm) != {7}:
        raise AssertionError("Denominador o alarmas negativas inesperados")
    expected_alarm_counts = {
        ("yolo26s", 0.01): 7, ("yolo26s", 0.02): 12,
        ("yolov8s", 0.01): 7, ("yolov8s", 0.02): 15,
    }
    for row in budgets.itertuples():
        key = (row.candidate, round(float(row.alarm_budget), 2))
        if int(row.negative_images_with_alarm) != expected_alarm_counts[key]:
            raise AssertionError(f"Alarmas inesperadas para {key}")
    for row in operating.itertuples():
        expected = config["candidates"][row.candidate]
        if not np.isclose(row.smoke_threshold, expected["smoke_threshold"]):
            raise AssertionError(f"Umbral de humo incorrecto: {row.candidate}")
        if not np.isclose(row.fire_threshold, expected["fire_threshold"]):
            raise AssertionError(f"Umbral de fuego incorrecto: {row.candidate}")
    winner = operating.sort_values("selection_rank").iloc[0]
    if winner.candidate != "yolo26s" or not bool(winner.selected):
        raise AssertionError("La selección no reproduce el ganador esperado")
    if not (winner.macro_recall > operating.loc[operating.candidate == "yolov8s", "macro_recall"].iloc[0]):
        raise AssertionError("El ganador no mejora el objetivo primario")
    if set(sizes.size_band) != {"small", "medium", "large"} or set(sizes.class_name) != {"smoke", "fire"}:
        raise AssertionError("Bandas de tamaño incompletas")

    report = {
        "status": "passed",
        "run_id": summary["run_id"],
        "selected_candidate": winner.candidate,
        "smoke_threshold": float(winner.smoke_threshold),
        "fire_threshold": float(winner.fire_threshold),
        "negative_alarm_rate": float(winner.negative_alarm_rate),
        "images": int(summary["images"]),
        "negative_images": int(summary["negative_images"]),
        "grid_points": len(grid),
        "test_inference_executed": False,
    }
    (output / "verification_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
