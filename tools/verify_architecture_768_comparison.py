"""Verifica integridad y coherencia de la última comparación de arquitecturas."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tfm_pipeline as pipeline


def main() -> None:
    parent = ROOT / "artifacts" / "09_architecture_768_comparison" / "validation"
    pointer = pipeline.read_json(parent / "latest.json")
    output = ROOT / pointer["run_rel"]
    summary = pipeline.read_json(output / "run_summary.json")
    if summary["status"] != "complete" or summary["split"] != "val":
        raise AssertionError("La ejecución no está completa o no pertenece a validación.")
    if summary["test_inference_executed"]:
        raise AssertionError("La comparación no debe consultar test.")
    if pipeline.sha256_file(output / "run_summary.json") != pointer["summary_sha256"]:
        raise AssertionError("El resumen no coincide con el puntero latest.")
    for relative, expected_hash in summary["output_hashes"].items():
        if pipeline.sha256_file(output / relative) != expected_hash:
            raise AssertionError(f"Artefacto modificado: {relative}")

    config = summary["config"]
    metrics = pd.read_csv(output / "threshold_metrics.csv")
    scenarios = pd.read_csv(output / "scenario_candidates.csv")
    selected = pd.read_csv(output / "selected_operating_points.csv")
    standard = pd.read_csv(output / "standard_metrics.csv")
    paired = pd.read_csv(output / "paired_comparison.csv")
    expected_configurations = len(config["candidates"]) * len(config["evaluation_sizes"])
    expected_points = expected_configurations * len(config["thresholds"])
    if len(metrics) != expected_points or metrics.model_key.nunique() != expected_configurations:
        raise AssertionError("Cobertura del barrido incorrecta.")
    if set(metrics.images) != {1721} or set(metrics.negative_images) != {783}:
        raise AssertionError("Cobertura de imágenes incorrecta.")
    if len(selected) != expected_configurations or not selected.feasible.all():
        raise AssertionError("Faltan puntos operativos factibles.")
    if not (selected.negative_images_with_alarm <= 0.02 * selected.negative_images + 1e-12).all():
        raise AssertionError("Algún punto excede el límite de falsas alarmas.")
    for row in selected.itertuples():
        source = metrics[(metrics.model_key == row.model_key) & np.isclose(metrics.threshold, row.threshold)]
        if len(source) != 1 or not np.isclose(source.iloc[0].micro_f1, row.micro_f1):
            raise AssertionError(f"Punto seleccionado no reproducible: {row.model_key}")
    if len(standard) != expected_configurations * 3 or set(standard.scope) != {"all", "smoke", "fire"}:
        raise AssertionError("Métricas estándar incompletas.")
    if len(paired) != len(config["evaluation_sizes"]) * 2 * 2:
        raise AssertionError("Comparaciones emparejadas incompletas.")
    if not paired.p_value.between(0, 1).all() or not (paired.discordant == paired.gains + paired.losses).all():
        raise AssertionError("Pruebas emparejadas incoherentes.")
    if set(scenarios.scenario) != {"reference_025", "max_f1", "alarm_02pct"}:
        raise AssertionError("Escenarios incompletos.")

    report = {
        "status": "passed",
        "run_id": summary["run_id"],
        "configurations": expected_configurations,
        "thresholds": len(config["thresholds"]),
        "metric_points": len(metrics),
        "images_per_configuration": 1721,
        "negative_images": 783,
        "test_inference_executed": False,
    }
    (output / "verification_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
