"""Verificación independiente del barrido de todos los modelos al 1 %."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import tfm_pipeline as pipeline


ROOT = pipeline.project_root()
PARENT = ROOT / "artifacts" / "13_all_models_01pct_comparison" / "validation"


def verify() -> dict:
    pointer = pipeline.read_json(PARENT / "latest.json")
    output = ROOT / pointer["run_rel"]
    summary_path = output / "run_summary.json"
    if pipeline.sha256_file(summary_path) != pointer["summary_sha256"]:
        raise AssertionError("El resumen no coincide con latest.json.")
    summary = pipeline.read_json(summary_path)
    if summary.get("status") != "complete" or summary.get("split") != "val":
        raise AssertionError("La ejecución no está completa o no usa validación.")
    if summary.get("test_inference_executed") is not False:
        raise AssertionError("El conjunto test no debe intervenir.")
    if not np.isclose(summary["alarm_budget"], 0.01):
        raise AssertionError("Presupuesto de alarmas incorrecto.")

    config = __import__("yaml").safe_load((output / "config.yaml").read_text(encoding="utf-8"))
    if config.get("test_locked") is not True or config["threshold_search"] != {
        "minimum": 0.16, "maximum": 0.60, "step": 0.01
    }:
        raise AssertionError("El protocolo final de umbrales no coincide.")

    operating = pd.read_csv(output / "operating_points.csv")
    ranked = pd.read_csv(output / "ranked_operating_points.csv")
    classes = pd.read_csv(output / "class_metrics_sensitivity.csv")
    sizes = pd.read_csv(output / "size_metrics_sensitivity.csv")
    standards = pd.read_csv(output / "standard_metrics.csv")
    grid = pd.read_csv(output / "class_threshold_grid.csv")
    sources = pd.read_csv(output / "prediction_sources.csv")

    expected_keys = set(config["configurations"])
    if set(operating.configuration) != expected_keys or len(operating) != 36:
        raise AssertionError("Puntos operativos incompletos.")
    if set(operating.scenario) != {"sensitivity", "balanced"}:
        raise AssertionError("Escenarios incompletos.")
    if len(grid) != 18 * 45 * 45 or grid.groupby("configuration").size().nunique() != 1:
        raise AssertionError("La cuadrícula no contiene 2.025 pares por configuración.")
    if int(grid.groupby("configuration").size().iloc[0]) != 2025:
        raise AssertionError("Tamaño incorrecto de la cuadrícula por configuración.")
    if (operating.negative_images_with_alarm > 7).any() or (operating.negative_alarm_rate > .01).any():
        raise AssertionError("Algún punto seleccionado supera el 1 %.")
    if len(classes) != 36 or set(classes.class_name) != {"smoke", "fire"}:
        raise AssertionError("Métricas por clase incompletas.")
    if len(sizes) != 108 or set(sizes.size_band) != {"small", "medium", "large"}:
        raise AssertionError("Métricas por tamaño incompletas.")
    if len(standards) != 42 or set(standards.scope) != {"all", "smoke", "fire"}:
        raise AssertionError("Métricas estándar incompletas para las 14 configuraciones principales.")
    if len(sources) != 18 or set(sources.configuration) != expected_keys:
        raise AssertionError("Fuentes de predicción incompletas.")
    for row in sources.itertuples():
        prediction_path = ROOT / row.predictions_rel
        if pipeline.sha256_file(prediction_path) != row.predictions_sha256:
            raise AssertionError(f"Predicciones modificadas: {row.configuration}")

    sensitivity = ranked[
        (ranked.comparison_group == "full_training") & (ranked.scenario == "sensitivity")
    ].sort_values("rank")
    balanced = ranked[
        (ranked.comparison_group == "full_training") & (ranked.scenario == "balanced")
    ].sort_values("rank")
    if sensitivity.iloc[0].configuration != "yolo26s_640_eval640":
        raise AssertionError("El ganador de recall macro no coincide.")
    if balanced.iloc[0].configuration != "yolov8s_768_eval640":
        raise AssertionError("El ganador de F1 no coincide.")
    selected = sensitivity.iloc[0]
    if not (np.isclose(selected.smoke_threshold, .16) and np.isclose(selected.fire_threshold, .16)):
        raise AssertionError("Umbrales del ganador incorrectos.")

    previous_26 = sensitivity[sensitivity.configuration == "yolo26s_768_eval768"].iloc[0]
    previous_8 = sensitivity[sensitivity.configuration == "yolov8s_768_eval640"].iloc[0]
    if not (np.isclose(previous_26.smoke_threshold, .36)
            and np.isclose(previous_26.fire_threshold, .16)
            and np.isclose(previous_8.smoke_threshold, .49)
            and np.isclose(previous_8.fire_threshold, .17)):
        raise AssertionError("La ampliación no reproduce la comparación final anterior.")

    for row in operating.itertuples():
        tp = row.smoke_tp + row.fire_tp
        fp = row.smoke_fp + row.fire_fp
        fn = row.smoke_fn + row.fire_fn
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * tp / (2 * tp + fp + fn)
        macro = (row.smoke_recall + row.fire_recall) / 2
        if not all(np.isclose(a, b) for a, b in (
            (precision, row.micro_precision), (recall, row.micro_recall),
            (f1, row.micro_f1), (macro, row.macro_recall)
        )):
            raise AssertionError(f"Métricas no reconciliadas: {row.configuration}/{row.scenario}")

    report = {
        "status": "passed",
        "run_id": summary["run_id"],
        "configurations": 18,
        "full_training_configurations": 14,
        "hp_screening_configurations": 4,
        "class_threshold_points": len(grid),
        "maximum_negative_images_with_alarm": int(operating.negative_images_with_alarm.max()),
        "sensitivity_winner": sensitivity.iloc[0].configuration,
        "balanced_winner": balanced.iloc[0].configuration,
        "previous_comparison_reproduced": True,
        "test_inference_executed": False,
    }
    pipeline.write_json_atomic(output / "verification_report.json", report)
    return report


if __name__ == "__main__":
    print(json.dumps(verify(), ensure_ascii=False, indent=2))
