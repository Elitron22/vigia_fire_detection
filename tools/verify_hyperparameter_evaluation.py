"""Verifica una ejecución completa de la evaluación de hiperparámetros."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tfm_pipeline as pipeline

PARENT = ROOT / "artifacts" / "11_yolo26s_hyperparameter_search" / "validation"


def verify(run_id: str | None = None) -> dict:
    pointer = pipeline.read_json(PARENT / "latest.json")
    if pointer.get("status") != "complete":
        raise AssertionError("El puntero no identifica una ejecución completa.")
    if run_id is not None and pointer["run_id"] != run_id:
        raise AssertionError("El run_id solicitado no coincide con latest.json.")
    output = ROOT / pointer["output_rel"]
    if pipeline.sha256_file(output / "run.json") != pointer["run_sha256"]:
        raise AssertionError("run.json no coincide con el puntero.")
    if pipeline.sha256_file(ROOT / pointer["summary_rel"]) != pointer["summary_sha256"]:
        raise AssertionError("El resumen no coincide con el puntero.")
    metadata = pipeline.read_json(output / "run.json")
    if metadata.get("status") != "complete" or metadata.get("split") != "val":
        raise AssertionError("Estado o split no válidos.")
    if metadata.get("test_inference_executed") is not False:
        raise AssertionError("No puede existir inferencia de test.")
    for relative, digest in metadata["files"].items():
        if pipeline.sha256_file(ROOT / relative) != digest:
            raise AssertionError(f"Artefacto modificado: {relative}")

    config = yaml.safe_load((output / "config.yaml").read_text(encoding="utf-8"))
    training = pd.read_csv(output / "training_screening.csv")
    candidates = pd.read_csv(output / "preliminary_candidates.csv")
    grid = pd.read_csv(output / "class_threshold_grid.csv")
    recommendations = pd.read_csv(output / "operating_recommendations.csv")
    winners = pd.read_csv(output / "overall_winners.csv")
    if len(training) != 5 or set(training.trial_id) != {"baseline", *config["active_trials"]}:
        raise AssertionError("La tabla de cribado no contiene las cinco configuraciones.")
    if not training.epochs_compared.eq(config["screening_epochs"]).all():
        raise AssertionError("Los horizontes de entrenamiento no son comparables.")
    selected = candidates[candidates.selected_for_operational_evaluation]
    if len(selected) != config["evaluation"]["candidate_count"]:
        raise AssertionError("Número incorrecto de candidatos preliminares.")
    expected_grid = (len(selected) * len(config["evaluation"]["smoke_thresholds"])
                     * len(config["evaluation"]["fire_thresholds"]))
    if len(grid) != expected_grid or grid[["profile", "smoke_threshold", "fire_threshold"]].duplicated().any():
        raise AssertionError("La cuadrícula por clase está incompleta o duplicada.")
    if len(recommendations) != len(selected) * 2 or set(recommendations.scenario) != {"sensitivity", "balanced"}:
        raise AssertionError("Recomendaciones operativas incompletas.")
    allowed = recommendations.negative_images * config["evaluation"]["alarm_budget"] + 1e-12
    if not recommendations.negative_images_with_alarm.le(allowed).all():
        raise AssertionError("Una recomendación incumple el presupuesto de alarmas.")
    if len(winners) != 2 or set(winners.scenario) != {"sensitivity", "balanced"}:
        raise AssertionError("Ganadores globales incompletos.")
    return {
        "status": "passed", "run_id": pointer["run_id"],
        "screening_configurations": len(training),
        "preliminary_candidates": len(selected), "class_threshold_points": len(grid),
        "operating_recommendations": len(recommendations),
        "test_inference_executed": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id")
    args = parser.parse_args()
    print(verify(args.run_id))


if __name__ == "__main__":
    main()
