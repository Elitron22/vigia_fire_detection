"""Verifica la cuadrícula por clase y la revisión manual de falsos positivos."""

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
    parent = ROOT / "artifacts" / "10_class_threshold_fp_review" / "validation"
    pointer = pipeline.read_json(parent / "latest.json")
    output = ROOT / pointer["run_rel"]
    summary = pipeline.read_json(output / "run_summary.json")
    if summary["status"] != "complete" or summary["split"] != "val" or summary["test_inference_executed"]:
        raise AssertionError("Ejecución incompleta, split incorrecto o uso indebido de test.")
    if pipeline.sha256_file(output / "run_summary.json") != pointer["summary_sha256"]:
        raise AssertionError("El puntero latest no coincide con el resumen.")
    for relative, expected in summary["output_hashes"].items():
        if relative == "RESUMEN_UMBRALES_POR_CLASE.md":
            continue  # El resumen incorpora después la revisión visual humana.
        if pipeline.sha256_file(output / relative) != expected:
            raise AssertionError(f"Artefacto automático modificado: {relative}")

    config = summary["config"]
    grid = pd.read_csv(output / "class_threshold_grid.csv")
    recommendations = pd.read_csv(output / "profile_recommendations.csv")
    geometric = pd.read_csv(output / "fp_geometric_summary.csv")
    candidates = pd.read_csv(output / "fp_review_candidates.csv")
    manual = pd.read_csv(output / "fp_manual_review.csv")
    expected_grid = len(config["profiles"]) * len(config["smoke_thresholds"]) * len(config["fire_thresholds"])
    if len(grid) != expected_grid or grid[["profile", "smoke_threshold", "fire_threshold"]].duplicated().any():
        raise AssertionError("Cobertura o unicidad incorrectas en la cuadrícula.")
    if set(grid.negative_images) != {783}:
        raise AssertionError("Número de imágenes negativas incorrecto.")
    if len(recommendations) != len(config["profiles"]) * 5:
        raise AssertionError("Escenarios de recomendación incompletos.")
    for profile, table in recommendations.groupby("profile"):
        original = table[table.scenario == "original"].iloc[0]
        guarded = table[table.scenario == "precision_recall_guardrail"].iloc[0]
        if guarded.smoke_recall < original.smoke_recall - config["recall_tolerance"] - 1e-12:
            raise AssertionError(f"Guardrail de humo incumplido en {profile}.")
        if guarded.fire_recall < original.fire_recall - config["recall_tolerance"] - 1e-12:
            raise AssertionError(f"Guardrail de fuego incumplido en {profile}.")
        if guarded.micro_precision < original.micro_precision:
            raise AssertionError(f"La precisión no mejora en {profile}.")
        if guarded.negative_images_with_alarm > config["alarm_budget"] * guarded.negative_images + 1e-12:
            raise AssertionError(f"Presupuesto de alarmas incumplido en {profile}.")
    if int(geometric.false_positive_boxes.sum()) != 931:
        raise AssertionError("El resumen geométrico no reproduce las 931 cajas FP.")
    positive_boxes = geometric.loc[geometric.image_scope == "positive", "false_positive_boxes"].sum()
    negative_boxes = geometric.loc[geometric.image_scope == "negative", "false_positive_boxes"].sum()
    if (int(positive_boxes), int(negative_boxes)) != (917, 14):
        raise AssertionError("Desglose positivo/negativo incoherente.")
    if len(candidates) != config["gallery_images"] or candidates.filename.duplicated().any():
        raise AssertionError("Muestra automática incompleta.")
    if len(manual) != len(candidates) or set(manual.review_id) != set(candidates.review_id):
        raise AssertionError("La revisión manual no cubre toda la muestra.")
    if not manual.review_status.eq("inspected").all() or manual.manual_category.isna().any():
        raise AssertionError("Quedan casos visuales sin revisar.")
    if not set(manual.manual_category).issubset({
        "artificial_light", "possible_unannotated_fire", "haze_or_possible_smoke",
        "red_object_or_reflection", "localization_extent", "class_ambiguity", "duplicate",
    }):
        raise AssertionError("Categoría manual no reconocida.")

    report = {
        "status": "passed", "run_id": summary["run_id"], "grid_points": len(grid),
        "profiles": sorted(grid.profile.unique()), "false_positive_boxes": 931,
        "positive_image_fp_boxes": 917, "negative_image_fp_boxes": 14,
        "manual_images_reviewed": len(manual), "test_inference_executed": False,
    }
    (output / "verification_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
