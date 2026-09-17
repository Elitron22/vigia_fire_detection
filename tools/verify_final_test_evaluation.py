"""Verificación independiente del único artefacto final de test."""
from __future__ import annotations

import argparse
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


def assert_close(actual, expected, message, tolerance=1e-10):
    if not np.isclose(float(actual), float(expected), atol=tolerance, rtol=0):
        raise AssertionError(f"{message}: {actual} != {expected}")


def verify(output: Path | None = None) -> Path:
    parent = ROOT / "artifacts/15_final_test_evaluation/test"
    state = json.loads((parent / "evaluation_state.json").read_text(encoding="utf-8"))
    if state.get("status") != "complete" or state.get("single_final_evaluation") is not True:
        raise AssertionError("La evaluación final no está cerrada como ejecución única")
    if output is None:
        output = ROOT / state["run_rel"]
    output = output.resolve()
    if output != (ROOT / state["run_rel"]).resolve():
        raise AssertionError("El directorio no coincide con el estado global")
    completed_runs = [p for p in parent.iterdir() if p.is_dir() and (p / "run_summary.json").is_file()]
    if completed_runs != [output]:
        raise AssertionError(f"Se esperaba una sola ejecución completa: {completed_runs}")

    summary = json.loads((output / "run_summary.json").read_text(encoding="utf-8"))
    config = yaml.safe_load((output / "config.yaml").read_text(encoding="utf-8"))
    freeze = json.loads((output / "freeze_manifest.json").read_text(encoding="utf-8"))
    if summary.get("status") != "complete" or summary.get("split") != "test":
        raise AssertionError("Resumen incompleto o partición incorrecta")
    if not summary.get("test_inference_executed"):
        raise AssertionError("No consta la inferencia final de test")
    if summary.get("threshold_search_executed") or summary.get("model_comparison_executed"):
        raise AssertionError("Se ejecutó una operación prohibida en test")
    if summary.get("experiment_id") != "yolo26s_dfire_seed42_20260912T165304Z":
        raise AssertionError("Se evaluó un modelo distinto del congelado")
    if config.get("threshold_search_allowed") is not False or config.get("selection_locked") is not True:
        raise AssertionError("El contrato no bloquea selección y umbrales")
    if summary["weights_sha256"] != freeze["weights_sha256"] != config["model"]["source_weights_sha256"]:
        raise AssertionError("Los hashes del checkpoint no coinciden")
    for field, expected in (("trained_imgsz", 768), ("inference_imgsz", 768),
                            ("smoke_threshold", .36), ("fire_threshold", .16),
                            ("maximum_negative_alarm_rate", .01)):
        assert_close(summary[field], expected, field)
    expected_population = {"images": 4306, "negative_images": 2005, "smoke_boxes": 2311, "fire_boxes": 2878}
    if summary["observed_test_population"] != expected_population:
        raise AssertionError("La población de test no coincide")

    standard = pd.read_csv(output / "test_standard_metrics.csv")
    classes = pd.read_csv(output / "test_class_metrics.csv")
    operating = pd.read_csv(output / "test_operating_metrics.csv")
    alarms = pd.read_csv(output / "test_negative_image_alarms.csv")
    sizes = pd.read_csv(output / "test_size_metrics.csv")
    images = pd.read_csv(output / "test_image_metrics.csv")
    gt = pd.read_csv(output / "test_ground_truth_details.csv")
    detections = pd.read_csv(output / "test_detection_details.csv")
    confusion = pd.read_csv(output / "test_confusion_matrix.csv", index_col=0)
    if len(standard) != 3 or set(standard.scope) != {"all_macro", "smoke", "fire"}:
        raise AssertionError("Métricas estándar incompletas")
    if len(images) != 4306 or images.filename.duplicated().any():
        raise AssertionError("Cobertura por imagen incorrecta")
    if int(images.is_negative.sum()) != 2005 or int(images.gt_count.sum()) != 5189:
        raise AssertionError("Denominadores de imagen o GT incorrectos")
    if len(gt) != 5189 or len(sizes) != 6:
        raise AssertionError("Detalle GT o bandas de tamaño incompletos")
    if set(classes.scope) != {"smoke", "fire", "all_micro"} or len(operating) != 1:
        raise AssertionError("Métricas operativas incompletas")
    if not (images.smoke_tp + images.smoke_fn == images.smoke_gt).all():
        raise AssertionError("No reconcilian TP+FN de humo")
    if not (images.fire_tp + images.fire_fn == images.fire_gt).all():
        raise AssertionError("No reconcilian TP+FN de fuego")
    if not (images.smoke_tp + images.smoke_fp == images.smoke_pred).all():
        raise AssertionError("No reconcilian TP+FP de humo")
    if not (images.fire_tp + images.fire_fp == images.fire_pred).all():
        raise AssertionError("No reconcilian TP+FP de fuego")
    for cls in ("smoke", "fire"):
        row = classes[classes.scope == cls].iloc[0]
        if int(row.tp) != int(images[f"{cls}_tp"].sum()) or int(row.fp) != int(images[f"{cls}_fp"].sum()) or int(row.fn) != int(images[f"{cls}_fn"].sum()):
            raise AssertionError(f"No reconcilian las métricas de {cls}")
    if int(sizes.gt_boxes.sum()) != 5189 or int(sizes.tp.sum()) != int(operating.tp.iloc[0]):
        raise AssertionError("El recall por tamaño no reconcilia")
    if confusion.to_numpy().sum() != int(operating.tp.iloc[0] + operating.fp.iloc[0] + operating.fn.iloc[0] - (
            confusion.loc["smoke", "fire"] + confusion.loc["fire", "smoke"])):
        raise AssertionError("La matriz de confusión no reconcilia FP/FN cruzados")
    for cls in ("smoke", "fire"):
        row = classes[classes.scope == cls].iloc[0]
        if int(confusion.loc[cls].sum()) != int(row.tp + row.fn):
            raise AssertionError(f"Las filas de confusión no reconcilian GT {cls}")
        if int(confusion[cls].sum()) != int(row.tp + row.fp):
            raise AssertionError(f"Las columnas de confusión no reconcilian predicciones {cls}")
    alarm = alarms[alarms.scope == "any"].iloc[0]
    expected_alarm_count = int(((images.is_negative) & (images.pred_count > 0)).sum())
    if int(alarm.negative_images_with_alarm) != expected_alarm_count or int(alarm.negative_images) != 2005:
        raise AssertionError("Las alarmas negativas no reconcilian")
    expected_rate = expected_alarm_count / 2005
    assert_close(alarm.negative_image_false_alarm_rate, expected_rate, "tasa de alarma negativa")

    forbidden = list(output.glob("*threshold_grid*")) + list(output.glob("*ranking*")) + list(output.glob("*model_comparison*"))
    if forbidden:
        raise AssertionError(f"Artefactos prohibidos en test: {forbidden}")
    for rel, digest in summary["output_hashes"].items():
        path = output / rel
        if not path.is_file() or pipeline.sha256_file(path) != digest:
            raise AssertionError(f"Hash incorrecto: {rel}")
    if not freeze["known_prior_test_exposure"]["exists"]:
        raise AssertionError("No se ha reconocido la exposición histórica de test")
    app = yaml.safe_load((ROOT / "configs/app.yaml").read_text(encoding="utf-8"))
    assert_close(app["alerts"]["minimum_confidence"]["smoke"], .36, "umbral app humo")
    assert_close(app["alerts"]["minimum_confidence"]["fire"], .16, "umbral app fuego")

    report = {
        "status": "passed", "single_complete_test_runs": 1,
        "model": summary["model_label"], "experiment_id": summary["experiment_id"],
        "test_images": len(images), "negative_images": int(alarm.negative_images),
        "negative_images_with_alarm": int(alarm.negative_images_with_alarm),
        "negative_alarm_rate": float(alarm.negative_image_false_alarm_rate),
        "budget_met": bool(float(alarm.negative_image_false_alarm_rate) <= .01),
        "threshold_search_executed": False, "model_comparison_executed": False,
        "historical_test_exposure_acknowledged": True,
        "checks": ["checkpoint_hash", "single_run_guard", "population", "standard_metrics",
                   "tp_fp_fn", "size_recall", "confusion_matrix", "negative_alarms",
                   "validation_test_comparisons", "artifact_hashes", "application_thresholds"],
    }
    target = output / "verification_report.json"
    pipeline.write_json_atomic(target, report)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default=None)
    args = parser.parse_args()
    target = verify(Path(args.run) if args.run else None)
    print(target)
    print(target.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()

