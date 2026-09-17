"""Congela el checkpoint y el punto operativo elegidos antes de abrir test."""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
import shutil
import sys

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tfm_pipeline as pipeline


def load_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("schema_version no compatible")
    if config.get("split") != "test" or config.get("selection_locked") is not True:
        raise ValueError("La configuración final debe estar bloqueada a test")
    if config.get("single_final_evaluation") is not True:
        raise ValueError("Debe declararse una única evaluación final")
    if config.get("threshold_search_allowed") is not False:
        raise ValueError("No se permite buscar umbrales en test")
    model = config["model"]
    point = config["operating_point"]
    expected = {
        "model_key": "yolo26s", "experiment_id": "yolo26s_dfire_seed42_20260912T165304Z",
        "trained_imgsz": 768, "inference_imgsz": 768,
    }
    for key, value in expected.items():
        if model.get(key) != value:
            raise ValueError(f"Modelo final inesperado: {key}={model.get(key)!r}")
    if float(point["smoke_threshold"]) != .36 or float(point["fire_threshold"]) != .16:
        raise ValueError("Los umbrales finales no coinciden con 0,36/0,16")
    if float(point["maximum_negative_alarm_rate"]) != .01:
        raise ValueError("El límite operativo debe ser 1 %")
    return config


def verify_existing(output: Path, config: dict) -> Path:
    manifest_path = output / "freeze_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"Existe un congelado incompleto: {output}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise RuntimeError("El congelado existente no está completo")
    frozen = output / manifest["frozen_weights_rel"]
    if pipeline.sha256_file(frozen) != manifest["weights_sha256"]:
        raise RuntimeError("El checkpoint congelado ha cambiado")
    if manifest["weights_sha256"] != config["model"]["source_weights_sha256"]:
        raise RuntimeError("El congelado existente no coincide con la configuración")
    return output


def freeze(config_path: str | Path) -> Path:
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config = load_config(config_path.resolve())
    output = ROOT / config["freeze_parent"] / "final"
    if output.exists():
        return verify_existing(output, config)

    source = ROOT / config["model"]["source_weights"]
    descriptor = source.parents[2] / "experiment.json"
    if not source.is_file() or not descriptor.is_file():
        raise FileNotFoundError("Faltan los pesos o el descriptor del experimento")
    experiment = json.loads(descriptor.read_text(encoding="utf-8"))
    if experiment.get("status") != "complete" or experiment.get("experiment_id") != config["model"]["experiment_id"]:
        raise ValueError("El experimento seleccionado no está completo o no coincide")
    source_hash = pipeline.sha256_file(source)
    if source_hash != config["model"]["source_weights_sha256"]:
        raise ValueError("El hash de los pesos actuales no coincide con el contrato")
    if experiment.get("dataset_manifest_sha256") != "97442af38322abe1bf5178bd6b92c30971d9f354eca3197762bb7a4a4b8c8eb6":
        raise ValueError("El manifiesto del entrenamiento no coincide con el dataset congelado")

    validation = ROOT / config["validation_source"]
    operating = pd.read_csv(validation / "final_operating_points.csv")
    row = operating[(operating.candidate == "yolo26s") & operating.selected.astype(bool)].iloc[0]
    if abs(float(row.smoke_threshold) - .36) > 1e-12 or abs(float(row.fire_threshold) - .16) > 1e-12:
        raise ValueError("La selección de validación no reproduce los umbrales finales")

    output.mkdir(parents=True, exist_ok=False)
    (output / "weights").mkdir()
    frozen = output / "weights" / "best.pt"
    shutil.copy2(source, frozen)
    if pipeline.sha256_file(frozen) != source_hash:
        raise RuntimeError("La copia congelada no conserva el hash")
    shutil.copy2(config_path, output / "final_test_evaluation.yaml")
    shutil.copy2(descriptor, output / "experiment.json")
    shutil.copy2(validation / "run_summary.json", output / "validation_selection_run_summary.json")

    prior = ROOT / config["known_prior_test_exposure"]
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "frozen_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "selection_stage": "before_final_yolo26s_test_evaluation",
        "model_label": config["model"]["label"],
        "model_key": config["model"]["model_key"],
        "experiment_id": config["model"]["experiment_id"],
        "trained_imgsz": 768,
        "inference_imgsz": 768,
        "smoke_threshold": .36,
        "fire_threshold": .16,
        "maximum_negative_alarm_rate": .01,
        "source_weights_rel": config["model"]["source_weights"],
        "frozen_weights_rel": "weights/best.pt",
        "weights_sha256": source_hash,
        "weights_bytes": frozen.stat().st_size,
        "dataset_version": config["dataset_version"],
        "dataset_manifest_sha256": experiment["dataset_manifest_sha256"],
        "validation_source_rel": config["validation_source"],
        "validation_source_sha256": pipeline.sha256_file(validation / "run_summary.json"),
        "selection_rationale": "Selección explícita del usuario: equilibrio entre clases y mayor recall de fuego frente al candidato 640→640.",
        "test_policy": {
            "single_final_evaluation": True,
            "threshold_search_allowed": False,
            "model_comparison_allowed": False,
            "post_test_changes_allowed": False,
        },
        "known_prior_test_exposure": {
            "exists": prior.is_file(),
            "model": "legacy YOLOv8s baseline",
            "path_rel": config["known_prior_test_exposure"],
            "note": "Exposición histórica anterior; el YOLO26s congelado no había sido evaluado en test.",
        },
    }
    pipeline.write_json_atomic(output / "freeze_manifest.json", manifest)
    pipeline.write_json_atomic(ROOT / config["freeze_parent"] / "latest.json", {
        "run_rel": pipeline.project_relative(output, ROOT),
        "weights_sha256": source_hash,
        "freeze_manifest_sha256": pipeline.sha256_file(output / "freeze_manifest.json"),
    })
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/final_test_evaluation.yaml")
    args = parser.parse_args()
    output = freeze(args.config)
    print(f"Modelo final congelado: {output}")


if __name__ == "__main__":
    main()

