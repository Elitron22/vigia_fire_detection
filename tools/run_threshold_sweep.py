"""Ejecuta el barrido inicial en val; caché verificable y salidas independientes."""
from __future__ import annotations

import argparse
import datetime as dt
import gc
import hashlib
import json
from pathlib import Path
import shutil
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tfm_pipeline as pipeline
import tfm_evaluation as evaluation
from tfm_thresholds import (
    choose_scenarios, evaluate_threshold, load_config, load_predictions,
    sweep_model, verify_baseline,
)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def get_prediction_cache(experiment, contract, config, *, offline=False):
    import torch
    import ultralytics
    from ultralytics import YOLO

    weights = Path(experiment["best_model"])
    identity = {
        "experiment_id": experiment["experiment_id"],
        "checkpoint_sha256": pipeline.sha256_file(weights),
        "manifest_sha256": pipeline.sha256_file(contract["manifest_path"]),
        "evaluation_code_sha256": pipeline.sha256_file(ROOT / "tfm_evaluation.py"),
        "ultralytics": ultralytics.__version__, "torch": torch.__version__,
        "protocol": {k: config[k] for k in ("split", "prediction_confidence", "match_iou", "nms_iou", "imgsz", "chunk_size", "seed")},
        "rect": False, "max_det": 300, "quantize": "fp32", "agnostic_nms": False,
    }
    cache_root = Path(experiment["experiment_root"]) / "evaluation" / "val" / "threshold_cache" / fingerprint(identity)[:16]
    pointer = cache_root / "cache.json"
    if pointer.exists():
        saved = pipeline.read_json(pointer)
        if saved["identity"] != identity:
            raise ValueError("La identidad de la caché no coincide.")
        for key in ("summary", "predictions"):
            path = ROOT / saved[f"{key}_rel"]
            if pipeline.sha256_file(path) != saved[f"{key}_sha256"]:
                raise ValueError(f"Caché modificada: {path}")
        if pipeline.read_json(ROOT / saved["summary_rel"])["status"] != "complete":
            raise ValueError("Caché incompleta.")
        print(f"{experiment['model_key']}: reutilizando predicciones verificadas", flush=True)
        return saved
    if offline:
        raise FileNotFoundError(f"Falta caché completa de {experiment['experiment_id']}. Ejecutar sin --offline.")
    if not torch.cuda.is_available():
        raise RuntimeError("Se necesita CUDA para generar las predicciones; el análisis posterior es offline.")
    staged = pipeline.stage_prepared_dataset(contract, workers=8)
    manifest = pipeline.rebased_manifest(contract, staged)
    model = YOLO(str(weights))
    pipeline.validate_class_mapping(model.names)
    print(f"{experiment['model_key']}: inferencia val a confianza {config['prediction_confidence']}", flush=True)
    output, _, summary = evaluation.run_error_analysis(
        model, manifest, cache_root / "runs", model_path=weights,
        manifest_path=contract["manifest_path"], split="val",
        conf=config["prediction_confidence"], match_iou=config["match_iou"],
        nms_iou=config["nms_iou"], imgsz=config["imgsz"],
        chunk_size=config["chunk_size"], device=0, seed=config["seed"],
        ram_limit_gib=config["ram_limit_gib"],
        expected_images=int((manifest.split == "val").sum()),
        expected_negatives=int(((manifest.split == "val") & (manifest.box_count == 0)).sum()),
    )
    paths = evaluation.split_artifact_paths(output, "val")
    saved = {"identity": identity, "end_to_end": summary["protocol"]["end_to_end"]}
    for key in ("summary", "predictions"):
        saved[f"{key}_rel"] = pipeline.project_relative(paths[key], ROOT)
        saved[f"{key}_sha256"] = pipeline.sha256_file(paths[key])
    pipeline.write_json_atomic(pointer, saved)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return saved


def run(config_path=None, *, offline=False, predictions_only=False):
    config_path = Path(config_path or ROOT / "configs" / "threshold_sweep.yaml")
    config = load_config(config_path)
    contract = pipeline.validate_prepared_dataset(ROOT, config["dataset_version"])
    # Rutas a originales solo para mostrar imágenes. GT corregidas desde la caché.
    manifest = pipeline.rebased_manifest(contract)
    manifest = manifest.loc[manifest.split == "val"].copy()
    index_path = ROOT / "artifacts" / "04_model_comparison" / "validation" / "evaluation_index.json"
    old_index = pipeline.read_json(index_path)
    if old_index["dataset_version"] != config["dataset_version"]:
        raise ValueError("La referencia pertenece a otro dataset.")
    historical = {r["experiment_id"]: r for r in old_index["evaluations"]}
    experiments, caches = {}, {}
    for model_key, experiment_id in config["experiments"].items():
        exp = pipeline.resolve_experiment(experiment_id=experiment_id, root=ROOT)
        if exp["model_key"] != model_key or exp["dataset_version"] != config["dataset_version"] or exp["status"] != "complete":
            raise ValueError("Experimento incompatible con la configuración.")
        if experiment_id not in historical:
            raise ValueError("Cada experimento necesita una referencia directa en val a 0,25 del notebook 03.")
        experiments[model_key] = exp
        caches[model_key] = get_prediction_cache(exp, contract, config, offline=offline)
    if predictions_only:
        print("Cachés completas. No se han generado conclusiones todavía.", flush=True)
        return None

    run_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + fingerprint(config)[:8]
    parent = ROOT / "artifacts" / "05_threshold_sweep" / "validation"
    output = parent / run_id
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(config_path, output / "config.yaml")
    source_files = ["tfm_thresholds.py", "tfm_evaluation.py", "tfm_pipeline.py", "tools/run_threshold_sweep.py", "tools/build_threshold_notebook.py", "tools/verify_threshold_sweep.py"]
    source_snapshot = output / "code"
    source_snapshot.mkdir()
    for name in source_files:
        if (ROOT / name).exists():
            shutil.copy2(ROOT / name, source_snapshot / Path(name).name)
    metadata = {
        "schema_version": 1, "status": "running", "run_id": run_id,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "dataset_version": config["dataset_version"], "split": "val", "config": config,
        "manifest_rel": pipeline.project_relative(contract["manifest_path"], ROOT),
        "manifest_sha256": pipeline.sha256_file(contract["manifest_path"]),
        "historical_index_rel": pipeline.project_relative(index_path, ROOT),
        "historical_index_sha256": pipeline.sha256_file(index_path),
        "inputs": caches, "environment": pipeline.environment_snapshot(),
        "decision_status": "diagnostic_scenarios_only_no_deployment_selection",
        "test_inference_executed": False,
    }
    pipeline.write_json_atomic(output / "run_summary.json", metadata)
    all_metrics, all_images, all_sizes, checks, payload_map = [], [], [], {}, {}
    try:
        for model_key, exp in experiments.items():
            print(f"{model_key}: calculando {len(config['thresholds'])} umbrales offline", flush=True)
            cache = caches[model_key]
            payloads = load_predictions(ROOT / cache["predictions_rel"], manifest, config["prediction_confidence"])
            payload_map[model_key] = payloads
            metrics, images, sizes = sweep_model(payloads, manifest, config, model_key)
            old_summary_path = ROOT / historical[exp["experiment_id"]]["error_summary_rel"]
            old_summary = pipeline.read_json(old_summary_path)
            if old_summary["protocol"]["confidence"] != config["reference_threshold"]:
                raise ValueError("Umbral histórico incompatible.")
            for actual, expected in ((old_summary["model_sha256"], cache["identity"]["checkpoint_sha256"]),
                                     (old_summary["manifest_sha256"], metadata["manifest_sha256"])):
                if actual != expected:
                    raise ValueError("Pesos o manifiesto distintos de la referencia histórica.")
            old_images_path = evaluation.split_artifact_paths(old_summary_path.parent, "val")["analysis"]
            checks[model_key] = verify_baseline(metrics, images, old_summary, pd.read_csv(old_images_path), config["reference_threshold"])
            checks[model_key].update(reference_summary_rel=pipeline.project_relative(old_summary_path, ROOT),
                                     reference_summary_sha256=pipeline.sha256_file(old_summary_path),
                                     reference_images_sha256=pipeline.sha256_file(old_images_path))
            pipeline.write_json_atomic(output / "baseline_verification.json", checks)
            if checks[model_key]["status"] != "passed":
                raise AssertionError(f"La caché filtrada no reproduce la referencia directa: {checks[model_key]}")
            all_metrics.append(metrics)
            all_images.append(images)
            all_sizes.append(sizes)
        metrics = pd.concat(all_metrics, ignore_index=True)
        images = pd.concat(all_images, ignore_index=True)
        sizes = pd.concat(all_sizes, ignore_index=True)
        scenarios = choose_scenarios(metrics, config)
        metrics.to_csv(output / "threshold_metrics.csv", index=False)
        images.to_csv(output / "image_metrics.csv", index=False)
        sizes.to_csv(output / "size_metrics.csv", index=False)
        scenarios.to_csv(output / "scenario_candidates.csv", index=False)
        records = {r["filename"]: r for r in manifest.to_dict("records")}
        gt_frames, det_frames = [], []
        for model_key in experiments:
            candidate_thresholds = set(scenarios.loc[(scenarios.model_key == model_key) & scenarios.feasible, "threshold"])
            candidate_thresholds.update([config["prediction_confidence"], config["reference_threshold"]])
            for threshold in sorted(candidate_thresholds):
                _, _, _, gt, detections = evaluate_threshold(payload_map[model_key], records, threshold, config, model_key, details=True)
                gt_frames.append(gt)
                det_frames.append(detections)
        gt_details = pd.concat(gt_frames, ignore_index=True)
        detection_details = pd.concat(det_frames, ignore_index=True)
        gt_details.to_csv(output / "ground_truth_review.csv", index=False)
        detection_details.to_csv(output / "detection_review.csv", index=False)
        # El renderizador usa exclusivamente los resultados ya calculados.
        from tfm_thresholds import build_figures, build_review_galleries, write_findings
        build_figures(metrics, scenarios, sizes, gt_details, detection_details, config, output)
        review = build_review_galleries(payload_map, records, scenarios, gt_details, detection_details, config, output)
        write_findings(metrics, scenarios, sizes, gt_details, detection_details, config, output)
        for model_key, exp in experiments.items():
            if pipeline.sha256_file(Path(exp["best_model"])) != caches[model_key]["identity"]["checkpoint_sha256"]:
                raise AssertionError("Los pesos han cambiado durante la evaluación.")
        if pipeline.sha256_file(contract["manifest_path"]) != metadata["manifest_sha256"] or pipeline.sha256_file(index_path) != metadata["historical_index_sha256"]:
            raise AssertionError("Las fuentes han cambiado durante la evaluación.")
        metadata.update(status="complete", images_per_model=len(manifest), negative_images=int((manifest.box_count == 0).sum()),
                        threshold_points=len(metrics), baseline_verification=checks, review_images=len(review),
                        output_hashes={p.relative_to(output).as_posix(): pipeline.sha256_file(p)
                                       for p in sorted(output.rglob("*")) if p.is_file() and p.name != "run_summary.json"})
        pipeline.write_json_atomic(output / "run_summary.json", metadata)
        pipeline.write_json_atomic(parent / "latest.json", {"run_id": run_id, "run_rel": pipeline.project_relative(output, ROOT),
                                                            "summary_sha256": pipeline.sha256_file(output / "run_summary.json")})
        print(f"Barrido completo: {output}", flush=True)
        print(scenarios[["model_key", "scenario", "threshold", "smoke_recall", "fire_recall", "micro_f1", "negative_images_with_alarm"]].to_string(index=False))
        return output
    except BaseException as exc:
        metadata.update(status="incomplete", error=f"{type(exc).__name__}: {exc}")
        pipeline.write_json_atomic(output / "run_summary.json", metadata)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "threshold_sweep.yaml")
    parser.add_argument("--offline", action="store_true", help="Exige cachés existentes; no infiere.")
    parser.add_argument("--predictions-only", action="store_true", help="Prepara cachés sin calcular el barrido.")
    args = parser.parse_args()
    run(args.config, offline=args.offline, predictions_only=args.predictions_only)


if __name__ == "__main__":
    main()
