"""Verifica cobertura, procedencia y reconciliación del análisis de interpretabilidad."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import tfm_pipeline as pipeline


def verify(run: Path|None=None)->Path:
    parent=ROOT/"artifacts/16_model_interpretability/validation"
    latest=json.loads((parent/"latest.json").read_text(encoding="utf-8"))
    output=(ROOT/latest["run_rel"]).resolve() if run is None else run.resolve()
    summary=json.loads((output/"run_summary.json").read_text(encoding="utf-8"))
    config=yaml.safe_load((output/"config.yaml").read_text(encoding="utf-8"))
    if summary.get("status")!="complete" or summary.get("split")!="val": raise AssertionError("Ejecución incompleta o split incorrecto")
    if summary.get("test_inference_executed") or summary.get("threshold_search_executed"): raise AssertionError("Test o barrido no debían ejecutarse")
    if summary["model"]["weights_sha256"]!="bfb54c4726519fb473bb7c4825577e51c05af2b0db5fd46bf1931a8a29285f67": raise AssertionError("Modelo incorrecto")
    if summary.get("methods")!=["multiscale_eigencam","occlusion_sensitivity","deletion_faithfulness"]: raise AssertionError("Métodos incompletos")
    if float(summary["operating_point"]["smoke_threshold"])!=.36 or float(summary["operating_point"]["fire_threshold"])!=.16: raise AssertionError("Umbrales incorrectos")
    selection=pd.read_csv(output/"selected_cases.csv")
    metrics=pd.read_csv(output/"interpretability_metrics.csv")
    curves=pd.read_csv(output/"deletion_curves.csv")
    expected={(mode,cls) for mode in ("true_positive","negative_false_alarm","borderline_false_negative") for cls in ("smoke","fire")}
    if len(selection)!=6 or set(zip(selection.case,selection.class_name))!=expected: raise AssertionError("Casos incompletos")
    if selection.filename.duplicated().any() or set(metrics.filename)!=set(selection.filename): raise AssertionError("Casos repetidos o desalineados")
    for row in selection.itertuples():
        if row.case in {"true_positive","negative_false_alarm"} and row.cached_target_confidence < row.operating_threshold: raise AssertionError("Acierto/alarma bajo umbral")
        if row.case=="borderline_false_negative" and row.cached_target_confidence >= row.operating_threshold: raise AssertionError("Omisión no limítrofe")
    contract=pipeline.validate_prepared_dataset(ROOT,config["dataset_version"])
    manifest=pd.read_csv(contract["manifest_path"])
    val_names=set(manifest[manifest.split=="val"].filename)
    test_names=set(manifest[manifest.split=="test"].filename)
    if not set(selection.filename)<=val_names or set(selection.filename)&test_names: raise AssertionError("La muestra no es exclusiva de validación")
    numeric=["fresh_target_confidence","occlusion_baseline_confidence","max_single_occlusion_drop",
             "mean_positive_occlusion_drop","top25_score_retained","random25_score_retained",
             "occlusion_mass_in_target","eigencam_mass_in_target"]
    if not np.isfinite(metrics[numeric].to_numpy(dtype=float)).all(): raise AssertionError("Métricas no finitas")
    if (metrics[["fresh_target_confidence","occlusion_baseline_confidence"]]<=0).any().any(): raise AssertionError("Objetivos no reproducidos")
    fractions=list(map(float,config["occlusion"]["deletion_fractions"])); repeats=int(config["occlusion"]["random_repeats"])
    expected_rows=6*len(fractions)*(1+repeats)
    if len(curves)!=expected_rows or set(np.round(curves.fraction,8))!=set(np.round(fractions,8)): raise AssertionError("Curvas incompletas")
    per_case=curves.groupby(["filename","fraction","strategy"]).size().unstack(fill_value=0)
    if not (per_case.important==1).all() or not (per_case.random==repeats).all(): raise AssertionError("Repeticiones incorrectas")
    maps=list((output/"maps").glob("*.npz")); case_figures=list((output/"figures/cases").glob("*.png"))
    if len(maps)!=6 or len(case_figures)!=6: raise AssertionError("Mapas o figuras de casos incompletos")
    for path in maps:
        with np.load(path) as values:
            if values["occlusion"].shape!=(6,6) or values["eigencam"].ndim!=2: raise AssertionError(f"Forma de mapa incorrecta: {path}")
            if not np.isfinite(values["occlusion"]).all() or not np.isfinite(values["eigencam"]).all(): raise AssertionError(f"Mapa no finito: {path}")
    required_figures={"07_deletion_curve.png","08_faithfulness_by_case.png","09_interpretability_contact_sheet.png"}
    if not required_figures<=set(path.name for path in (output/"figures").glob("*.png")): raise AssertionError("Faltan figuras de síntesis")
    report=(output/"RESUMEN_INTERPRETABILIDAD.md").read_text(encoding="utf-8")
    for phrase in ("validación","Eigen-CAM","Oclusión","No se ha consultado `test`"):
        if phrase not in report: raise AssertionError(f"Falta una limitación o método: {phrase}")
    for rel,digest in summary["output_hashes"].items():
        path=output/rel
        if not path.is_file() or pipeline.sha256_file(path)!=digest: raise AssertionError(f"Hash incorrecto: {rel}")
    result={"status":"passed","split":"val","test_inference_executed":False,"cases":len(selection),
            "coverage":sorted([f"{a}:{b}" for a,b in expected]),"maps":len(maps),"deletion_rows":len(curves),
            "checkpoint_sha256":summary["model"]["weights_sha256"],"checks":["val_only","checkpoint_hash","case_coverage",
              "threshold_relations","map_shapes","finite_metrics","deletion_repetitions","figures","artifact_hashes"]}
    target=output/"verification_report.json"; pipeline.write_json_atomic(target,result); return target


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--run",default=None); args=parser.parse_args()
    target=verify(Path(args.run) if args.run else None); print(target); print(target.read_text(encoding="utf-8"))


if __name__=="__main__": main()

