"""Construye el notebook 08 para el cribado dirigido de YOLO26s."""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.build_pipeline_notebooks import code, markdown, notebook

OUTPUT = ROOT / "notebooks" / "08_DFire_optimizacion_hiperparametros_YOLO26s.ipynb"


def hyperparameter_notebook():
    return notebook([
        markdown(
            """
# D-Fire — cribado dirigido de hiperparámetros para YOLO26s

**Fase 08 del TFM — entrenamiento nocturno reproducible.**

Este cuaderno aplica una estrategia multifidelidad: compara las primeras 50
épocas del entrenamiento de referencia con cuatro ensayos nuevos de 50 épocas.
Después permite entrenar la receta seleccionada durante 100 épocas. El objetivo
es explorar más configuraciones dentro de una noche sin presentar el cribado
corto como si fuera una búsqueda exhaustiva.
"""
        ),
        markdown(
            """
## 1. Diseño y límites

- Se reutiliza exactamente `dfire_seed42_val10_v1` y el mismo YOLO26s inicial.
- Resolución, optimizador, batch y semilla permanecen fijos dentro del cribado.
- Cada ensayo modifica un bloque interpretable de hiperparámetros.
- El baseline ya entrenado cuenta como referencia y no vuelve a ejecutarse.
- La comparación del cribado recorta también el baseline a sus primeras 50
  épocas, para que todas las configuraciones tengan la misma fidelidad.
- La selección preliminar usa únicamente validación; **test permanece cerrado**.
- La receta elegida se entrena después desde los pesos iniciales durante 100
  épocas; no se continúa desde el checkpoint corto.
- La decisión final requiere el barrido operativo por clase y semillas
  adicionales para la receta elegida.
- `RUN_TRAINING=False` evita iniciar por accidente una cola de varias horas.
"""
        ),
        markdown("## 2. Parámetros de ejecución"),
        code(
            '''
RUN_TRAINING = False
RUN_EVALUATION = False
EVALUATION_OFFLINE = False
RUN_FINAL_TRAINING = False
RESUME_INCOMPLETE = True
CONTINUE_ON_ERROR = True
CONFIG_PATH = "configs/yolo26s_hyperparameter_search.yaml"

# None usa exactamente active_trials del YAML. Para una prueba controlada puede
# indicarse, por ejemplo, ["hp01_lr5e4"].
TRIAL_IDS = None

# Segunda fase. Debe fijarse manualmente tras revisar el cribado y la evaluación
# operativa. Ejemplo: "hp04_lr5e4_conservative_aug".
FINAL_TRIAL_ID = None
FINAL_SEED = 42

ALLOW_CPU_TRAINING = False
WORKERS = 4
USE_FAST_LOCAL_DATASET = True
FAST_DATA_ROOT = None
STAGING_WORKERS = 8
''',
            tag="parameters",
        ),
        markdown("## 3. Cargar configuración y comprobar el plan"),
        code(
            '''
from pathlib import Path
import datetime as dt
import gc
import os
import random
import sys

import numpy as np
import pandas as pd
import torch
import yaml
from IPython.display import display, Markdown
from ultralytics import YOLO

PROJECT_ROOT = Path(os.environ.get("TFM_PROJECT_ROOT", "/workspace/TFM"))
if not PROJECT_ROOT.is_dir():
    PROJECT_ROOT = Path("C:/Users/elitr/Documents/UPM Data/TFM")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import tfm_pipeline as pipeline

PROJECT_ROOT = pipeline.project_root()
config_path = PROJECT_ROOT / CONFIG_PATH
config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
if config.get("schema_version") != 2:
    raise ValueError("Solo se admite schema_version=2.")
if config.get("model_key") != "yolo26s":
    raise ValueError("Este cuaderno está limitado explícitamente a YOLO26s.")
if not config.get("selection", {}).get("test_locked"):
    raise ValueError("La configuración debe mantener test_locked=true.")

trial_ids = list(config["active_trials"] if TRIAL_IDS is None else TRIAL_IDS)
unknown = set(trial_ids) - set(config["trials"])
if unknown:
    raise KeyError(f"Ensayos desconocidos: {sorted(unknown)}")
if len(trial_ids) != len(set(trial_ids)):
    raise ValueError("TRIAL_IDS contiene duplicados.")

estimated_hours = len(trial_ids) * float(config["expected_hours_per_screening_run"])
if estimated_hours > float(config["night_budget_hours"]):
    raise ValueError(
        f"La cola estimada ({estimated_hours:.1f} h) supera el presupuesto nocturno "
        f"({config['night_budget_hours']:.1f} h)."
    )

screening_parameters = dict(config["screening_parameters"])
if screening_parameters.get("epochs") != config["screening_epochs"]:
    raise ValueError("screening_parameters.epochs debe coincidir con screening_epochs.")
if config["selection"].get("screening_horizon_epochs") != config["screening_epochs"]:
    raise ValueError("El horizonte de selección debe coincidir con el cribado.")

if RUN_FINAL_TRAINING and FINAL_TRIAL_ID not in config["trials"]:
    raise ValueError("Para la fase final debes fijar FINAL_TRIAL_ID a un ensayo conocido.")

rows = []
for trial_id in trial_ids:
    spec = config["trials"][trial_id]
    effective = screening_parameters | dict(spec.get("overrides", {}))
    rows.append({
        "ensayo": trial_id,
        "experiment_id": spec["experiment_id"],
        "seed": spec["seed"],
        "epochs": effective["epochs"],
        "lr0": effective.get("lr0"),
        "weight_decay": effective.get("weight_decay", "Ultralytics"),
        "hsv_s": effective.get("hsv_s", "Ultralytics"),
        "scale": effective.get("scale", "Ultralytics"),
        "mosaic": effective.get("mosaic", "Ultralytics"),
        "estimación_h": config["expected_hours_per_screening_run"],
        "justificación": spec["rationale"],
    })

display(pd.DataFrame(rows))
display(pd.Series({
    "search_id": config["search_id"],
    "baseline": config["baseline_experiment_id"],
    "ensayos_nuevos": len(trial_ids),
    "configuraciones_de_cribado_con_baseline": 1 + len(trial_ids),
    "duración_estimativa_h": estimated_hours,
    "presupuesto_nocturno_h": config["night_budget_hours"],
}, name="valor").to_frame())
'''
        ),
        markdown("## 4. Preflight reproducible"),
        code(
            '''
contract = pipeline.validate_prepared_dataset(PROJECT_ROOT, config["dataset_version"])
registry = pipeline.load_model_registry(PROJECT_ROOT)
model_config = dict(registry["models"][config["model_key"]])
initial_weights = pipeline.resolve_model_weights(model_config, PROJECT_ROOT)
baseline = pipeline.resolve_experiment(
    experiment_id=config["baseline_experiment_id"], root=PROJECT_ROOT
)
if baseline["status"] != "complete" or baseline["model_key"] != "yolo26s":
    raise ValueError("El baseline declarado no es un YOLO26s completo.")
if baseline["dataset_version"] != config["dataset_version"]:
    raise ValueError("El baseline utiliza otra versión del dataset.")
if baseline.get("dataset_manifest_sha256") != pipeline.sha256_file(contract["manifest_path"]):
    raise ValueError("El manifiesto del baseline no coincide con el dataset preparado.")

DEVICE = 0 if torch.cuda.is_available() else "cpu"
if (RUN_TRAINING or RUN_FINAL_TRAINING) and DEVICE == "cpu" and not ALLOW_CPU_TRAINING:
    raise RuntimeError("Entrenamiento bloqueado: CUDA no está disponible.")

environment = pipeline.environment_snapshot()
display(pd.Series({
    "pesos_iniciales": initial_weights,
    "dataset": config["dataset_version"],
    "manifest_sha256": pipeline.sha256_file(contract["manifest_path"]),
    "dispositivo": environment["gpu_name"] or "CPU",
    "test_cargado": False,
}, name="valor").to_frame())
print("Preflight correcto. El conjunto de test no se ha cargado.")
'''
        ),
        markdown("## 5. Ejecutar o reanudar los cuatro cribados nocturnos"),
        code(
            '''
completed_runs = []
failed_runs = []
skipped_runs = []

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def update_descriptor(descriptor_path, **updates):
    descriptor = pipeline.read_json(descriptor_path)
    descriptor.update(updates)
    pipeline.write_json_atomic(descriptor_path, descriptor)
    return pipeline.resolve_descriptor_paths(descriptor, PROJECT_ROOT)

if RUN_TRAINING:
    staging_base = (
        Path(FAST_DATA_ROOT) if FAST_DATA_ROOT
        else pipeline.default_fast_data_root(PROJECT_ROOT)
        if USE_FAST_LOCAL_DATASET
        else PROJECT_ROOT / "artifacts" / "runtime_datasets"
    )
    staged_dataset = pipeline.stage_prepared_dataset(
        contract, fast_base=staging_base, workers=STAGING_WORKERS
    )

    for position, trial_id in enumerate(trial_ids, start=1):
        spec = config["trials"][trial_id]
        experiment_id = spec["experiment_id"]
        experiment_root = pipeline.experiments_root(PROJECT_ROOT) / experiment_id
        descriptor_path = experiment_root / "experiment.json"
        training_model = None
        try:
            if descriptor_path.exists():
                existing = pipeline.resolve_descriptor_paths(
                    pipeline.read_json(descriptor_path), PROJECT_ROOT
                )
                if existing["status"] == "complete":
                    print(f"[{position}/{len(trial_ids)}] Ya completo: {experiment_id}")
                    completed_runs.append(existing)
                    skipped_runs.append(experiment_id)
                    continue
                checkpoint = Path(existing["last_model"])
                if not RESUME_INCOMPLETE or not checkpoint.exists():
                    raise FileExistsError(
                        f"Existe un experimento incompleto sin reanudación utilizable: {experiment_id}"
                    )
                print(f"[{position}/{len(trial_ids)}] Reanudando {experiment_id}")
                training_model = YOLO(str(checkpoint))
                training_result = training_model.train(resume=True)
            else:
                experiment_root.mkdir(parents=True)
                runtime_yaml = pipeline.write_runtime_data_yaml(
                    contract, experiment_root / "data_runtime.yaml", staged_dataset
                )
                train_config = screening_parameters | dict(spec.get("overrides", {}))
                expected_run_dir = experiment_root / "train"
                descriptor = {
                    "schema_version": 1,
                    "experiment_id": experiment_id,
                    "search_id": config["search_id"],
                    "trial_id": trial_id,
                    "queue_position": position,
                    "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                    "status": "running",
                    "model_key": config["model_key"],
                    "model_family": model_config["family"],
                    "model_scale": model_config["scale"],
                    "initial_weights": initial_weights,
                    "dataset_version": config["dataset_version"],
                    "dataset_manifest_sha256": pipeline.sha256_file(contract["manifest_path"]),
                    "profile": f"{config['search_id']}_screening_50ep",
                    "fidelity": "screening",
                    "seed": int(spec["seed"]),
                    "train_config": train_config,
                    "hyperparameter_rationale": spec["rationale"],
                    "training_run_dir_rel": pipeline.project_relative(expected_run_dir, PROJECT_ROOT),
                    "best_model_rel": pipeline.project_relative(expected_run_dir / "weights" / "best.pt", PROJECT_ROOT),
                    "last_model_rel": pipeline.project_relative(expected_run_dir / "weights" / "last.pt", PROJECT_ROOT),
                }
                pipeline.write_json_atomic(descriptor_path, descriptor)
                pipeline.write_json_atomic(experiment_root / "environment.json", environment)
                print(f"[{position}/{len(trial_ids)}] Iniciando {experiment_id}")
                set_seed(int(spec["seed"]))
                training_model = YOLO(initial_weights)
                training_result = training_model.train(
                    data=str(runtime_yaml),
                    device=DEVICE,
                    workers=WORKERS,
                    project=str(experiment_root),
                    name="train",
                    exist_ok=False,
                    seed=int(spec["seed"]),
                    **train_config,
                )

            run_dir = Path(training_result.save_dir)
            completed = update_descriptor(
                descriptor_path,
                status="complete",
                completed_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                training_run_dir_rel=pipeline.project_relative(run_dir, PROJECT_ROOT),
                best_model_rel=pipeline.project_relative(run_dir / "weights" / "best.pt", PROJECT_ROOT),
                last_model_rel=pipeline.project_relative(run_dir / "weights" / "last.pt", PROJECT_ROOT),
            )
            completed_runs.append(completed)
            print(f"Completado: {experiment_id}")
        except BaseException as exc:
            if descriptor_path.exists():
                update_descriptor(
                    descriptor_path,
                    status="failed",
                    failed_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
            failed_runs.append({
                "experiment_id": experiment_id,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            })
            print(f"Falló {experiment_id}: {type(exc).__name__}: {exc}")
            if not CONTINUE_ON_ERROR or isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
        finally:
            del training_model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
else:
    print("Modo seguro: RUN_TRAINING=False; no se ha iniciado el cribado.")
'''
        ),
        markdown("## 6. Resumir resultados disponibles"),
        code(
            '''
def training_summary(experiment, horizon_epochs):
    results_path = Path(experiment["training_run_dir"]) / "results.csv"
    if not results_path.exists():
        return None
    results = pd.read_csv(results_path)
    results.columns = [column.strip() for column in results.columns]
    results = results.iloc[:horizon_epochs].copy()
    metric = config["selection"]["primary_metric"]
    if metric not in results:
        raise KeyError(f"Falta la métrica {metric!r} en {results_path}.")
    best_index = results[metric].astype(float).idxmax()
    best = results.loc[best_index]
    return {
        "experiment_id": experiment["experiment_id"],
        "trial_id": experiment.get("trial_id", "baseline"),
        "estado": experiment["status"],
        "horizonte_comparado": len(results),
        "mejor_época": int(best["epoch"]),
        "mAP50-95": float(best["metrics/mAP50-95(B)"]),
        "mAP50": float(best["metrics/mAP50(B)"]),
        "precisión": float(best["metrics/precision(B)"]),
        "recall": float(best["metrics/recall(B)"]),
        "épocas_totales_disponibles": len(pd.read_csv(results_path)),
    }

available_experiments = [baseline]
for trial_id in trial_ids:
    descriptor = pipeline.experiments_root(PROJECT_ROOT) / config["trials"][trial_id]["experiment_id"] / "experiment.json"
    if descriptor.exists():
        candidate = pipeline.resolve_descriptor_paths(pipeline.read_json(descriptor), PROJECT_ROOT)
        if candidate.get("status") == "complete":
            available_experiments.append(candidate)

screening_horizon = int(config["selection"]["screening_horizon_epochs"])
summary_rows = [training_summary(exp, screening_horizon) for exp in available_experiments]
summary_rows = [row for row in summary_rows if row is not None]
summary_table = pd.DataFrame(summary_rows).sort_values("mAP50-95", ascending=False)
display(summary_table.style.format({
    "mAP50-95": "{:.4f}", "mAP50": "{:.4f}",
    "precisión": "{:.4f}", "recall": "{:.4f}",
}))

if failed_runs:
    display(Markdown("### Ejecuciones fallidas"))
    display(pd.DataFrame(failed_runs))
if skipped_runs:
    print("Ya estaban completas y no se repitieron:", ", ".join(skipped_runs))

if len(summary_table) > 1:
    import matplotlib.pyplot as plt
    plot_table = summary_table.sort_values("mAP50-95")
    colors = ["#275D8C" if value == "baseline" else "#C58A1B" for value in plot_table["trial_id"]]
    ax = plot_table.plot.barh(
        x="trial_id", y="mAP50-95", color=colors, legend=False, figsize=(9, 4.5)
    )
    ax.set(
        title=f"Cribado YOLO26s al mismo horizonte ({screening_horizon} épocas)",
        xlabel="mAP50-95 de validación", ylabel="Configuración",
    )
    ax.grid(axis="x", alpha=.2)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.4f", padding=3)
    plt.tight_layout()
    plt.show()
'''
        ),
        markdown("## 7. Evaluación automática del cribado"),
        code(
            '''
if RUN_EVALUATION:
    import subprocess
    command = [sys.executable, str(PROJECT_ROOT / "tools" / "run_hyperparameter_evaluation.py")]
    if EVALUATION_OFFLINE:
        command.append("--offline")
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)
else:
    print("Evaluación desactivada. Puede revisarse el estado con:")
    print("python tools/run_hyperparameter_evaluation.py --preflight")
'''
        ),
        markdown("## 8. Entrenamiento completo de la receta seleccionada"),
        code(
            '''
final_run = None
if RUN_FINAL_TRAINING:
    spec = config["trials"][FINAL_TRIAL_ID]
    final_experiment_id = f"yolo26s_hp_final_{FINAL_TRIAL_ID}_100ep_seed{FINAL_SEED}"
    final_root = pipeline.experiments_root(PROJECT_ROOT) / final_experiment_id
    final_descriptor_path = final_root / "experiment.json"

    if final_descriptor_path.exists():
        existing = pipeline.resolve_descriptor_paths(
            pipeline.read_json(final_descriptor_path), PROJECT_ROOT
        )
        if existing["status"] == "complete":
            final_run = existing
            print(f"El entrenamiento final ya estaba completo: {final_experiment_id}")
        else:
            checkpoint = Path(existing["last_model"])
            if not RESUME_INCOMPLETE or not checkpoint.exists():
                raise FileExistsError("Existe una fase final incompleta sin checkpoint reanudable.")
            print(f"Reanudando entrenamiento final: {final_experiment_id}")
            final_model = YOLO(str(checkpoint))
            final_result = final_model.train(resume=True)
    else:
        staging_base = (
            Path(FAST_DATA_ROOT) if FAST_DATA_ROOT
            else pipeline.default_fast_data_root(PROJECT_ROOT)
            if USE_FAST_LOCAL_DATASET
            else PROJECT_ROOT / "artifacts" / "runtime_datasets"
        )
        staged_dataset = pipeline.stage_prepared_dataset(
            contract, fast_base=staging_base, workers=STAGING_WORKERS
        )
        final_root.mkdir(parents=True)
        runtime_yaml = pipeline.write_runtime_data_yaml(
            contract, final_root / "data_runtime.yaml", staged_dataset
        )
        final_config = dict(config["final_parameters"]) | dict(spec.get("overrides", {}))
        expected_run_dir = final_root / "train"
        final_descriptor = {
            "schema_version": 1,
            "experiment_id": final_experiment_id,
            "search_id": config["search_id"],
            "source_trial_id": FINAL_TRIAL_ID,
            "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "status": "running",
            "model_key": config["model_key"],
            "model_family": model_config["family"],
            "model_scale": model_config["scale"],
            "initial_weights": initial_weights,
            "dataset_version": config["dataset_version"],
            "dataset_manifest_sha256": pipeline.sha256_file(contract["manifest_path"]),
            "profile": f"{config['search_id']}_final_100ep",
            "fidelity": "full",
            "seed": int(FINAL_SEED),
            "train_config": final_config,
            "hyperparameter_rationale": spec["rationale"],
            "training_run_dir_rel": pipeline.project_relative(expected_run_dir, PROJECT_ROOT),
            "best_model_rel": pipeline.project_relative(expected_run_dir / "weights" / "best.pt", PROJECT_ROOT),
            "last_model_rel": pipeline.project_relative(expected_run_dir / "weights" / "last.pt", PROJECT_ROOT),
        }
        pipeline.write_json_atomic(final_descriptor_path, final_descriptor)
        pipeline.write_json_atomic(final_root / "environment.json", environment)
        print(f"Iniciando entrenamiento final desde pesos iniciales: {final_experiment_id}")
        set_seed(int(FINAL_SEED))
        final_model = YOLO(initial_weights)
        final_result = final_model.train(
            data=str(runtime_yaml), device=DEVICE, workers=WORKERS,
            project=str(final_root), name="train", exist_ok=False,
            seed=int(FINAL_SEED), **final_config,
        )

    if final_run is None:
        final_run_dir = Path(final_result.save_dir)
        final_run = update_descriptor(
            final_descriptor_path,
            status="complete",
            completed_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
            training_run_dir_rel=pipeline.project_relative(final_run_dir, PROJECT_ROOT),
            best_model_rel=pipeline.project_relative(final_run_dir / "weights" / "best.pt", PROJECT_ROOT),
            last_model_rel=pipeline.project_relative(final_run_dir / "weights" / "last.pt", PROJECT_ROOT),
        )
        del final_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    display(pd.Series({
        "experimento_final": final_run["experiment_id"],
        "receta_origen": FINAL_TRIAL_ID,
        "épocas_máximas": final_run["train_config"]["epochs"],
        "best.pt": final_run["best_model"],
    }, name="valor").to_frame())
else:
    print("Fase final desactivada. Evalúa primero los cribados y fija FINAL_TRIAL_ID.")
'''
        ),
        markdown("## 9. Interpretación y siguiente decisión"),
        markdown(
            """
Las métricas de entrenamiento sirven para el cribado, pero no determinan por sí
solas el modelo final. Cuando terminen los cuatro ensayos:

1. Ejecutar sobre validación el mismo barrido de confianza usado en el proyecto.
2. Ajustar por separado los umbrales de humo y fuego, manteniendo el límite del
   2 % de imágenes negativas con alarma.
3. Comparar recall de humo/fuego, precisión, F1, objetos pequeños y falsas cajas.
4. Fijar `FINAL_TRIAL_ID` y activar `RUN_FINAL_TRAINING=True` para entrenar la
   receta ganadora desde cero durante 100 épocas.
5. Repetir la receta ganadora con al menos dos semillas nuevas antes de formular
   una conclusión fuerte sobre el hiperparámetro.
6. Mantener test cerrado hasta fijar modelo, resolución, umbrales y regla temporal.

En la memoria debe describirse como **búsqueda manual dirigida con presupuesto
computacional limitado**, indicando exactamente los ensayos realmente ejecutados.
"""
        ),
    ])


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    value = hyperparameter_notebook()
    nbformat.validate(value)
    nbformat.write(value, OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
