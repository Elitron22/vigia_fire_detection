"""Construye los notebooks organizados del TFM con nbformat.

El notebook histórico se usa únicamente como fuente de las funciones de
auditoría ya probadas. Los cuadernos resultantes son autocontenidos.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
NOTEBOOK_DIR = ROOT / "notebooks"
LEGACY_NOTEBOOK_DIR = ROOT / "archive" / "legacy_notebooks"


def code(source: str, *, tag: str | None = None):
    cell = new_code_cell(source.strip() + "\n")
    if tag:
        cell.metadata["tags"] = [tag]
    return cell


def markdown(source: str):
    return new_markdown_cell(source.strip() + "\n")


def notebook(cells):
    return new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python 3 (TFM Docker)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12"},
        },
    )


def indent(source: str, spaces: int = 4) -> str:
    prefix = " " * spaces
    return "\n".join(prefix + line if line.strip() else line for line in source.splitlines())


def preparation_notebook(run_preparation: bool = False):
    historical = nbformat.read(
        LEGACY_NOTEBOOK_DIR / "01_DFire_YOLOv8s_baseline.ipynb", as_version=4
    )
    audit_functions = historical.cells[8].source
    validator_tests = historical.cells[10].source
    audit_body = historical.cells[13].source

    # La entrega conserva únicamente la comprobación exacta, que sí es
    # bloqueante para impedir contaminación directa entre particiones.
    audit_body = re.sub(
        r"\nnear_duplicates = find_near_duplicates\(.*?\n\s*\)\n",
        "\n",
        audit_body,
        flags=re.S,
    )
    # Sustituye el bloque restante de candidatos cercanos si el patrón anterior
    # no abarca los to_csv/warnings del notebook histórico.
    start = audit_body.find("\nnear_duplicates = find_near_duplicates")
    if start < 0:
        start = audit_body.find("\nnear_duplicates.to_csv")
    end = audit_body.find("\nif IN_DOCKER:")
    if start >= 0 and end > start:
        audit_body = audit_body[:start] + "\n" + audit_body[end:]
    platform_start = audit_body.find("if IN_DOCKER:")
    platform_end = audit_body.find("if repair_events or dropped_events", platform_start)
    if platform_start < 0 or platform_end < 0:
        raise RuntimeError("No se encontró el bloque de vista derivada histórico.")
    audit_body = (
        audit_body[:platform_start]
        + 'dataset_view_root = PREPARED_ROOT / "dataset"\n'
        + audit_body[platform_end:]
    )
    materialize_start = audit_body.find('dataset_view_root = PREPARED_ROOT / "dataset"')
    materialize_end = audit_body.find("manifest_path = EXECUTION_ROOT", materialize_start)
    if materialize_start < 0 or materialize_end < 0:
        raise RuntimeError("No se encontró el bloque de materialización histórico.")
    audit_body = (
        audit_body[:materialize_start]
        + "dataset_view_root = None  # La vista corregida se crea en staging rápido al entrenar.\n\n"
        + audit_body[materialize_end:]
    )
    audit_body = audit_body.replace(
        "Se copiarán a la vista temporal para que Ultralytics no reescriba los originales.",
        "Se corregirán al crear el staging rápido; los originales permanecerán intactos.",
    )
    audit_body = audit_body.replace(
        '"near_duplicate_hamming_distance": NEAR_DUPLICATE_DISTANCE,\n'
        '    "near_duplicate_max_pairs_per_split_pair": MAX_NEAR_DUPLICATE_PAIRS,\n',
        '"near_duplicate_audit": "not_in_delivery_scope",\n',
    )
    audit_body = audit_body.replace(
        '"jpeg_missing_eoi_handling": "runtime_copy"',
        '"jpeg_missing_eoi_handling": "staging_copy_with_eoi_marker"',
    )

    cells = [
        markdown(
            """
# D-Fire — preparación y auditoría reproducible del dataset

Este cuaderno crea **una única versión persistente** del dataset que consumirán
todos los modelos. El D-Fire original se trata como solo lectura. La salida
versionada contiene las particiones, etiquetas corregidas, manifiesto y
trazabilidad necesarios para reproducir los experimentos.

> Ejecución normal: deja `RUN_PREPARATION=False` para inspeccionar una versión
> existente. Actívalo solo la primera vez o al crear deliberadamente otra versión.
"""
        ),
        markdown(
            """
## Objetivo y contrato de salida

- Entrada: `data/D-Fire`, con los directorios oficiales `train` y `test`.
- Validación: imágenes legibles, etiquetas YOLO, clases, cajas y duplicados exactos.
- Partición: el 10 % del train oficial se reserva como validación con semilla 42.
- Salida: `artifacts/datasets/<DATASET_VERSION>`.
- Alcance: se bloquean duplicados exactos; no se afirma independencia por escena
  porque D-Fire no proporciona identificadores fiables de vídeo, cámara o evento.
"""
        ),
        markdown("## 1. Parámetros"),
        code(
            f"""
RUN_PREPARATION = {run_preparation!r}
DATASET_VERSION = "dfire_seed42_val10_v1"
VALIDATION_FRACTION = 0.10
SEED = 42
REPAIR_OUT_OF_BOUNDS_BOXES = True
DROP_DEGENERATE_BOXES = True
ALLOW_REBUILD = False
AUDIT_WORKERS = 8
NEAR_DUPLICATE_DISTANCE = 4  # Solo se conserva para las pruebas internas históricas.
MAX_NEAR_DUPLICATE_PAIRS = 200
CLASS_NAMES = {{0: "smoke", 1: "fire"}}
CLASS_COLORS = {{0: "#F2C14E", 1: "#D62828"}}
""",
            tag="parameters",
        ),
        markdown("## 2. Entorno"),
        code(
            """
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
import random
import shutil
import tempfile
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from IPython.display import display
from PIL import Image, UnidentifiedImageError
from matplotlib.patches import Rectangle

import sys
PROJECT_ROOT = Path(os.environ.get("TFM_PROJECT_ROOT", "/workspace/TFM"))
if not PROJECT_ROOT.is_dir():
    PROJECT_ROOT = Path("C:/Users/elitr/Documents/UPM Data/TFM")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import tfm_pipeline as pipeline

PROJECT_ROOT = pipeline.project_root()
DATASET_ROOT = pipeline.dataset_root(PROJECT_ROOT)
PREPARED_ROOT = pipeline.prepared_root(PROJECT_ROOT, DATASET_VERSION)
EXECUTION_ROOT = PREPARED_ROOT
random.seed(SEED)
np.random.seed(SEED)

print(f"Dataset original: {DATASET_ROOT}")
print(f"Versión preparada: {PREPARED_ROOT}")
"""
        ),
        markdown("## 3. Funciones de auditoría"),
        code(audit_functions),
        markdown("### 3.1 Pruebas internas"),
        code(validator_tests),
        markdown("## 4. Preparación o carga de la versión persistente"),
        code(
            """
if RUN_PREPARATION:
    completed_contract = (
        (PREPARED_ROOT / "dataset_manifest.csv").exists()
        and (PREPARED_ROOT / "split_metadata.json").exists()
        and (PREPARED_ROOT / "data.yaml").exists()
    )
    incomplete_view = PREPARED_ROOT / "dataset"
    if not completed_contract and incomplete_view.exists():
        print(f"Eliminando vista parcial de una ejecución interrumpida: {incomplete_view}")
        shutil.rmtree(incomplete_view, ignore_errors=True)
        if incomplete_view.exists():
            warnings.warn(
                "Windows mantiene algunos enlaces de la vista parcial. Se ignorarán: "
                "el contrato compacto no los utiliza."
            )
    if completed_contract and not ALLOW_REBUILD:
        raise FileExistsError(
            f"La versión {DATASET_VERSION} ya existe. Déjala inmutable y usa "
            "RUN_PREPARATION=False, o elige otro DATASET_VERSION."
        )
    PREPARED_ROOT.mkdir(parents=True, exist_ok=True)
    split_layout = resolve_split_layout(DATASET_ROOT)
    display(pd.DataFrame([
        {"split": name, "images": str(paths["images"]), "labels": str(paths["labels"])}
        for name, paths in split_layout.items()
    ]))
"""
        ),
        code("if RUN_PREPARATION:\n" + indent(audit_body)),
        code(
            """
if RUN_PREPARATION:
    data_yaml_path = PREPARED_ROOT / "data.yaml"
    data_config = {
        "path": str((PREPARED_ROOT / "dataset").resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": CLASS_NAMES,
    }
    data_yaml_path.write_text(
        yaml.safe_dump(data_config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    metadata_path = PREPARED_ROOT / "split_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update({
        "dataset_version": DATASET_VERSION,
        "manifest_sha256": pipeline.sha256_file(manifest_path),
        "prepared_root": str(PREPARED_ROOT.resolve()),
        "portable_runtime_yaml_required_after_move": True,
        "storage_mode": "compact_manifest_plus_repairs",
        "requires_corrected_staging_for_training_and_new_evaluation": True,
    })
    pipeline.write_json_atomic(metadata_path, metadata)
else:
    contract = pipeline.validate_prepared_dataset(PROJECT_ROOT, DATASET_VERSION)
    PREPARED_ROOT = contract["prepared_root"]
    EXECUTION_ROOT = PREPARED_ROOT
    manifest_path = contract["manifest_path"]
    data_yaml_path = contract["data_yaml_path"]
    manifest = pipeline.rebased_manifest(contract)
    metadata = contract["metadata"]
    dataset_view_root = None
    repairs_path = PREPARED_ROOT / "annotation_repairs.csv"
    dropped_path = PREPARED_ROOT / "annotation_dropped_boxes.csv"
    repair_events = pd.read_csv(repairs_path).to_dict("records") if repairs_path.exists() else []
    dropped_events = pd.read_csv(dropped_path).to_dict("records") if dropped_path.exists() else []

contract = pipeline.validate_prepared_dataset(
    PROJECT_ROOT, DATASET_VERSION, check_all_images=False
)
manifest = pipeline.rebased_manifest(contract)
print(f"Contrato válido: {len(manifest):,} imágenes")
"""
        ),
        markdown("## 5. Perfil del dataset preparado"),
        code(
            """
split_summary = (
    manifest.groupby("split", observed=True)
    .agg(
        images=("filename", "count"),
        smoke_boxes=("smoke_boxes", "sum"),
        fire_boxes=("fire_boxes", "sum"),
        backgrounds=("box_count", lambda values: int((values == 0).sum())),
    )
    .reset_index()
)
category_summary = pd.crosstab(manifest["split"], manifest["category"])
display(split_summary)
display(category_summary)

assert split_summary.set_index("split")["images"].to_dict() == {
    "test": 4306, "train": 15500, "val": 1721
}
print(f"Cajas recortadas: {len(repair_events):,}")
print(f"Cajas degeneradas excluidas: {len(dropped_events):,}")
"""
        ),
        markdown("## 6. Comprobación visual acotada"),
        code(
            """
def show_examples(table, seed=SEED):
    selected = []
    for category in ("background", "smoke", "fire", "smoke+fire"):
        candidates = table[table["category"] == category]
        if not candidates.empty:
            selected.append(candidates.sample(1, random_state=seed))
    sample = pd.concat(selected)
    figure, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    axes = axes.ravel()
    for axis, row in zip(axes, sample.itertuples()):
        with Image.open(row.image_path) as image:
            axis.imshow(image.convert("RGB"))
        axis.set_title(f"{row.split} · {row.category}\\n{row.filename}")
        axis.axis("off")
    for axis in axes[len(sample):]:
        axis.axis("off")
    plt.show()

show_examples(manifest[manifest["split"].isin(["train", "val"])])
"""
        ),
        markdown("## 7. Resultado"),
        code(
            """
display(pd.Series({
    "dataset_version": DATASET_VERSION,
    "prepared_root": str(PREPARED_ROOT),
    "manifest": str(manifest_path),
    "data_yaml": str(data_yaml_path),
    "manifest_sha256": pipeline.sha256_file(manifest_path),
    "original_modified": False,
}, name="valor").to_frame())

print("La versión preparada ya puede consumirse desde el notebook 02.")
"""
        ),
    ]
    return notebook(cells)


def training_notebook():
    cells = [
        markdown(
            """
# D-Fire — entrenamiento reproducible de varios modelos

Este cuaderno consume una versión ya preparada de D-Fire. Puede ejecutar un solo
modelo o una cola de experimentos secuenciales. Cada trabajo conserva su propia
configuración y directorio para que las comparaciones sigan siendo trazables.
"""
        ),
        markdown(
            """
## Objetivo y reglas

- No prepara datos ni consulta el test.
- Cada lanzamiento crea un directorio de experimento independiente.
- `best.pt`, `last.pt`, entorno y configuración quedan registrados juntos.
- La cola se ejecuta en orden y libera la memoria de la GPU entre modelos.
- Un experimento fallido queda marcado como `failed`; la política de continuación
  se controla con `CONTINUE_ON_ERROR`.
- El modo seguro por defecto es `RUN_TRAINING=False`.
"""
        ),
        markdown("## 1. Parámetros"),
        code(
            """
RUN_TRAINING = False
RESUME_TRAINING = False
MODEL_KEY = "yolov8s"       # modo de un solo modelo cuando TRAINING_QUEUE está vacía
TRAINING_PROFILE = "controlled"
TRAINING_QUEUE = [
    # Ejemplos (elimina # y ajusta el perfil o los overrides que quieras comparar):
    # {"model_key": "yolov8s", "training_profile": "controlled"},
    # {"model_key": "yolo26s", "training_profile": "controlled"},
    # {"model_key": "yolo26n", "training_profile": "controlled"},
]
CONTINUE_ON_ERROR = True    # intenta el siguiente trabajo y muestra los fallos al final
DATASET_VERSION = "dfire_seed42_val10_v1"
SEED = 42
EXPERIMENT_ID = None         # modo individual; obligatorio para RESUME_TRAINING=True
RESUME_CHECKPOINT = None     # opcional; por defecto usa last.pt del experimento
ALLOW_CPU_TRAINING = False
WORKERS = 4
USE_FAST_LOCAL_DATASET = True
FAST_DATA_ROOT = None       # None usa /workspace/.cache/tfm-datasets en Docker
STAGING_WORKERS = 8
""",
            tag="parameters",
        ),
        markdown("## 2. Entorno, dataset y registro de modelos"),
        code(
            """
from pathlib import Path
import datetime as dt
import gc
import json
import os
import random
import sys

import numpy as np
import pandas as pd
import torch
from IPython.display import display
from ultralytics import YOLO

PROJECT_ROOT = Path(os.environ.get("TFM_PROJECT_ROOT", "/workspace/TFM"))
if not PROJECT_ROOT.is_dir():
    PROJECT_ROOT = Path("C:/Users/elitr/Documents/UPM Data/TFM")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import tfm_pipeline as pipeline

PROJECT_ROOT = pipeline.project_root()
contract = pipeline.validate_prepared_dataset(PROJECT_ROOT, DATASET_VERSION)
registry = pipeline.load_model_registry(PROJECT_ROOT)
DEVICE = 0 if torch.cuda.is_available() else "cpu"
environment = pipeline.environment_snapshot()

RESERVED_OVERRIDES = {"data", "device", "workers", "project", "name", "exist_ok", "resume", "seed"}

def normalise_job(raw_job, position):
    if not isinstance(raw_job, dict):
        raise TypeError(f"El trabajo {position} debe ser un diccionario.")
    model_key = raw_job.get("model_key", MODEL_KEY)
    profile = raw_job.get("training_profile", TRAINING_PROFILE)
    job_seed = int(raw_job.get("seed", SEED))
    overrides = dict(raw_job.get("overrides", {}))
    forbidden = RESERVED_OVERRIDES.intersection(overrides)
    if forbidden:
        raise ValueError(f"Overrides reservados en el trabajo {position}: {sorted(forbidden)}")
    if model_key not in registry["models"]:
        raise KeyError(f"Modelo desconocido en el trabajo {position}: {model_key!r}")
    if profile not in registry["profiles"]:
        raise KeyError(f"Perfil desconocido en el trabajo {position}: {profile!r}")

    model_config = dict(registry["models"][model_key])
    train_config = dict(registry["profiles"][profile])
    train_config.update(model_config.get("overrides", {}))
    train_config.update(overrides)
    return {
        "position": position,
        "model_key": model_key,
        "training_profile": profile,
        "seed": job_seed,
        "experiment_id": raw_job.get("experiment_id"),
        "model_config": model_config,
        "train_config": train_config,
        "weights": pipeline.resolve_model_weights(model_config, PROJECT_ROOT),
    }

queue_source = TRAINING_QUEUE or [{
    "model_key": MODEL_KEY,
    "training_profile": TRAINING_PROFILE,
    "seed": SEED,
    "experiment_id": EXPERIMENT_ID,
}]
planned_jobs = [normalise_job(job, index) for index, job in enumerate(queue_source, start=1)]

plan_rows = []
for job in planned_jobs:
    plan_rows.append({
        "orden": job["position"],
        "modelo": job["model_key"],
        "perfil": job["training_profile"],
        "imgsz": job["train_config"].get("imgsz"),
        "epochs": job["train_config"].get("epochs"),
        "batch": job["train_config"].get("batch"),
        "seed": job["seed"],
        "experiment_id": job["experiment_id"] or "automático",
    })

display(pd.DataFrame(plan_rows))
display(pd.Series({
    "dataset_version": DATASET_VERSION,
    "dataset_manifest_sha256": pipeline.sha256_file(contract["manifest_path"]),
    "device": environment["gpu_name"] or "CPU",
    "trabajos_en_cola": len(planned_jobs),
    "continuar_si_falla": CONTINUE_ON_ERROR,
}, name="valor").to_frame())
"""
        ),
        markdown("## 3. Comprobaciones previas"),
        code(
            """
if (RUN_TRAINING or RESUME_TRAINING) and not torch.cuda.is_available() and not ALLOW_CPU_TRAINING:
    raise RuntimeError(
        "Entrenamiento bloqueado: CUDA no está disponible. La preparación y la "
        "inspección sí pueden ejecutarse en CPU."
    )
if RUN_TRAINING and RESUME_TRAINING:
    raise ValueError("Activa RUN_TRAINING o RESUME_TRAINING, pero no ambos.")
if RESUME_TRAINING and not EXPERIMENT_ID:
    raise ValueError("Para reanudar debes fijar EXPERIMENT_ID.")
if RESUME_TRAINING and TRAINING_QUEUE:
    raise ValueError("La reanudación admite un experimento individual; vacía TRAINING_QUEUE.")
if not planned_jobs:
    raise ValueError("No hay trabajos de entrenamiento configurados.")

available = pipeline.list_experiments(PROJECT_ROOT)
if not available.empty:
    display(available[[
        column for column in (
            "experiment_id", "model_key", "status", "created_utc", "best_model_exists"
        ) if column in available.columns
    ]])
print("Preflight correcto. El test no se ha cargado.")
"""
        ),
        markdown("## 4. Crear o reanudar el experimento"),
        code(
            """
completed_runs = []
failed_runs = []

def set_random_seed(job_seed):
    random.seed(job_seed)
    np.random.seed(job_seed)
    torch.manual_seed(job_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(job_seed)

def next_experiment_id(job):
    requested = job["experiment_id"]
    base_id = requested or pipeline.utc_experiment_id(job["model_key"], job["seed"])
    candidate = base_id
    suffix = 2
    while (pipeline.experiments_root(PROJECT_ROOT) / candidate).exists():
        if requested:
            raise FileExistsError(
                f"El experimento ya existe: {candidate}. Cambia experiment_id o reanúdalo."
            )
        candidate = f"{base_id}_{suffix:02d}"
        suffix += 1
    return candidate

if RESUME_TRAINING:
    experiment = pipeline.resolve_experiment(
        experiment_id=EXPERIMENT_ID, root=PROJECT_ROOT
    )
    checkpoint = Path(RESUME_CHECKPOINT or experiment["last_model"])
    if not checkpoint.exists():
        raise FileNotFoundError(f"No existe el checkpoint: {checkpoint}")
    print(f"Reanudando {EXPERIMENT_ID} desde {checkpoint}")
    training_model = YOLO(str(checkpoint))
    training_result = training_model.train(resume=True)
    run_dir = Path(training_result.save_dir)
    descriptor_path = Path(experiment["experiment_root"]) / "experiment.json"
    saved = pipeline.read_json(descriptor_path)
    saved.update({
        "status": "complete",
        "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "training_run_dir_rel": pipeline.project_relative(run_dir, PROJECT_ROOT),
        "best_model_rel": pipeline.project_relative(run_dir / "weights" / "best.pt", PROJECT_ROOT),
        "last_model_rel": pipeline.project_relative(run_dir / "weights" / "last.pt", PROJECT_ROOT),
    })
    pipeline.write_json_atomic(descriptor_path, saved)
    completed_runs.append(pipeline.resolve_descriptor_paths(saved, PROJECT_ROOT))

elif RUN_TRAINING:
    staging_base = (
        Path(FAST_DATA_ROOT) if FAST_DATA_ROOT
        else pipeline.default_fast_data_root(PROJECT_ROOT)
        if USE_FAST_LOCAL_DATASET
        else PROJECT_ROOT / "artifacts" / "runtime_datasets"
    )
    staged_dataset = pipeline.stage_prepared_dataset(
        contract, fast_base=staging_base, workers=STAGING_WORKERS
    )

    for job in planned_jobs:
        experiment_id = next_experiment_id(job)
        experiment_root = pipeline.experiments_root(PROJECT_ROOT) / experiment_id
        experiment_root.mkdir(parents=True)
        runtime_yaml = pipeline.write_runtime_data_yaml(
            contract, experiment_root / "data_runtime.yaml", staged_dataset
        )
        expected_run_dir = experiment_root / "train"
        model_config = job["model_config"]
        descriptor_path = experiment_root / "experiment.json"
        experiment = {
            "schema_version": 1,
            "experiment_id": experiment_id,
            "queue_position": job["position"],
            "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "status": "running",
            "model_key": job["model_key"],
            "model_family": model_config["family"],
            "model_scale": model_config["scale"],
            "initial_weights": job["weights"],
            "dataset_version": DATASET_VERSION,
            "dataset_manifest_sha256": pipeline.sha256_file(contract["manifest_path"]),
            "profile": job["training_profile"],
            "seed": job["seed"],
            "train_config": job["train_config"],
            "training_run_dir_rel": pipeline.project_relative(expected_run_dir, PROJECT_ROOT),
            "best_model_rel": pipeline.project_relative(expected_run_dir / "weights" / "best.pt", PROJECT_ROOT),
            "last_model_rel": pipeline.project_relative(expected_run_dir / "weights" / "last.pt", PROJECT_ROOT),
        }
        pipeline.write_json_atomic(descriptor_path, experiment)
        pipeline.write_json_atomic(experiment_root / "environment.json", environment)

        training_model = None
        try:
            print(f"[{job['position']}/{len(planned_jobs)}] Iniciando {experiment_id}")
            set_random_seed(job["seed"])
            training_model = YOLO(job["weights"])
            training_result = training_model.train(
                data=str(runtime_yaml),
                device=DEVICE,
                workers=WORKERS,
                project=str(experiment_root),
                name="train",
                exist_ok=False,
                seed=job["seed"],
                **job["train_config"],
            )
            run_dir = Path(training_result.save_dir)
            saved = pipeline.read_json(descriptor_path)
            saved.update({
                "status": "complete",
                "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "training_run_dir_rel": pipeline.project_relative(run_dir, PROJECT_ROOT),
                "best_model_rel": pipeline.project_relative(run_dir / "weights" / "best.pt", PROJECT_ROOT),
                "last_model_rel": pipeline.project_relative(run_dir / "weights" / "last.pt", PROJECT_ROOT),
            })
            pipeline.write_json_atomic(descriptor_path, saved)
            completed_runs.append(pipeline.resolve_descriptor_paths(saved, PROJECT_ROOT))
            print(f"Completado: {experiment_id}")
        except BaseException as exc:
            saved = pipeline.read_json(descriptor_path)
            saved.update({
                "status": "failed",
                "failed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            })
            pipeline.write_json_atomic(descriptor_path, saved)
            failed_runs.append({
                "experiment_id": experiment_id,
                "model_key": job["model_key"],
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
    print("Modo seguro: no se ha iniciado ningún entrenamiento.")
"""
        ),
        markdown("## 5. Cerrar y registrar el entrenamiento"),
        code(
            """
if completed_runs:
    display(pd.DataFrame([{
        "experiment_id": run["experiment_id"],
        "modelo": run["model_key"],
        "estado": run["status"],
        "best.pt": run["best_model"],
    } for run in completed_runs]))
if failed_runs:
    display(pd.DataFrame(failed_runs))
if not completed_runs and not failed_runs:
    print("Sin cambios: configura RUN_TRAINING=True cuando quieras lanzar el modelo seleccionado.")
else:
    print(f"Cola finalizada: {len(completed_runs)} completos y {len(failed_runs)} fallidos.")
    print("La evaluación en validación corresponde al notebook 03; el test permanece reservado.")
"""
        ),
        markdown("## 6. Cómo añadir otro modelo"),
        markdown(
            """
Edita `configs/model_registry.yaml`, añade una clave con `family`, `scale` y
`weights`, y selecciónala en `MODEL_KEY` o añádela a `TRAINING_QUEUE`. Si los
pesos son compatibles con la API `YOLO(...)` de Ultralytics no es necesario
modificar este notebook.

Cada entrada de la cola admite `model_key`, `training_profile`, `seed`,
`experiment_id` y `overrides`. Por ejemplo, `overrides={"imgsz": 1024}` permite
una prueba puntual, aunque para una comparación formal conviene registrar un
perfil específico en `configs/model_registry.yaml`.

Para una comparación principal usa siempre el perfil `controlled`. Los ajustes
específicos de una arquitectura deben registrarse como un experimento separado,
no mezclarse silenciosamente con la comparación controlada.
"""
        ),
    ]
    return notebook(cells)


def evaluation_notebook():
    cells = [
        markdown(
            """
# D-Fire — evaluación, métricas y comparación de modelos

Este cuaderno carga un `best.pt` ya entrenado. Por defecto trabaja sobre
**validación**, que sirve para comparar modelos y tomar decisiones. El test queda
reservado para una única evaluación final. Las salidas de ambas particiones se
guardan en directorios distintos para impedir que se mezclen.
"""
        ),
        markdown("## 1. Parámetros"),
        code(
            """
RUN_STANDARD_EVALUATION = False
RUN_ERROR_ANALYSIS = False
EVALUATION_SPLIT = "val"       # val para comparar/decidir; test solo al final
MODEL_KEY = "yolov8s"
EXPERIMENT_ID = None       # None selecciona el último experimento completo del modelo
DATASET_VERSION = "dfire_seed42_val10_v1"
IMAGE_SIZE = 640
BATCH_SIZE = 16
OPERATING_CONFIDENCE = 0.25
MATCH_IOU = 0.50
NMS_IOU = 0.70
ERROR_CHUNK_SIZE = 4
ERROR_RAM_LIMIT_GIB = 6.0
SEED = 42
ALLOW_VALIDATION_RERUN = False
ALLOW_TEST_RERUN = False
USE_FAST_LOCAL_DATASET = True
FAST_DATA_ROOT = None
STAGING_WORKERS = 8
""",
            tag="parameters",
        ),
        markdown("## 2. Selección del experimento y preflight"),
        code(
            """
from pathlib import Path
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from IPython.display import Markdown, display, Image as DisplayImage
from ultralytics import YOLO

PROJECT_ROOT = Path(os.environ.get("TFM_PROJECT_ROOT", "/workspace/TFM"))
if not PROJECT_ROOT.is_dir():
    PROJECT_ROOT = Path("C:/Users/elitr/Documents/UPM Data/TFM")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import tfm_pipeline as pipeline
from tfm_evaluation import (
    model_is_end_to_end, run_error_analysis, save_error_gallery,
    split_artifact_paths, write_summary_markdown,
)

PROJECT_ROOT = pipeline.project_root()
if EVALUATION_SPLIT not in {"val", "test"}:
    raise ValueError("EVALUATION_SPLIT debe ser 'val' o 'test'.")
contract = pipeline.validate_prepared_dataset(PROJECT_ROOT, DATASET_VERSION)
manifest = pipeline.rebased_manifest(contract)
experiment = pipeline.resolve_experiment(
    experiment_id=EXPERIMENT_ID, model_key=None if EXPERIMENT_ID else MODEL_KEY,
    root=PROJECT_ROOT,
)
best_model_path = Path(experiment["best_model"])
if not best_model_path.exists():
    raise FileNotFoundError(best_model_path)
evaluation_root = Path(experiment["experiment_root"]) / "evaluation" / EVALUATION_SPLIT
summary_path = evaluation_root / "evaluation_summary.json"
metrics_path = evaluation_root / "metrics.csv"

display(pd.Series({
    "experiment_id": experiment["experiment_id"],
    "model_key": experiment["model_key"],
    "best_model": str(best_model_path),
    "dataset_version": DATASET_VERSION,
    "evaluation_split": EVALUATION_SPLIT,
    "images": int((manifest["split"] == EVALUATION_SPLIT).sum()),
    "negative_images": int(((manifest["split"] == EVALUATION_SPLIT) & (manifest["box_count"] == 0)).sum()),
}, name="valor").to_frame())
"""
        ),
        markdown("## 3. Evaluación estándar de Ultralytics"),
        code(
            """
evaluation_metrics = None
evaluation_model = None
if RUN_STANDARD_EVALUATION:
    allow_rerun = ALLOW_VALIDATION_RERUN if EVALUATION_SPLIT == "val" else ALLOW_TEST_RERUN
    if summary_path.exists() and not allow_rerun:
        raise FileExistsError(
            f"{EVALUATION_SPLIT} ya fue evaluado para este experimento: {summary_path}. "
            "Carga el resultado guardado o activa conscientemente el permiso de repetición."
        )
    evaluation_root.mkdir(parents=True, exist_ok=True)
    staging_base = (
        Path(FAST_DATA_ROOT) if FAST_DATA_ROOT
        else pipeline.default_fast_data_root(PROJECT_ROOT)
        if USE_FAST_LOCAL_DATASET
        else PROJECT_ROOT / "artifacts" / "runtime_datasets"
    )
    staged_dataset = pipeline.stage_prepared_dataset(
        contract, fast_base=staging_base, workers=STAGING_WORKERS
    )
    runtime_yaml = pipeline.write_runtime_data_yaml(
        contract, evaluation_root / "data_runtime.yaml", staged_dataset
    )
    evaluation_model = YOLO(str(best_model_path))
    pipeline.validate_class_mapping(evaluation_model.names)
    device = 0 if torch.cuda.is_available() else "cpu"
    evaluation_metrics = evaluation_model.val(
        data=str(runtime_yaml), split=EVALUATION_SPLIT, imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE, device=device, plots=True,
        project=str(evaluation_root), name="ultralytics", exist_ok=False,
        verbose=True,
    )
else:
    print(f"No se ha ejecutado inferencia sobre {EVALUATION_SPLIT}.")
"""
        ),
        code(
            """
def metric_array(metric, name):
    value = getattr(metric, name, None)
    if value is None:
        return np.array([], dtype=float)
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=float).reshape(-1)

if evaluation_metrics is not None:
    box = evaluation_metrics.box
    precision, recall = metric_array(box, "p"), metric_array(box, "r")
    ap50, ap = metric_array(box, "ap50"), metric_array(box, "ap")
    rows = [{
        "scope": "all", "precision": float(precision.mean()),
        "recall": float(recall.mean()), "mAP50": float(box.map50),
        "mAP50_95": float(box.map),
    }]
    for class_id, class_name in pipeline.CLASS_NAMES.items():
        rows.append({
            "scope": class_name, "precision": float(precision[class_id]),
            "recall": float(recall[class_id]), "mAP50": float(ap50[class_id]),
            "mAP50_95": float(ap[class_id]),
        })
    summary = {
        "experiment_id": experiment["experiment_id"],
        "model_key": experiment["model_key"],
        "model_path": str(best_model_path),
        "dataset_version": DATASET_VERSION,
        "split": EVALUATION_SPLIT,
        "end_to_end": model_is_end_to_end(evaluation_model),
        "metrics": rows,
        "speed_ms_per_image": {key: float(value) for key, value in evaluation_metrics.speed.items()},
        "evaluation_run_dir": str(Path(evaluation_metrics.save_dir)),
    }
    pipeline.write_json_atomic(summary_path, summary)
    pd.DataFrame(rows).to_csv(metrics_path, index=False)
else:
    candidate = experiment.get("legacy_evaluation_summary") if EVALUATION_SPLIT == "test" else None
    load_path = summary_path if summary_path.exists() else Path(candidate) if candidate else None
    summary = pipeline.read_json(load_path) if load_path and load_path.exists() else None

if summary:
    display(pd.DataFrame(summary["metrics"]).style.format({
        "precision": "{:.4f}", "recall": "{:.4f}",
        "mAP50": "{:.4f}", "mAP50_95": "{:.4f}",
    }))
else:
    display(Markdown(f"**Este experimento todavía no tiene evaluación estándar guardada en `{EVALUATION_SPLIT}`.**"))
"""
        ),
        markdown("## 4. Análisis operativo de FP, FN y negativos"),
        code(
            """
error_table = error_summary = error_output = None
if RUN_ERROR_ANALYSIS:
    staging_base = (
        Path(FAST_DATA_ROOT) if FAST_DATA_ROOT
        else pipeline.default_fast_data_root(PROJECT_ROOT)
        if USE_FAST_LOCAL_DATASET
        else PROJECT_ROOT / "artifacts" / "runtime_datasets"
    )
    staged_dataset = pipeline.stage_prepared_dataset(
        contract, fast_base=staging_base, workers=STAGING_WORKERS
    )
    manifest = pipeline.rebased_manifest(contract, staged_dataset)
    if evaluation_model is None:
        evaluation_model = YOLO(str(best_model_path))
        pipeline.validate_class_mapping(evaluation_model.names)
    error_output, error_table, error_summary = run_error_analysis(
        evaluation_model, manifest, evaluation_root / "error_analysis",
        model_path=best_model_path, manifest_path=contract["manifest_path"],
        conf=OPERATING_CONFIDENCE, match_iou=MATCH_IOU, nms_iou=NMS_IOU,
        imgsz=IMAGE_SIZE, chunk_size=ERROR_CHUNK_SIZE,
        device=0 if torch.cuda.is_available() else "cpu",
        seed=SEED, ram_limit_gib=ERROR_RAM_LIMIT_GIB,
        split=EVALUATION_SPLIT,
        expected_images=int((manifest["split"] == EVALUATION_SPLIT).sum()),
        expected_negatives=int(((manifest["split"] == EVALUATION_SPLIT) & (manifest["box_count"] == 0)).sum()),
    )
    write_summary_markdown(error_output, error_summary)
    error_paths = split_artifact_paths(error_output, EVALUATION_SPLIT)
    for mode, filename in (
        ("hardest", "hardest_examples.png"),
        ("negative_alarms", "negative_false_alarms.png"),
        ("random", "qualitative_predictions.png"),
    ):
        save_error_gallery(
            error_table, error_paths["predictions"],
            error_output / filename, count=12, seed=SEED, mode=mode,
        )
else:
    candidates = sorted(
        (evaluation_root / "error_analysis").glob(
            f"*/{EVALUATION_SPLIT}_error_summary.json"
        ), reverse=True,
    )
    if candidates:
        error_summary = pipeline.read_json(candidates[0])
        error_output = candidates[0].parent
    elif EVALUATION_SPLIT == "test":
        legacy_error = experiment.get("legacy_error_summary")
        if legacy_error and Path(legacy_error).exists():
            error_summary = pipeline.read_json(Path(legacy_error))

if error_summary:
    display(pd.DataFrame(error_summary["box_metrics"]))
    display(pd.DataFrame(error_summary["negative_image_alarms"]))
    print(
        "Umbral operativo:", error_summary["protocol"]["confidence"],
        "· partición:", error_summary["split"],
        "· negativas:", error_summary["negative_images"],
    )
else:
    print("No hay análisis operativo guardado para este experimento.")
"""
        ),
        markdown("## 5. Tabla comparativa de experimentos evaluados"),
        code(
            """
comparison_rows = []
experiments = pipeline.list_experiments(PROJECT_ROOT)
for row in experiments.to_dict("records"):
    root = Path(row["experiment_root"])
    possible = root / "evaluation" / EVALUATION_SPLIT / "evaluation_summary.json"
    if EVALUATION_SPLIT == "test" and not possible.exists() and row.get("legacy_evaluation_summary"):
        possible = Path(row["legacy_evaluation_summary"])
    if not possible.exists():
        continue
    saved = pipeline.read_json(possible)
    for metric in saved["metrics"]:
        comparison_rows.append({
            "experiment_id": row["experiment_id"],
            "model_key": row["model_key"],
            **metric,
        })

comparison = pd.DataFrame(comparison_rows)
if not comparison.empty:
    display(comparison.sort_values(["scope", "mAP50_95"], ascending=[True, False]))
else:
    print("Aún no hay modelos evaluados para comparar.")
"""
        ),
        markdown("## 6. Interpretación"),
        markdown(
            """
- El mAP se calcula mediante el barrido estándar de confianza de Ultralytics.
- `OPERATING_CONFIDENCE` solo controla el análisis de errores y falsas alarmas.
- Los modelos se comparan y el umbral se calibra en `val`. Después se congelan
  modelo, pesos y umbral antes de medir `test` una sola vez.
- La comparación principal debe usar la misma versión del dataset, semilla,
  resolución y perfil de entrenamiento.
"""
        ),
    ]
    return notebook(cells)


def write_all(run_preparation: bool = False):
    from tools.build_threshold_notebook import threshold_notebook, LINK_TEXT
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    evaluation = evaluation_notebook()
    evaluation.cells.append(markdown(LINK_TEXT))
    evaluation.cells[-1].id = "threshold-followup"
    outputs = {
        "01_DFire_preparacion_dataset.ipynb": preparation_notebook(run_preparation),
        "02_DFire_entrenamiento_modelos.ipynb": training_notebook(),
        "03_DFire_evaluacion_modelos.ipynb": evaluation,
        "05_DFire_barrido_umbrales.ipynb": threshold_notebook(),
    }
    for name, value in outputs.items():
        nbformat.validate(value)
        nbformat.write(value, NOTEBOOK_DIR / name)
        print(NOTEBOOK_DIR / name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-preparation",
        action="store_true",
        help="Genera temporalmente el notebook 01 con la preparación activada.",
    )
    args = parser.parse_args()
    write_all(run_preparation=args.run_preparation)


if __name__ == "__main__":
    main()
