"""Utilidades compartidas para los notebooks reproducibles del TFM.

Este módulo no entrena ni evalúa por sí solo. Centraliza rutas, contratos de
artefactos y el registro de experimentos para que cada notebook pueda arrancar
desde un kernel limpio.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import platform
import re
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


CLASS_NAMES = {0: "smoke", 1: "fire"}
DEFAULT_DATASET_VERSION = "dfire_seed42_val10_v1"


def project_root() -> Path:
    """Devuelve la raíz del proyecto en Docker, Colab o Windows local."""
    configured = os.environ.get("TFM_PROJECT_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    docker_root = Path("/workspace/TFM")
    if docker_root.is_dir():
        return docker_root.resolve()
    colab_root = Path("/content/drive/MyDrive/TFM")
    if colab_root.is_dir():
        return colab_root.resolve()
    return Path(r"C:\Users\elitr\Documents\UPM Data\TFM").resolve()


def dataset_root(root: Path | None = None) -> Path:
    configured = os.environ.get("TFM_DATASET_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    return (root or project_root()) / "data" / "D-Fire"


def prepared_root(
    root: Path | None = None, version: str = DEFAULT_DATASET_VERSION
) -> Path:
    return (root or project_root()) / "artifacts" / "datasets" / version


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json_atomic(path: Path, value: Any) -> Path:
    """Escribe JSON sin dejar un fichero final parcialmente escrito."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as temporary:
        json.dump(value, temporary, ensure_ascii=False, indent=2)
        temporary.write("\n")
        temporary_path = Path(temporary.name)
    temporary_path.replace(path)
    return path


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def validate_class_mapping(names: dict[Any, Any]) -> dict[int, str]:
    normalized = {int(key): str(value).lower() for key, value in names.items()}
    if normalized != CLASS_NAMES:
        raise ValueError(
            f"Mapeo de clases incompatible: {normalized}; se esperaba {CLASS_NAMES}."
        )
    return normalized


def load_model_registry(root: Path | None = None) -> dict[str, Any]:
    path = (root or project_root()) / "configs" / "model_registry.yaml"
    if not path.exists():
        raise FileNotFoundError(f"No existe el registro de modelos: {path}")
    registry = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(registry, dict) or not registry.get("models"):
        raise ValueError("El registro debe contener una sección 'models' no vacía.")
    required = {"weights", "family", "scale"}
    for key, model in registry["models"].items():
        missing = required - set(model)
        if missing:
            raise ValueError(f"Modelo {key!r}: faltan {sorted(missing)}.")
    return registry


def resolve_model_weights(model: dict[str, Any], root: Path | None = None) -> str:
    """Prefiere un peso local; conserva el nombre remoto si debe descargarse."""
    root = root or project_root()
    candidate = str(model["weights"])
    path = Path(candidate)
    if path.is_absolute() and path.exists():
        return str(path)
    local_path = root / path
    if local_path.exists():
        return str(local_path.resolve())
    return candidate


def validate_prepared_dataset(
    root: Path | None = None,
    version: str = DEFAULT_DATASET_VERSION,
    *,
    check_all_images: bool = False,
) -> dict[str, Any]:
    """Valida el contrato que consumen entrenamiento y evaluación."""
    prepared = prepared_root(root, version)
    required = {
        "manifest": prepared / "dataset_manifest.csv",
        "metadata": prepared / "split_metadata.json",
        "data_yaml": prepared / "data.yaml",
    }
    missing = [f"{name}: {path}" for name, path in required.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "La versión preparada no está completa. Ejecuta primero el notebook 01.\n"
            + "\n".join(missing)
        )

    metadata = read_json(required["metadata"])
    manifest = pd.read_csv(required["manifest"])
    expected_columns = {
        "split",
        "filename",
        "image_path",
        "label_path",
        "box_count",
        "smoke_boxes",
        "fire_boxes",
    }
    if missing_columns := expected_columns - set(manifest.columns):
        raise ValueError(f"Faltan columnas en el manifiesto: {sorted(missing_columns)}")
    if len(manifest) != 21_527 or manifest["filename"].duplicated().any():
        raise ValueError(
            "El manifiesto preparado debe contener 21.527 nombres de imagen únicos."
        )
    observed = manifest["split"].value_counts().to_dict()
    expected = {key: int(value) for key, value in metadata["images"].items()}
    if observed != expected:
        raise ValueError(f"Particiones inconsistentes: manifiesto={observed}; metadata={expected}")

    source_root = dataset_root(root or project_root())
    rows = manifest if check_all_images else manifest.groupby("split").head(5)
    missing_images = []
    for row in rows.itertuples():
        official_split = getattr(row, "official_split", "test" if row.split == "test" else "train")
        image = source_root / official_split / "images" / row.filename
        if not image.exists():
            missing_images.append(str(image))
    if missing_images:
        raise FileNotFoundError("Faltan imágenes preparadas:\n" + "\n".join(missing_images[:20]))

    validate_class_mapping(
        yaml.safe_load(required["data_yaml"].read_text(encoding="utf-8"))["names"]
    )
    return {
        "prepared_root": prepared,
        "manifest_path": required["manifest"],
        "metadata_path": required["metadata"],
        "data_yaml_path": required["data_yaml"],
        "source_dataset_root": source_root,
        "manifest": manifest,
        "metadata": metadata,
    }


def rebased_manifest(
    contract: dict[str, Any], dataset_directory: Path | None = None
) -> pd.DataFrame:
    """Recrea rutas locales para poder mover el proyecto entre máquinas."""
    manifest = contract["manifest"].copy()
    if dataset_directory is not None:
        dataset_directory = Path(dataset_directory)
        manifest["image_path"] = [
            str(dataset_directory / "images" / split / filename)
            for split, filename in zip(manifest["split"], manifest["filename"])
        ]
        manifest["label_path"] = [
            str(dataset_directory / "labels" / split / f"{Path(filename).stem}.txt")
            for split, filename in zip(manifest["split"], manifest["filename"])
        ]
    else:
        source_root = Path(contract["source_dataset_root"])
        official_splits = (
            manifest["official_split"]
            if "official_split" in manifest
            else manifest["split"].map(lambda value: "test" if value == "test" else "train")
        )
        manifest["image_path"] = [
            str(source_root / official / "images" / filename)
            for official, filename in zip(official_splits, manifest["filename"])
        ]
        manifest["label_path"] = [
            str(source_root / official / "labels" / f"{Path(filename).stem}.txt")
            for official, filename in zip(official_splits, manifest["filename"])
        ]
    return manifest


def write_runtime_data_yaml(
    contract: dict[str, Any], destination: Path, dataset_directory: Path | None = None
) -> Path:
    """Genera un YAML válido en la máquina actual, sin rutas históricas absolutas."""
    dataset_directory = Path(dataset_directory or (Path(contract["prepared_root"]) / "dataset"))
    config = {
        "path": str(dataset_directory.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": CLASS_NAMES,
    }
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    return destination


def default_fast_data_root(root: Path | None = None) -> Path:
    configured = os.environ.get("TFM_FAST_DATA_ROOT")
    if configured:
        return Path(configured).expanduser()
    docker_cache = Path("/workspace/.cache")
    if docker_cache.is_dir():
        return docker_cache / "tfm-datasets"
    return (root or project_root()) / "artifacts" / "runtime_datasets"


def _render_prepared_label(
    source: Path,
    repairs: dict[int, dict[str, Any]],
    drops: set[int],
) -> str:
    if not source.exists() or source.stat().st_size == 0:
        return ""
    lines = source.read_text(encoding="utf-8-sig").splitlines()
    for line_number, event in repairs.items():
        lines[line_number - 1] = (
            f"{int(event['class_id'])} "
            + " ".join(
                format(float(event[key]), ".17g")
                for key in (
                    "clipped_x_center",
                    "clipped_y_center",
                    "clipped_width",
                    "clipped_height",
                )
            )
        )
    retained = [line for number, line in enumerate(lines, 1) if number not in drops]
    return "\n".join(retained) + ("\n" if retained else "")


def stage_prepared_dataset(
    contract: dict[str, Any],
    *,
    fast_base: Path | None = None,
    workers: int = 8,
) -> Path:
    """Copia una vez a almacenamiento Docker rápido y aplica las correcciones.

    El marcador solo se escribe al finalizar. Una ejecución interrumpida puede
    reanudarse: los ficheros de imagen ya copiados con el tamaño esperado se
    conservan.
    """
    prepared = Path(contract["prepared_root"])
    version = str(contract["metadata"].get("dataset_version", prepared.name))
    target_root = Path(fast_base or default_fast_data_root()) / version
    target_dataset = target_root / "dataset"
    marker = target_root / "stage_complete.json"
    manifest_hash = sha256_file(Path(contract["manifest_path"]))
    if marker.exists():
        saved = read_json(marker)
        if saved.get("manifest_sha256") == manifest_hash:
            return target_dataset
        raise RuntimeError(
            f"El staging {target_root} pertenece a otro manifiesto. Usa otra versión."
        )

    manifest = rebased_manifest(contract)
    repairs_path = prepared / "annotation_repairs.csv"
    drops_path = prepared / "annotation_dropped_boxes.csv"
    repairs_table = pd.read_csv(repairs_path) if repairs_path.exists() else pd.DataFrame()
    drops_table = pd.read_csv(drops_path) if drops_path.exists() else pd.DataFrame()
    repairs_by_key: dict[tuple[str, str], dict[int, dict[str, Any]]] = {}
    for event in repairs_table.to_dict("records"):
        key = (str(event["official_split"]), Path(event["label_path"]).name)
        repairs_by_key.setdefault(key, {})[int(event["line_number"])] = event
    drops_by_key: dict[tuple[str, str], set[int]] = {}
    for event in drops_table.to_dict("records"):
        key = (str(event["official_split"]), Path(event["label_path"]).name)
        drops_by_key.setdefault(key, set()).add(int(event["line_number"]))

    target_dataset.mkdir(parents=True, exist_ok=True)
    jpeg_missing = set(
        manifest.loc[manifest.get("jpeg_missing_eoi", False).astype(bool), "filename"]
        if "jpeg_missing_eoi" in manifest
        else []
    )

    def copy_record(record: dict[str, Any]) -> None:
        source_image = Path(record["image_path"])
        destination_image = target_dataset / "images" / record["split"] / record["filename"]
        destination_image.parent.mkdir(parents=True, exist_ok=True)
        expected_size = source_image.stat().st_size + (2 if record["filename"] in jpeg_missing else 0)
        if not destination_image.exists() or destination_image.stat().st_size != expected_size:
            shutil.copy2(source_image, destination_image)
            if record["filename"] in jpeg_missing:
                with destination_image.open("ab") as image_file:
                    image_file.write(b"\xff\xd9")

        official = str(record.get("official_split", "test" if record["split"] == "test" else "train"))
        label_name = f"{Path(record['filename']).stem}.txt"
        source_label = Path(contract["source_dataset_root"]) / official / "labels" / label_name
        destination_label = target_dataset / "labels" / record["split"] / label_name
        destination_label.parent.mkdir(parents=True, exist_ok=True)
        key = (official, label_name)
        text = _render_prepared_label(
            source_label, repairs_by_key.get(key, {}), drops_by_key.get(key, set())
        )
        destination_label.write_text(text, encoding="utf-8")

    records = manifest.to_dict("records")
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        for position, _ in enumerate(executor.map(copy_record, records), 1):
            if position % 2500 == 0 or position == len(records):
                print(f"Staging rápido: {position:,}/{len(records):,}")

    write_json_atomic(
        marker,
        {
            "dataset_version": version,
            "manifest_sha256": manifest_hash,
            "images": len(records),
            "source": str(contract["source_dataset_root"]),
            "dataset_directory": str(target_dataset.resolve()),
        },
    )
    return target_dataset


def utc_experiment_id(model_key: str, seed: int) -> str:
    safe_key = re.sub(r"[^a-zA-Z0-9_-]+", "-", model_key).strip("-")
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{safe_key}_dfire_seed{seed}_{stamp}"


def experiments_root(root: Path | None = None) -> Path:
    return (root or project_root()) / "artifacts" / "experiments"


def project_relative(path: Path, root: Path | None = None) -> str:
    """Serializa una ruta del proyecto sin fijarla a Windows, Docker o Colab."""
    return Path(path).resolve().relative_to((root or project_root()).resolve()).as_posix()


def resolve_descriptor_paths(
    descriptor: dict[str, Any], root: Path | None = None
) -> dict[str, Any]:
    root = (root or project_root()).resolve()
    result = dict(descriptor)
    for key in (
        "best_model",
        "last_model",
        "training_run_dir",
        "legacy_evaluation_summary",
        "legacy_error_summary",
    ):
        relative_key = f"{key}_rel"
        if result.get(relative_key):
            result[key] = str((root / result[relative_key]).resolve())
    return result


def experiment_descriptor(path: Path) -> dict[str, Any]:
    descriptor = resolve_descriptor_paths(read_json(Path(path) / "experiment.json"))
    descriptor["experiment_root"] = str(Path(path).resolve())
    return descriptor


def list_experiments(root: Path | None = None) -> pd.DataFrame:
    base = experiments_root(root)
    rows = []
    if base.exists():
        for descriptor_path in sorted(base.glob("*/experiment.json")):
            descriptor = resolve_descriptor_paths(read_json(descriptor_path), root)
            descriptor["experiment_root"] = str(descriptor_path.parent.resolve())
            best = descriptor.get("best_model")
            descriptor["best_model_exists"] = bool(best and Path(best).exists())
            rows.append(descriptor)
    return pd.DataFrame(rows)


def resolve_experiment(
    *,
    experiment_id: str | None = None,
    model_key: str | None = None,
    root: Path | None = None,
) -> dict[str, Any]:
    table = list_experiments(root)
    if table.empty:
        raise FileNotFoundError("No hay experimentos registrados.")
    if experiment_id:
        selected = table[table["experiment_id"] == experiment_id]
    elif model_key:
        selected = table[(table["model_key"] == model_key) & table["best_model_exists"]]
        if not selected.empty:
            selected = selected.sort_values("created_utc").tail(1)
    else:
        selected = table[table["best_model_exists"]].sort_values("created_utc").tail(1)
    if selected.empty:
        raise FileNotFoundError(
            f"No se encontró un experimento válido: id={experiment_id!r}, modelo={model_key!r}."
        )
    return selected.iloc[0].to_dict()


def environment_snapshot() -> dict[str, Any]:
    import torch
    import ultralytics

    cuda = torch.cuda.is_available()
    return {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "cuda_available": cuda,
        "cuda_runtime": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if cuda else None,
        "gpu_memory_gb": (
            round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
            if cuda
            else None
        ),
    }
