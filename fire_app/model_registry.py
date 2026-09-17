"""Descubrimiento de checkpoints registrados y completos."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path

from .config import ROOT


@dataclass(frozen=True)
class RegisteredModel:
    experiment_id: str
    model_key: str
    trained_imgsz: int | None
    weights: Path
    completed_utc: str
    backend: str = "pytorch"
    frozen: bool = False
    sha256: str | None = None

    def public_dict(self, *, selected: bool = False) -> dict[str, object]:
        data = asdict(self)
        data.pop("weights")
        data["selected"] = selected
        return data


class ModelRegistry:
    def __init__(self, root: Path = ROOT) -> None:
        self.root = Path(root)

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _environment_model(self) -> RegisteredModel | None:
        """Modelo desplegado fuera del árbol de experimentos (por ejemplo NCNN en la Pi)."""
        value = os.environ.get("TFM_APP_MODEL_PATH")
        if not value:
            return None
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = self.root / path
        if not path.exists():
            raise FileNotFoundError(f"TFM_APP_MODEL_PATH no existe: {path}")
        backend = "ncnn" if path.is_dir() else path.suffix.lower().lstrip(".") or "unknown"
        expected_hash = os.environ.get("TFM_APP_MODEL_SHA256")
        actual_hash = self._sha256(path) if path.is_file() else None
        if expected_hash and actual_hash != expected_hash:
            raise RuntimeError("TFM_APP_MODEL_PATH no coincide con TFM_APP_MODEL_SHA256.")
        return RegisteredModel(
            experiment_id=os.environ.get("TFM_APP_MODEL_ID", "yolo26s_final_rpi5"),
            model_key=os.environ.get("TFM_APP_MODEL_KEY", "yolo26s"),
            trained_imgsz=int(os.environ.get("TFM_APP_TRAINED_IMGSZ", "768")),
            weights=path.resolve(),
            completed_utc=os.environ.get("TFM_APP_MODEL_CREATED_UTC", ""),
            backend=backend,
            frozen=True,
            sha256=actual_hash or expected_hash,
        )

    def _frozen_model(self) -> RegisteredModel | None:
        freeze_root = self.root / "artifacts" / "14_final_model_freeze" / "final"
        manifest_path = freeze_root / "freeze_manifest.json"
        if not manifest_path.is_file():
            return None
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            weights = freeze_root / str(manifest["frozen_weights_rel"])
            if manifest.get("status") != "complete" or not weights.is_file():
                return None
            expected = str(manifest.get("weights_sha256", ""))
            actual = self._sha256(weights)
            if expected and actual != expected:
                raise RuntimeError("El hash de los pesos congelados no coincide con el manifiesto.")
            return RegisteredModel(
                experiment_id=str(manifest["experiment_id"]),
                model_key=str(manifest["model_key"]),
                trained_imgsz=int(manifest["trained_imgsz"]),
                weights=weights.resolve(),
                completed_utc=str(manifest.get("frozen_at_utc", "")),
                backend="pytorch",
                frozen=True,
                sha256=actual,
            )
        except (KeyError, TypeError, json.JSONDecodeError, OSError):
            return None

    def discover(self) -> list[RegisteredModel]:
        models: list[RegisteredModel] = []
        environment_model = self._environment_model()
        frozen_model = self._frozen_model()
        if environment_model is not None:
            models.append(environment_model)
        if frozen_model is not None and all(model.experiment_id != frozen_model.experiment_id for model in models):
            models.append(frozen_model)
        for descriptor_path in sorted((self.root / "artifacts" / "experiments").glob("*/experiment.json")):
            try:
                descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
                if descriptor.get("status") != "complete" or not descriptor.get("best_model_rel"):
                    continue
                if any(model.experiment_id == str(descriptor["experiment_id"]) for model in models):
                    continue
                weights = self.root / descriptor["best_model_rel"]
                if not weights.is_file():
                    continue
                train_config = descriptor.get("train_config", {})
                models.append(
                    RegisteredModel(
                        experiment_id=str(descriptor["experiment_id"]),
                        model_key=str(descriptor.get("model_key", "unknown")),
                        trained_imgsz=int(train_config["imgsz"]) if train_config.get("imgsz") else None,
                        weights=weights.resolve(),
                        completed_utc=str(descriptor.get("completed_utc", "")),
                        backend="pytorch",
                    )
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError, OSError):
                continue
        return sorted(
            models,
            key=lambda model: (model.frozen, model.completed_utc, model.experiment_id),
            reverse=True,
        )

    def resolve(self, experiment_id: str | None, default_experiment_id: str) -> RegisteredModel:
        models = self.discover()
        if not models:
            raise FileNotFoundError("No hay ningún experimento completo con best.pt disponible.")
        requested = experiment_id or default_experiment_id
        for model in models:
            if model.experiment_id == requested:
                return model
        if experiment_id:
            raise KeyError(f"El experimento solicitado no está disponible: {experiment_id}")
        return models[0]
