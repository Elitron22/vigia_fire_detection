#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MODEL_DIR="${PROJECT_ROOT}/model/yolo26s_768_ncnn_model"
VENV="${PROJECT_ROOT}/.venv-rpi5-ncnn"
INSTALL_SERVICE=0
if [[ "${1:-}" == "--service" ]]; then
  INSTALL_SERVICE=1
fi

"${PYTHON:-python3}" - "${PROJECT_ROOT}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
manifest = json.loads((root / "bundle_manifest.json").read_text(encoding="utf-8"))
for relative, expected in manifest["files"].items():
    path = (root / relative).resolve()
    if root not in path.parents or not path.is_file():
        raise SystemExit(f"Artefacto ausente o fuera del bundle: {relative}")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f"SHA-256 no válido: {relative}")
print(f"Integridad verificada: {len(manifest['files'])} archivos.")
PY

if [[ "$(uname -m)" != "aarch64" ]]; then
  echo "AVISO: este instalador está preparado para Raspberry Pi OS de 64 bits (aarch64)." >&2
fi
for required in model.ncnn.param model.ncnn.bin metadata.yaml deployment_calibration.json; do
  if [[ ! -f "${MODEL_DIR}/${required}" ]]; then
    echo "Falta ${MODEL_DIR}/${required}." >&2
    exit 2
  fi
done

"${PYTHON:-python3}" - "${MODEL_DIR}/deployment_calibration.json" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if report.get("status") != "passed" or report.get("source_split") != "val":
    raise SystemExit("La calibración NCNN no está aprobada exclusivamente sobre validación.")
if report.get("test_inference_executed") is not False:
    raise SystemExit("El informe no acredita que test permaneció sin consultar.")
selected = report["selected_ncnn"]
expected = {"smoke_threshold": 0.415, "fire_threshold": 0.165}
if any(abs(float(selected[key]) - value) > 1e-12 for key, value in expected.items()):
    raise SystemExit("Los umbrales del artefacto no coinciden con la configuración desplegada.")
PY

sudo apt update
sudo apt install -y \
  python3-venv python3-pip python3-torch python3-torchvision \
  ffmpeg libgl1 libglib2.0-0
python3 -m venv --system-site-packages "${VENV}"
"${VENV}/bin/python" -m pip install --upgrade pip
"${VENV}/bin/python" -m pip install --no-cache-dir \
  -r "${PROJECT_ROOT}/deployment/rpi5_ncnn/requirements.txt"

export TFM_PROJECT_ROOT="${PROJECT_ROOT}"
export TFM_APP_CONFIG="${PROJECT_ROOT}/configs/app.rpi5.ncnn.yaml"
export TFM_APP_MODEL_PATH="${MODEL_DIR}"
export TFM_APP_MODEL_ID="yolo26s_768_ncnn_rpi5"
export TFM_APP_MODEL_KEY="yolo26s"
export TFM_APP_TRAINED_IMGSZ="768"
export TFM_APP_DEVICE="cpu"
"${VENV}/bin/python" - <<'PY'
from fire_app.config import load_settings
from fire_app.model_registry import ModelRegistry

settings = load_settings()
model = ModelRegistry().resolve(None, settings.inference.default_experiment_id)
assert model.backend == "ncnn"
print({"output": str(settings.output_directory), "model": model.public_dict(selected=True)})
PY

if [[ "${INSTALL_SERVICE}" == "1" ]]; then
  SERVICE_TMP="$(mktemp)"
  sed \
    -e "s|__TFM_USER__|${USER}|g" \
    -e "s|__TFM_ROOT__|${PROJECT_ROOT}|g" \
    "${PROJECT_ROOT}/deployment/rpi5_ncnn/tfm-fire-ncnn.service.template" > "${SERVICE_TMP}"
  sudo install -m 0644 "${SERVICE_TMP}" /etc/systemd/system/tfm-fire-ncnn.service
  rm -f "${SERVICE_TMP}"
  sudo systemctl daemon-reload
  sudo systemctl enable --now tfm-fire-ncnn.service
  sudo systemctl --no-pager --full status tfm-fire-ncnn.service || true
else
  echo "Instalación terminada. Usa --service para activar el servicio en 127.0.0.1:8002."
fi
