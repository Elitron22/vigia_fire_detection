#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -d "${SCRIPT_DIR}/fire_app" ]]; then
  PROJECT_ROOT="${SCRIPT_DIR}"
else
  PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi
INSTALL_SERVICE=0
if [[ "${1:-}" == "--service" ]]; then
  INSTALL_SERVICE=1
fi

if [[ "$(uname -m)" != "aarch64" ]]; then
  echo "AVISO: este instalador está preparado para Raspberry Pi OS de 64 bits (aarch64)." >&2
fi
if [[ ! -f "${PROJECT_ROOT}/model/yolo26s_768_final.pt" ]]; then
  echo "Falta model/yolo26s_768_final.pt. Copiar primero el modelo final (ver deployment/README.md)." >&2
  exit 2
fi

sudo apt update
sudo apt install -y \
  python3-venv python3-pip python3-torch python3-torchvision \
  ffmpeg libgl1 libglib2.0-0
python3 -m venv --system-site-packages "${PROJECT_ROOT}/.venv-rpi5"
"${PROJECT_ROOT}/.venv-rpi5/bin/python" -m pip install --upgrade pip
# En aarch64, el wheel generico reciente de PyTorch puede arrastrar varios GB
# de bibliotecas CUDA para Jetson. Raspberry Pi OS proporciona builds CPU
# nativas; el venv con system-site-packages las reutiliza.
"${PROJECT_ROOT}/.venv-rpi5/bin/python" -m pip install --no-cache-dir \
  -r "${PROJECT_ROOT}/deployment/rpi5/requirements.txt"

export TFM_PROJECT_ROOT="${PROJECT_ROOT}"
export TFM_APP_CONFIG="${PROJECT_ROOT}/configs/app.rpi5.yaml"
export TFM_APP_MODEL_PATH="${PROJECT_ROOT}/model/yolo26s_768_final.pt"
export TFM_APP_MODEL_ID="yolo26s_final_rpi5"
export TFM_APP_DEVICE="cpu"
export TFM_APP_MODEL_SHA256="bfb54c4726519fb473bb7c4825577e51c05af2b0db5fd46bf1931a8a29285f67"
"${PROJECT_ROOT}/.venv-rpi5/bin/python" - <<'PY'
from fire_app.config import load_settings
from fire_app.model_registry import ModelRegistry

settings = load_settings()
model = ModelRegistry().resolve(None, settings.inference.default_experiment_id)
print({"config": str(settings.output_directory), "model": model.public_dict(selected=True)})
PY

if [[ "${INSTALL_SERVICE}" == "1" ]]; then
  SERVICE_TMP="$(mktemp)"
  sed \
    -e "s|__TFM_USER__|${USER}|g" \
    -e "s|__TFM_ROOT__|${PROJECT_ROOT}|g" \
    "${PROJECT_ROOT}/deployment/rpi5/tfm-fire.service.template" > "${SERVICE_TMP}"
  sudo install -m 0644 "${SERVICE_TMP}" /etc/systemd/system/tfm-fire.service
  rm -f "${SERVICE_TMP}"
  sudo systemctl daemon-reload
  sudo systemctl enable --now tfm-fire.service
  sudo systemctl --no-pager --full status tfm-fire.service || true
else
  echo
  echo "Instalación terminada. Arranque manual:"
  echo "TFM_PROJECT_ROOT='${PROJECT_ROOT}' TFM_APP_CONFIG='${PROJECT_ROOT}/configs/app.rpi5.yaml' TFM_APP_MODEL_PATH='${PROJECT_ROOT}/model/yolo26s_768_final.pt' '${PROJECT_ROOT}/.venv-rpi5/bin/python' '${PROJECT_ROOT}/tools/run_detection_app.py' --host 127.0.0.1 --port 8000"
fi
