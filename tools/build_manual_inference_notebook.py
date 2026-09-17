"""Construye el notebook 07 para pruebas manuales de imagen y vídeo."""
from __future__ import annotations

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


ROOT = Path(__file__).resolve().parents[1]
NAME = "07_DFire_pruebas_manual_inferencia.ipynb"


def build_notebook():
    def md(value):
        return new_markdown_cell(value.strip())

    def code(value, tag=None):
        cell = new_code_cell(value.strip() + "\n")
        if tag:
            cell.metadata["tags"] = [tag]
        return cell

    cells = [
        md("""# D-Fire · Pruebas manuales de imagen y vídeo

Este cuaderno aplica un checkpoint de detección ya entrenado a una imagen o vídeo
elegido por la persona usuaria. No entrena, no modifica el dataset D-Fire y no
calcula métricas de validación o test. Cada resultado se guarda con su
configuración en `artifacts/07_manual_inference/`.
"""),
        md("""## Objetivo

Seleccionar un modelo registrado o un archivo de pesos, proporcionar una imagen o
vídeo y visualizar las cajas de humo y fuego. El notebook sirve para una
demostración cualitativa; sus detecciones no sustituyen a la evaluación formal en
validación.
"""),
        md("## 1. Preparación"),
        code('''from pathlib import Path
import datetime as dt
import hashlib
import json
import os
import sys

import cv2
import numpy as np
import pandas as pd
from PIL import Image as PILImage
from IPython.display import display, Markdown, Image, Video

roots = [Path(os.environ["TFM_PROJECT_ROOT"])] if os.environ.get("TFM_PROJECT_ROOT") else []
roots += [Path.cwd(), *Path.cwd().parents]
PROJECT_ROOT = next((p for p in roots if (p / "tfm_pipeline.py").is_file()), None)
if PROJECT_ROOT is None:
    raise FileNotFoundError("Abrir el notebook dentro de TFM o definir TFM_PROJECT_ROOT.")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO
import ultralytics

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v", ".webm", ".ogv"}
OUTPUT_PARENT = Path(os.environ.get(
    "TFM_MANUAL_OUTPUT_PARENT", PROJECT_ROOT / "artifacts" / "07_manual_inference"
))
print(f"Proyecto: {PROJECT_ROOT}")
print(f"Ultralytics: {ultralytics.__version__} · OpenCV: {cv2.__version__}")
'''),
        md("### 1.1 Modelos disponibles"),
        code('''def discover_models():
    rows = []
    experiments_dir = PROJECT_ROOT / "artifacts" / "experiments"
    for descriptor_path in sorted(experiments_dir.glob("*/experiment.json")):
        descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
        weights_rel = descriptor.get("best_model_rel")
        weights_path = PROJECT_ROOT / weights_rel if weights_rel else None
        if weights_path and weights_path.is_file():
            rows.append({
                "experiment_id": descriptor["experiment_id"],
                "modelo": descriptor.get("model_key", "—"),
                "estado": descriptor.get("status", "—"),
                "imgsz_entrenamiento": descriptor.get("train_config", {}).get("imgsz", "histórico"),
                "pesos": str(weights_path),
            })
    return pd.DataFrame(rows).sort_values(["estado", "experiment_id"]).reset_index(drop=True)

AVAILABLE_MODELS = discover_models()
display(AVAILABLE_MODELS[["experiment_id", "modelo", "estado", "imgsz_entrenamiento"]])
'''),
        md("""## 2. Elegir modelo y entrada

Hay dos formas de proporcionar el archivo:

1. Definir `INPUT_PATH` con una ruta relativa a la raíz del proyecto, por ejemplo
   `manual_inputs/mi_video.mp4`. La carpeta `manual_inputs/` es una ubicación
   cómoda para copiar archivos manualmente.
2. Usar el selector de subida de la sección siguiente. Tras elegir un archivo,
   ejecutar de nuevo las celdas de resolución y de inferencia.

Para evitar competir con un entrenamiento activo, `DEVICE` empieza en `"cpu"`.
Cambiarlo a `0` cuando la GPU esté libre. El valor de `CONF` es un filtro de
confianza para la demostración; debe elegirse aparte para un despliegue real.
"""),
        code('''# Parámetros editables
EXPERIMENT_ID = "legacy_yolov8s_baseline"  # Debe aparecer en la tabla anterior.
WEIGHTS_PATH = None  # Alternativa: "weights/mis_pesos.pt". Tiene prioridad sobre EXPERIMENT_ID.
INPUT_PATH = os.environ.get("TFM_MANUAL_INPUT") or None
# Ejemplo: "manual_inputs/mi_imagen.jpg". La variable TFM_MANUAL_INPUT permite automatizar pruebas.

CONF = 0.25
IOU = 0.70
IMGSZ = 640
DEVICE = "cpu"      # Usar 0 solo cuando no haya otro entrenamiento en la GPU.
VID_STRIDE = 1       # 1 procesa todos los fotogramas; 2 procesa uno de cada dos.
MAX_FRAMES = (
    int(os.environ["TFM_MANUAL_MAX_FRAMES"])
    if os.environ.get("TFM_MANUAL_MAX_FRAMES") else None
)                    # None procesa todo el vídeo; usar un entero para una prueba rápida.
SHOW_LABELS = True
SHOW_CONF = True
''', "parameters"),
        md("### 2.1 Subir un archivo desde Jupyter (opcional)"),
        code('''UPLOAD_WIDGET = None
try:
    import ipywidgets as widgets
    UPLOAD_WIDGET = widgets.FileUpload(
        accept=",".join(sorted(IMAGE_EXTENSIONS | VIDEO_EXTENSIONS)),
        multiple=False,
        description="Elegir archivo",
    )
    display(Markdown("Selecciona una imagen o vídeo y ejecuta después la siguiente celda."))
    display(UPLOAD_WIDGET)
except ImportError:
    display(Markdown("`ipywidgets` no está disponible. Indica el archivo mediante `INPUT_PATH`."))
'''),
        code('''def save_uploaded_file(widget):
    if widget is None or not widget.value:
        return None
    value = widget.value
    entry = next(iter(value.values())) if isinstance(value, dict) else value[0]
    name = entry["name"] if isinstance(entry, dict) else entry.name
    content = entry["content"] if isinstance(entry, dict) else entry.content
    suffix = Path(name).suffix.lower()
    if suffix not in IMAGE_EXTENSIONS | VIDEO_EXTENSIONS:
        raise ValueError(f"Formato no admitido: {suffix}")
    upload_dir = OUTPUT_PARENT / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    target = upload_dir / f"{dt.datetime.now():%Y%m%dT%H%M%S}_{Path(name).name}"
    target.write_bytes(bytes(content))
    return target

UPLOADED_PATH = save_uploaded_file(UPLOAD_WIDGET)
if UPLOADED_PATH:
    print(f"Archivo subido: {UPLOADED_PATH}")
else:
    print("No se ha subido ningún archivo; se usará INPUT_PATH si está definido.")
'''),
        md("### 2.2 Validar pesos y archivo"),
        code('''def project_path(value):
    if value is None:
        return None
    candidate = Path(value).expanduser()
    if candidate.is_file():
        return candidate.resolve()
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"No se encuentra el archivo: {value}")

def resolve_weights():
    if WEIGHTS_PATH:
        path = project_path(WEIGHTS_PATH)
        return path, "pesos_externos"
    match = AVAILABLE_MODELS[AVAILABLE_MODELS.experiment_id == EXPERIMENT_ID]
    if len(match) != 1:
        raise ValueError("EXPERIMENT_ID no está disponible. Elegir uno de la tabla o indicar WEIGHTS_PATH.")
    if match.iloc[0].estado != "complete":
        raise ValueError("El experimento seleccionado no está completo; usar un checkpoint explícito solo si se desea una prueba provisional.")
    return Path(match.iloc[0].pesos), EXPERIMENT_ID

def resolve_input():
    path = UPLOADED_PATH or project_path(INPUT_PATH)
    if path is None:
        return None, None
    suffix = path.suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        return path, "image"
    if suffix in VIDEO_EXTENSIONS:
        return path, "video"
    raise ValueError(f"Formato no admitido: {suffix}")

SELECTED_WEIGHTS, MODEL_REFERENCE = resolve_weights()
SELECTED_INPUT, INPUT_TYPE = resolve_input()
print(f"Modelo: {MODEL_REFERENCE} · pesos: {SELECTED_WEIGHTS}")
print("Entrada:", SELECTED_INPUT if SELECTED_INPUT else "pendiente")
'''),
        md("## 3. Aplicar el detector"),
        code('''def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def detection_summary(result, names):
    if result.boxes is None or len(result.boxes) == 0:
        return {"detections": 0, "by_class": {}}
    classes = result.boxes.cls.detach().cpu().numpy().astype(int)
    by_class = {}
    for class_id in classes:
        label = str(names.get(class_id, class_id)) if isinstance(names, dict) else str(names[class_id])
        by_class[label] = by_class.get(label, 0) + 1
    return {"detections": int(len(classes)), "by_class": by_class}

def run_image(model, source_path, output_dir):
    result = model.predict(
        source=str(source_path), conf=CONF, iou=IOU, imgsz=IMGSZ,
        device=DEVICE, verbose=False,
    )[0]
    annotated_bgr = result.plot(labels=SHOW_LABELS, conf=SHOW_CONF)
    target = output_dir / f"{source_path.stem}_annotated.jpg"
    if not cv2.imwrite(str(target), annotated_bgr):
        raise RuntimeError(f"No se pudo guardar {target}")
    display(Image(filename=str(target), width=1100))
    return target, {"frames_processed": 1, **detection_summary(result, model.names)}

def open_webm_writer(target, fps, frame_size):
    # OpenCV/FFmpeg imprime un aviso inocuo porque WebM no guarda la etiqueta
    # fourcc VP80 como MP4/AVI. Se silencia solo durante la apertura; el estado
    # del writer y la decodificación posterior siguen verificándose de forma
    # explícita y producen errores Python claros si algo falla.
    saved_stderr = os.dup(2)
    null_stderr = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(null_stderr, 2)
        return cv2.VideoWriter(
            str(target), cv2.VideoWriter_fourcc(*"VP80"), fps, frame_size
        )
    finally:
        os.dup2(saved_stderr, 2)
        os.close(saved_stderr)
        os.close(null_stderr)

def run_video(model, source_path, output_dir):
    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV no puede abrir el vídeo: {source_path}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(source_fps) or source_fps <= 0:
        source_fps = 25.0
    capture.release()
    # WebM/VP8 es reproducible directamente por los navegadores del notebook.
    # OpenCV/mp4v genera un MP4 válido, pero no un códec HTML5 fiable; además,
    # una ruta externa no siempre queda accesible desde el frontend de Jupyter.
    target = output_dir / f"{source_path.stem}_annotated.webm"
    writer = None
    frames_processed = 0
    detections_total = 0
    class_totals = {}
    stream = model.predict(
        source=str(source_path), stream=True, conf=CONF, iou=IOU, imgsz=IMGSZ,
        device=DEVICE, vid_stride=VID_STRIDE, verbose=False,
    )
    try:
        for result in stream:
            annotated_bgr = result.plot(labels=SHOW_LABELS, conf=SHOW_CONF)
            if writer is None:
                height, width = annotated_bgr.shape[:2]
                output_fps = max(1.0, source_fps / VID_STRIDE)
                writer = open_webm_writer(target, output_fps, (width, height))
                if not writer.isOpened():
                    raise RuntimeError(
                        "No se pudo crear el vídeo WebM/VP8 de salida. "
                        "Comprobar que OpenCV dispone del backend FFmpeg."
                    )
            writer.write(annotated_bgr)
            summary = detection_summary(result, model.names)
            detections_total += summary["detections"]
            for label, count in summary["by_class"].items():
                class_totals[label] = class_totals.get(label, 0) + count
            frames_processed += 1
            if frames_processed % 50 == 0:
                print(f"{frames_processed} fotogramas procesados")
            if MAX_FRAMES is not None and frames_processed >= MAX_FRAMES:
                break
    finally:
        if writer is not None:
            writer.release()
    if frames_processed == 0:
        raise RuntimeError("El vídeo no produjo fotogramas procesables.")
    verification_capture = cv2.VideoCapture(str(target))
    output_readable, _ = verification_capture.read()
    verification_capture.release()
    if not output_readable or target.stat().st_size == 0:
        raise RuntimeError("El vídeo se guardó, pero no se puede volver a decodificar.")
    display(Video(
        filename=str(target), embed=True, mimetype="video/webm", width=1100,
        html_attributes="controls preload='metadata'",
    ))
    return target, {
        "frames_processed": frames_processed,
        "detections": detections_total,
        "by_class": class_totals,
        "source_fps": source_fps,
        "output_codec": "VP8",
        "output_mime": "video/webm",
        "output_bytes": target.stat().st_size,
        "audio_preserved": False,
    }
'''),
        code('''if SELECTED_INPUT is None:
    display(Markdown("Define `INPUT_PATH` o sube un archivo para ejecutar la inferencia."))
    RESULT_DIR = None
else:
    run_id = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    RESULT_DIR = OUTPUT_PARENT / run_id
    RESULT_DIR.mkdir(parents=True, exist_ok=False)
    model = YOLO(str(SELECTED_WEIGHTS))
    if INPUT_TYPE == "image":
        OUTPUT_FILE, RUN_DETAILS = run_image(model, SELECTED_INPUT, RESULT_DIR)
    else:
        OUTPUT_FILE, RUN_DETAILS = run_video(model, SELECTED_INPUT, RESULT_DIR)
    metadata = {
        "run_id": run_id,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "model_reference": MODEL_REFERENCE,
        "weights_path": str(SELECTED_WEIGHTS),
        "weights_sha256": sha256_file(SELECTED_WEIGHTS),
        "input_path": str(SELECTED_INPUT),
        "input_sha256": sha256_file(SELECTED_INPUT),
        "input_type": INPUT_TYPE,
        "output_path": str(OUTPUT_FILE),
        "parameters": {
            "conf": CONF, "iou": IOU, "imgsz": IMGSZ, "device": str(DEVICE),
            "vid_stride": VID_STRIDE, "max_frames": MAX_FRAMES,
            "show_labels": SHOW_LABELS, "show_conf": SHOW_CONF,
        },
        "details": RUN_DETAILS,
        "ultralytics_version": ultralytics.__version__,
    }
    (RESULT_DIR / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    display(Markdown(
        f"**Resultado guardado:** `{OUTPUT_FILE}`  \\n"
        f"**Detecciones:** {RUN_DETAILS['detections']} · por clase: {RUN_DETAILS['by_class']}"
    ))
    print("Trazabilidad:", RESULT_DIR / "run_metadata.json")
'''),
        md("## 4. Comprobaciones y límites"),
        md("""- La salida de vídeo se codifica como WebM/VP8 compatible con HTML5 y se
  incrusta en el notebook; **no conserva audio**.
- `VID_STRIDE > 1` reduce el número de fotogramas analizados y ajusta los FPS de
  la salida para mantener una duración aproximada; úsalo solo si priorizas tiempo.
- Las detecciones dependen de `CONF`, `IOU`, `IMGSZ` y los pesos elegidos. Guarda
  esos parámetros junto al resultado mediante `run_metadata.json`.
- Un resultado atractivo en una imagen o vídeo no demuestra rendimiento general.
  Las comparaciones de modelos y umbrales siguen haciéndose con el protocolo de
  validación documentado en los notebooks 05 y 06.
"""),
        md("## 5. Siguientes pasos"),
        md("""Para probar otro caso, cambiar los parámetros de la sección 2 y ejecutar
desde la celda de validación de pesos. Cada ejecución crea una carpeta nueva, por
lo que no sobrescribe resultados anteriores.

Si se va a presentar una demostración, conviene registrar junto al vídeo el
modelo, la resolución de inferencia, el umbral y si el contenido pertenece a
D-Fire, Boreal u otra fuente externa.
"""),
    ]
    notebook = new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {"display_name": "Python 3 (TFM Docker)", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
    )
    for index, cell in enumerate(notebook.cells):
        cell.id = f"manual-inference-{index:02d}"
    return notebook


def main():
    target = ROOT / "notebooks" / NAME
    nbformat.write(build_notebook(), target)
    print(target)


if __name__ == "__main__":
    main()
