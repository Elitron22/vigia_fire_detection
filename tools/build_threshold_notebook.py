"""Construye únicamente el notebook 05 y enlaza la fase desde el notebook 03."""
from __future__ import annotations

from pathlib import Path
import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[1]
NAME = "05_DFire_barrido_umbrales.ipynb"
LINK_TEXT = """## Continuación: barrido de umbrales

La evaluación de este cuaderno proporciona el mAP estándar y un diagnóstico a
confianza fija. El siguiente paso está en
[05_DFire_barrido_umbrales.ipynb](05_DFire_barrido_umbrales.ipynb): compara los
finalistas en validación, muestra escenarios de falsas alarmas y revisa errores.
"""


def threshold_notebook():
    def md(text):
        return new_markdown_cell(text.strip())
    def code(text, tag=None):
        cell = new_code_cell(text.strip() + "\n")
        if tag:
            cell.metadata["tags"] = [tag]
        return cell
    cells = [
        md("""# D-Fire · Barrido inicial de umbrales y revisión de errores

**Fase 05 del TFM — selección exploratoria en validación.**

Este cuaderno continúa la comparación del notebook 03. Presenta resultados
guardados, candidatos bajo distintas tasas de falsas alarmas y errores que
orientan los siguientes entrenamientos. Arranca en modo consulta: no entrena,
no hace inferencia y no evalúa test al ejecutar sus celdas con los valores por defecto.
"""),
        md("## 1. Contexto y método"),
        md("""La pregunta es **qué recall de humo y fuego ofrece cada modelo para una tasa
comparable de alarmas en imágenes negativas**.

- Se utilizan YOLOv8s, YOLO26s y YOLO26n con sus `best.pt`, dataset y semilla originales.
- Una inferencia por modelo a confianza **0,01** guarda todas las cajas retenidas.
- El barrido filtra las cajas y vuelve a emparejarlas con las anotaciones: misma clase,
  confianza descendente, IoU ≥0,50 y una sola coincidencia por caja real.
- Se conserva el preprocesado del análisis operativo: 640, `rect=False`, FP32 y lotes de 4.
  YOLOv8 usa NMS IoU 0,70; las ejecuciones YOLO26 registradas usan cabeza end-to-end.
- El punto **0,25 debe reproducir los TP/FP/FN de cada imagen** de la inferencia histórica.

**Supuestos y límites.** Las etiquetas corregidas constituyen la referencia; las escenas
relacionadas entre particiones y una única semilla limitan las conclusiones. Este barrido
selecciona filtros de confianza, no calibra probabilidades ni vuelve a calcular mAP.
Los candidatos son exploratorios: el test final y la elección de despliegue quedan pendientes.
"""),
        md("### 1.1 Parámetros y entorno"),
        code('''from pathlib import Path
import os
import sys
import json
import pandas as pd
import numpy as np
from IPython.display import display, Markdown, Image

roots = [Path(os.environ["TFM_PROJECT_ROOT"])] if os.environ.get("TFM_PROJECT_ROOT") else []
roots += [Path.cwd(), *Path.cwd().parents]
PROJECT_ROOT = next((p for p in roots if (p / "tfm_pipeline.py").is_file()), None)
if PROJECT_ROOT is None:
    raise FileNotFoundError("Abrir el notebook dentro de la carpeta TFM o definir TFM_PROJECT_ROOT.")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import tfm_pipeline as pipeline
from tfm_thresholds import MODEL_LABELS

CONFIG_PATH = PROJECT_ROOT / "configs" / "threshold_sweep.yaml"
RUN_SWEEP = False
OFFLINE = True  # Al generar un barrido: exige predicciones guardadas.
RUN_ID = None   # None lee la última ejecución completa; se puede fijar su identificador.
''', "parameters"),
        md("""Para crear una nueva ejecución, cambiar `RUN_SWEEP=True`. Si faltan predicciones
para esa configuración, usar también `OFFLINE=False` dentro del contenedor con GPU.
Las inferencias se guardan separadas de la evaluación histórica a 0,25 y se reutilizan
solo cuando coinciden pesos, manifiesto, código de inferencia, parámetros y versiones.
"""),
        code('''result_parent = PROJECT_ROOT / "artifacts" / "05_threshold_sweep" / "validation"
if RUN_SWEEP:
    from tools.run_threshold_sweep import run
    RESULT_DIR = run(CONFIG_PATH, offline=OFFLINE)
elif RUN_ID:
    if Path(RUN_ID).name != RUN_ID:
        raise ValueError("RUN_ID debe ser un identificador de ejecución, no una ruta.")
    RESULT_DIR = result_parent / RUN_ID
else:
    latest = pipeline.read_json(result_parent / "latest.json")
    RESULT_DIR = PROJECT_ROOT / latest["run_rel"]
    if pipeline.sha256_file(RESULT_DIR / "run_summary.json") != latest["summary_sha256"]:
        raise ValueError("El resumen no coincide con el índice de resultados.")
summary = pipeline.read_json(RESULT_DIR / "run_summary.json")
if summary["status"] != "complete" or summary["split"] != "val":
    raise ValueError("Se necesita una ejecución completa de validación.")
for relative, expected_hash in summary["output_hashes"].items():
    if pipeline.sha256_file(RESULT_DIR / relative) != expected_hash:
        raise ValueError(f"Artefacto modificado: {relative}")
metrics = pd.read_csv(RESULT_DIR / "threshold_metrics.csv")
candidates = pd.read_csv(RESULT_DIR / "scenario_candidates.csv")
sizes = pd.read_csv(RESULT_DIR / "size_metrics.csv")
ground_truth = pd.read_csv(RESULT_DIR / "ground_truth_review.csv")
detections = pd.read_csv(RESULT_DIR / "detection_review.csv")
print(f"Ejecución: {summary['run_id']} · dataset: {summary['dataset_version']} · partición: val")
print(f"Resultados: {RESULT_DIR}")
'''),
        md("## 2. Resumen de los resultados observados"),
        code('''scenario = f"alarm_{round(summary['config']['review_budget'] * 100):02d}pct"
selected = candidates[(candidates.scenario == scenario) & candidates.feasible]
sentences = ["**Escenario diagnóstico: hasta el 2 % de negativas con alarma.**"]
for row in selected.itertuples():
    sentences.append(f"- **{MODEL_LABELS[row.model_key]}**, confianza **{row.threshold:.2f}**: "
                     f"recall humo **{row.smoke_recall:.1%}**, fuego **{row.fire_recall:.1%}**; "
                     f"{int(row.negative_images_with_alarm)}/783 negativas con alarma ({row.negative_alarm_rate:.2%}).")
sentences.append("Estos candidatos maximizan el recall medio de ambas clases dentro del límite y la malla evaluada. "
                 "El 2 % es un escenario ilustrativo, no una decisión de despliegue.")
display(Markdown("\\n\\n".join(sentences)))
'''),
        md("### 2.1 Población y comprobaciones"),
        code('''coverage = pd.DataFrame([
    {"Modelo": MODEL_LABELS[model], "Imágenes": summary["images_per_model"],
     "Negativas": summary["negative_images"], "Umbrales": len(summary["config"]["thresholds"]),
     "Referencia 0,25": check["status"]}
    for model, check in summary["baseline_verification"].items()
])
display(coverage)
print("GT: 960 cajas de humo y 1.155 de fuego. Cada imagen negativa carece de ambas clases.")
print("Referencia 0,25 reconciliada por imagen, clase y TP/FP/FN; pesos y manifiesto verificados por SHA-256.")
'''),
        md("## 3. Comparación de umbrales"),
        md("### 3.1 Recall, F1 y alarmas"),
        code('''display(Image(filename=str(RESULT_DIR / "figures" / "01_threshold_curves.png"), width=1100))'''),
        md("""Bajar el umbral permite recuperar detecciones de menor puntuación, a cambio de
admitir más cajas falsas. La referencia vertical de 0,25 permite conectar estos resultados
con el notebook 03. F1 micro suma TP, FP y FN de ambas clases antes de calcular la razón;
el recall macro da el mismo peso a humo y fuego.
"""),
        md("### 3.2 Referencia y máximo F1 de la malla"),
        code('''def present_table(table, columns):
    names = {
        "model_key": "Modelo", "scenario": "Escenario", "threshold": "Confianza",
        "smoke_recall": "Recall humo", "fire_recall": "Recall fuego",
        "smoke_precision": "P humo", "fire_precision": "P fuego",
        "smoke_f1": "F1 humo", "fire_f1": "F1 fuego",
        "micro_precision": "P micro", "micro_f1": "F1 micro",
        "negative_images_with_alarm": "Negativas con alarma", "negative_alarm_rate": "% negativas con alarma",
    }
    table = table[columns].copy()
    table["model_key"] = table["model_key"].map(MODEL_LABELS)
    if "scenario" in table:
        table["scenario"] = table["scenario"].replace({"reference_025": "Referencia 0,25", "max_f1": "Máximo F1",
                                                        "alarm_01pct": "≤1 %", "alarm_02pct": "≤2 %", "alarm_05pct": "≤5 %"})
    formats = {names[c]: "{:.1%}" for c in columns if c.endswith(("recall", "precision", "f1", "rate"))}
    formats.update({"Confianza": "{:.2f}", "Negativas con alarma": "{:.0f}"})
    return table.rename(columns=names).style.format(formats, na_rep="—").hide(axis="index")

columns = ["model_key", "scenario", "threshold", "smoke_recall", "fire_recall", "micro_precision", "micro_f1", "negative_images_with_alarm"]
display(present_table(candidates[candidates.scenario.isin(["reference_025", "max_f1"])], columns))
'''),
        md("### 3.3 Candidatos con límites de falsas alarmas"),
        md("""Se muestran límites exploratorios del **1 %, 2 % y 5 %**. Con 783 negativas,
equivalen como máximo a **7, 15 y 39 imágenes con alarma**, respectivamente.
Se utiliza el recuento exacto, sin redondear porcentajes antes de seleccionar.

Regla: máximo recall medio de humo y fuego. Desempates: mayor recall mínimo por clase,
mayor F1 micro, menos alarmas y mayor umbral. Si ningún punto satisface el límite,
el escenario se marca no factible, sin relajar el criterio.
"""),
        code('''display(present_table(candidates[candidates.scenario.str.startswith("alarm_")], columns))'''),
        code('''display(Image(filename=str(RESULT_DIR / "figures" / "02_recall_alarm_tradeoff.png"), width=1100))'''),
        md("### 3.4 Detalle por clase en el escenario del 2 %"),
        code('''display(present_table(selected, ["model_key", "threshold", "smoke_precision", "smoke_recall", "smoke_f1", "fire_precision", "fire_recall", "fire_f1"]))'''),
        md("## 4. Revisión de errores"),
        md("### 4.1 Tamaño de las cajas"),
        code('''display(Image(filename=str(RESULT_DIR / "figures" / "03_recall_by_size.png"), width=1100))'''),
        code('''small = sizes[np.isclose(sizes.threshold, .25) & (sizes.size_band == "small")]
lines = ["Recall de cajas pequeñas (<1 % del área de imagen), con confianza 0,25:"]
for model in summary["config"]["experiments"]:
    row = small[small.model_key == model].set_index("class_name")
    lines.append(f"- {MODEL_LABELS[model]}: humo {row.loc['smoke', 'recall']:.1%} "
                 f"(n={int(row.loc['smoke', 'gt_boxes'])}), fuego {row.loc['fire', 'recall']:.1%} "
                 f"(n={int(row.loc['fire', 'gt_boxes'])}).")
display(Markdown("\\n".join(lines)))
'''),
        md("### 4.2 FN recuperables y persistentes"),
        code('''display(Image(filename=str(RESULT_DIR / "figures" / "04_false_negative_diagnosis.png"), width=1100))'''),
        md("""**Recuperable** significa que una anotación perdida a 0,25 encuentra una
predicción correcta a 0,01. **Persistente** significa que sigue sin emparejarse al
mínimo evaluado; puede deberse a ausencia de caja, clase o localización. No es un
límite absoluto del modelo. La figura no recomienda usar 0,01: debe leerse junto a las alarmas.
"""),
        md("### 4.3 Tipos geométricos de falsos positivos"),
        code('''review_points = selected[["model_key", "threshold"]]
review_predictions = detections.merge(review_points, on=["model_key", "threshold"], validate="many_to_one")
fp_types = review_predictions[review_predictions.status == "fp"].groupby(["model_key", "fp_reason"]).size().unstack(fill_value=0)
fp_types = fp_types.rename(index=MODEL_LABELS, columns={"duplicate": "Duplicada", "wrong_class": "Clase incorrecta",
    "localization": "Localización", "weak_overlap": "Solapamiento débil", "no_overlap": "Sin solapamiento"})
display(fp_types)
print("Clasificación geométrica: no identifica automáticamente nubes, niebla, reflejos ni errores de anotación.")
print("Las predicciones de área cero tras recorte se conservan como FP, para mantener el protocolo histórico.")
'''),
        md("### 4.4 Galerías y observaciones visuales"),
        md("""Muestra dirigida y reproducible: alarmas negativas con mayor confianza,
FN persistentes de menor área y FN recuperables de menor área. Las galerías ayudan
a formular hipótesis; no permiten estimar la proporción de cada causa visual.
"""),
    ]
    for model in ("yolov8s", "yolo26s", "yolo26n"):
        cells.append(code(f'''gallery = RESULT_DIR / "review" / "{model}_review.png"
if gallery.exists():
    display(Image(filename=str(gallery), width=1100))'''))
    cells += [
        code('''visual_path = RESULT_DIR / "visual_review.csv"
if visual_path.exists():
    visual = pd.read_csv(visual_path).fillna("")
    display(visual[["model_key", "filename", "observation", "hypothesis"]].rename(columns={
        "model_key": "Modelo", "filename": "Imagen", "observation": "Observación", "hypothesis": "Hipótesis / siguiente comprobación"}).style
        .set_properties(**{"text-align": "left", "white-space": "normal", "max-width": "340px", "vertical-align": "top"}).hide(axis="index"))
else:
    display(Markdown("La selección automática está guardada. La inspección visual todavía no está registrada."))
'''),
        md("## 5. Interpretación y siguientes pasos"),
        md("""1. Conservar estos puntos como **referencia operativa** para los siguientes entrenamientos.
2. Separar las mejoras de confianza de las mejoras del detector: los FN persistentes,
   especialmente los pequeños, motivan un experimento controlado de resolución.
3. Revisar los ejemplos concretos antes de atribuir errores a datos o anotaciones.
4. Repetir la comparación en validación para cada nueva configuración y comprobar
   estabilidad con semillas adicionales antes de afirmar que una arquitectura es superior.
5. Congelar pesos, resolución, modo de inferencia y confianza antes de la evaluación final.

**Límites para la memoria:** métricas por caja, alarmas por imagen y escenas relacionadas;
no se han medido aún alarmas por hora ni generalización a vídeo de dron. El baseline
YOLOv8s ya tiene un test histórico; este barrido no lo utiliza para seleccionar candidatos.
"""),
        md("### 5.1 Reproducir y localizar las fuentes"),
        code('''print("Configuración ejecutada:", RESULT_DIR / "config.yaml")
print("Resumen y fuentes:", RESULT_DIR / "RESUMEN_BARRIDO.md")
print("Tablas completas:", RESULT_DIR / "threshold_metrics.csv")
print("Escenarios:", RESULT_DIR / "scenario_candidates.csv")
print("Predicciones y versiones: run_summary.json → inputs")
print("Figuras para la memoria: figures/*.svg y figures/*.png")
print("Ejecutar desde la raíz del proyecto:")
print("  python tools/run_threshold_sweep.py              # reutiliza cachés válidas o infiere con GPU")
print("  python tools/run_threshold_sweep.py --offline    # solo cachés existentes")
print("  python -m unittest discover -s tests -p test_tfm_thresholds.py")
'''),
        md("""Referencias: [Ultralytics Predict](https://docs.ultralytics.com/modes/predict/),
[Ultralytics Val](https://docs.ultralytics.com/modes/val/).
La implementación ejecutada se fija mediante versiones y copias de código en los artefactos.
"""),
    ]
    nb = new_notebook(cells=cells, metadata={"kernelspec": {"display_name": "Python 3 (TFM Docker)", "language": "python", "name": "python3"},
                                          "language_info": {"name": "python", "version": "3.12"}})
    # Identificadores estables: reconstruir no produce cambios aleatorios de celdas.
    for index, cell in enumerate(nb.cells):
        cell.id = f"threshold-{index:02d}"
    nbformat.validate(nb)
    return nb


def main():
    path = ROOT / "notebooks" / NAME
    nbformat.write(threshold_notebook(), path)
    evaluation_path = ROOT / "notebooks" / "03_DFire_evaluacion_modelos.ipynb"
    nb = nbformat.read(evaluation_path, as_version=4)
    existing = next((c for c in nb.cells if c.get("id") == "threshold-followup"), None)
    if existing is None:
        existing = new_markdown_cell(LINK_TEXT, id="threshold-followup")
        nb.cells.append(existing)
    else:
        existing.source = LINK_TEXT
    nbformat.validate(nb)
    nbformat.write(nb, evaluation_path)
    print(path)


if __name__ == "__main__":
    main()
