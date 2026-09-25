"""Construye el notebook 10 con la comparación completa al 1 %."""
from __future__ import annotations

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


ROOT = Path(__file__).resolve().parents[1]
NAME = "10_DFire_comparacion_todos_modelos_1pct.ipynb"


def build_notebook():
    def md(value: str):
        return new_markdown_cell(value.strip())

    def code(value: str):
        return new_code_cell(value.strip() + "\n")

    cells = [
        md("""# D-Fire · Comparación completa con presupuesto del 1 %

Este cuaderno amplía la selección final a todos los checkpoints completos y a
las combinaciones de resolución ya evaluadas. Los cuatro ensayos de
hiperparámetros de 50 épocas se muestran por separado. Todo el análisis usa
**validación**; el conjunto de test permanece bloqueado.
"""),
        md("## tl;dr"),
        code('''from pathlib import Path
import json
import os
import subprocess
import sys

import pandas as pd
import yaml
from IPython.display import display, Markdown, Image

roots = [Path(os.environ["TFM_PROJECT_ROOT"])] if os.environ.get("TFM_PROJECT_ROOT") else []
roots += [Path.cwd(), *Path.cwd().parents]
PROJECT_ROOT = next((path for path in roots if (path / "tfm_pipeline.py").is_file()), None)
if PROJECT_ROOT is None:
    raise FileNotFoundError("Abrir el notebook dentro de TFM o definir TFM_PROJECT_ROOT.")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import tfm_pipeline as pipeline

CONFIG_PATH = PROJECT_ROOT / "configs" / "all_models_01pct_comparison.yaml"
config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
assert config["split"] == "val" and config["test_locked"] is True

latest = pipeline.read_json(
    PROJECT_ROOT / "artifacts" / "13_all_models_01pct_comparison" / "validation" / "latest.json"
)
RUN_DIR = PROJECT_ROOT / latest["run_rel"]
summary = pipeline.read_json(RUN_DIR / "run_summary.json")
assert summary["test_inference_executed"] is False

ranked = pd.read_csv(RUN_DIR / "ranked_operating_points.csv")
full_sensitivity = ranked[
    (ranked.comparison_group == "full_training") & (ranked.scenario == "sensitivity")
].sort_values("rank")
winner = full_sensitivity.iloc[0]
previous = full_sensitivity[full_sensitivity.configuration == "yolo26s_768_eval768"].iloc[0]
display(Markdown(
    f"**Mejor recall medio con el criterio del 1 %:** {winner.label}, con recall macro "
    f"**{winner.macro_recall:.2%}** y F1 **{winner.micro_f1:.2%}**.  \\n"
    f"**Modelo final (elegido en el notebook 09):** {previous.label}, recall macro **{previous.macro_recall:.2%}**, "
    f"F1 **{previous.micro_f1:.2%}** y recall de fuego **{previous.fire_recall:.2%}**."
))
'''),
        md("""## Contexto y método

### Supuestos clave

- “1 %” significa como máximo 7 de las 783 imágenes negativas con alguna alarma.
- Se barren umbrales independientes para humo y fuego de 0,16 a 0,60, en pasos de 0,01.
- El ranking de sensibilidad maximiza primero el recall macro y usa el recall de
  la peor clase, F1 y precisión únicamente como desempates.
- El ranking equilibrado maximiza F1. No sustituye al criterio principal.
- Las variantes de 50 épocas no son equivalentes a los entrenamientos completos.
"""),
        md("### 1. Verificación y cobertura"),
        code('''verification = subprocess.run(
    [sys.executable, str(PROJECT_ROOT / "tools" / "verify_all_models_01pct_comparison.py")],
    check=True, cwd=PROJECT_ROOT, capture_output=True, text=True,
)
verification_report = json.loads(verification.stdout)
display(pd.DataFrame([verification_report]))

coverage = pd.DataFrame([{
    "imágenes de validación": summary["images"],
    "imágenes negativas": summary["negative_images"],
    "configuraciones completas": summary["full_training_configurations"],
    "ensayos HP de 50 épocas": summary["hp_screening_configurations"],
    "parejas de umbrales": summary["class_threshold_points"],
    "máximo de alarmas": summary["maximum_negative_images_with_alarm"],
    "test consultado": summary["test_inference_executed"],
}])
display(coverage)
'''),
        md("## Resultados"),
        md("### 2. Ranking de máxima sensibilidad"),
        code('''columns = [
    "rank", "label", "smoke_threshold", "fire_threshold", "smoke_recall", "fire_recall",
    "macro_recall", "minimum_class_recall", "micro_precision", "micro_recall", "micro_f1",
    "negative_images_with_alarm",
]
table = full_sensitivity[columns].rename(columns={
    "rank": "Puesto", "label": "Configuración", "smoke_threshold": "Umbral humo",
    "fire_threshold": "Umbral fuego", "smoke_recall": "Recall humo",
    "fire_recall": "Recall fuego", "macro_recall": "Recall macro",
    "minimum_class_recall": "Peor recall", "micro_precision": "Precisión micro",
    "micro_recall": "Recall micro", "micro_f1": "F1 micro",
    "negative_images_with_alarm": "Negativas con alarma",
})
display(table.style.format({
    "Umbral humo": "{:.2f}", "Umbral fuego": "{:.2f}", "Recall humo": "{:.2%}",
    "Recall fuego": "{:.2%}", "Recall macro": "{:.2%}", "Peor recall": "{:.2%}",
    "Precisión micro": "{:.2%}", "Recall micro": "{:.2%}", "F1 micro": "{:.2%}",
}).hide(axis="index"))
display(Image(filename=str(RUN_DIR / "figures" / "01_full_training_macro_recall.png"), width=1100))
'''),
        md("""El criterio lexicográfico vigente coloca primero a YOLO26s
640→640. La diferencia de recall macro respecto a YOLO26s 768→768 es pequeña y
se obtiene principalmente por mayor recall de humo, no por una mejora de fuego.
"""),
        md("### 3. Intercambio entre las dos configuraciones YOLO26s"),
        code('''comparison = full_sensitivity[
    full_sensitivity.configuration.isin(["yolo26s_640_eval640", "yolo26s_768_eval768"])
][["label", "smoke_recall", "fire_recall", "minimum_class_recall", "macro_recall",
   "micro_precision", "micro_recall", "micro_f1", "smoke_fp", "fire_fp", "smoke_fn", "fire_fn"]]
display(comparison.rename(columns={
    "label": "Configuración", "smoke_recall": "Recall humo", "fire_recall": "Recall fuego",
    "minimum_class_recall": "Peor recall", "macro_recall": "Recall macro",
    "micro_precision": "Precisión", "micro_recall": "Recall micro", "micro_f1": "F1",
    "smoke_fp": "FP humo", "fire_fp": "FP fuego", "smoke_fn": "FN humo", "fire_fn": "FN fuego",
}).style.format({
    "Recall humo": "{:.2%}", "Recall fuego": "{:.2%}", "Peor recall": "{:.2%}",
    "Recall macro": "{:.2%}", "Precisión": "{:.2%}", "Recall micro": "{:.2%}", "F1": "{:.2%}",
}).hide(axis="index"))

display(Markdown(
    f"Frente a 640→640, 768→768 pierde **{100*(winner.smoke_recall-previous.smoke_recall):.2f} puntos** de recall de humo, "
    f"pero gana **{100*(previous.fire_recall-winner.fire_recall):.2f} puntos** de recall de fuego, "
    f"**{100*(previous.minimum_class_recall-winner.minimum_class_recall):.2f} puntos** en la peor clase, "
    f"**{100*(previous.micro_precision-winner.micro_precision):.2f} puntos** de precisión y "
    f"**{100*(previous.micro_f1-winner.micro_f1):.2f} puntos** de F1."
))
display(Image(filename=str(RUN_DIR / "figures" / "02_full_training_class_recall.png"), width=1100))
'''),
        md("### 4. Perfil equilibrado: máximo F1 dentro del 1 %"),
        code('''full_balanced = ranked[
    (ranked.comparison_group == "full_training") & (ranked.scenario == "balanced")
].sort_values("rank")
balanced_table = full_balanced[[
    "rank", "label", "smoke_threshold", "fire_threshold", "micro_precision",
    "micro_recall", "micro_f1", "macro_recall", "negative_images_with_alarm",
]].rename(columns={
    "rank": "Puesto", "label": "Configuración", "smoke_threshold": "Umbral humo",
    "fire_threshold": "Umbral fuego", "micro_precision": "Precisión", "micro_recall": "Recall",
    "micro_f1": "F1", "macro_recall": "Recall macro",
    "negative_images_with_alarm": "Negativas con alarma",
})
display(balanced_table.style.format({
    "Umbral humo": "{:.2f}", "Umbral fuego": "{:.2f}", "Precisión": "{:.2%}",
    "Recall": "{:.2%}", "F1": "{:.2%}", "Recall macro": "{:.2%}",
}).hide(axis="index"))
display(Image(filename=str(RUN_DIR / "figures" / "03_full_training_precision_recall.png"), width=950))
'''),
        md("""YOLOv8s 768→640 obtiene el máximo F1, pero su ventaja sobre
YOLO26s 768→768 es inferior a una centésima de punto porcentual. YOLO26s mantiene
mayor recall macro en ese perfil, por lo que el resultado es un empate práctico
en F1, no una superioridad material de YOLOv8s.
"""),
        md("### 5. Métricas estándar y tamaños"),
        code('''standard = pd.read_csv(RUN_DIR / "standard_metrics.csv")
standard_all = standard[standard.scope == "all"][[
    "label", "precision", "recall", "mAP50", "mAP50_95", "inference_ms"
]].sort_values("mAP50_95", ascending=False)
display(standard_all.rename(columns={
    "label": "Configuración", "precision": "Precisión", "recall": "Recall",
    "mAP50_95": "mAP50-95", "inference_ms": "Inferencia ms",
}).style.format({
    "Precisión": "{:.2%}", "Recall": "{:.2%}", "mAP50": "{:.2%}",
    "mAP50-95": "{:.2%}", "Inferencia ms": "{:.3f}",
}).hide(axis="index"))

sizes = pd.read_csv(RUN_DIR / "size_metrics_sensitivity.csv")
top_sizes = sizes[sizes.configuration.isin([
    "yolo26s_640_eval640", "yolo26s_768_eval768", "yolov8s_768_eval640"
])][["label", "class_name", "size_band", "gt_boxes", "recall"]]
display(top_sizes.rename(columns={
    "label": "Configuración", "class_name": "Clase", "size_band": "Tamaño",
    "gt_boxes": "Cajas GT", "recall": "Recall",
}).style.format({"Recall": "{:.2%}"}).hide(axis="index"))
'''),
        md("### 6. Ensayos de hiperparámetros de 50 épocas"),
        code('''hp = ranked[
    (ranked.comparison_group == "hp_screening_50ep") & (ranked.scenario == "sensitivity")
].sort_values("rank")
display(hp[[
    "rank", "label", "smoke_threshold", "fire_threshold", "smoke_recall", "fire_recall",
    "macro_recall", "micro_precision", "micro_f1", "negative_images_with_alarm",
]].rename(columns={
    "rank": "Puesto", "label": "Ensayo", "smoke_threshold": "Umbral humo",
    "fire_threshold": "Umbral fuego", "smoke_recall": "Recall humo",
    "fire_recall": "Recall fuego", "macro_recall": "Recall macro",
    "micro_precision": "Precisión", "micro_f1": "F1",
    "negative_images_with_alarm": "Negativas con alarma",
}).style.format({
    "Umbral humo": "{:.2f}", "Umbral fuego": "{:.2f}", "Recall humo": "{:.2%}",
    "Recall fuego": "{:.2%}", "Recall macro": "{:.2%}", "Precisión": "{:.2%}", "F1": "{:.2%}",
}).hide(axis="index"))
display(Image(filename=str(RUN_DIR / "figures" / "04_hp_screening_macro_recall.png"), width=1000))
'''),
        md("""Estos cuatro modelos se incluyen para completar el inventario,
pero no participan en la selección principal. Sus 50 épocas constituyen un
cribado de hiperparámetros y no un entrenamiento final equivalente.
"""),
        md("## Takeaways"),
        code('''display(Markdown(
    "1. **Aplicación literal del criterio:** YOLO26s 640→640 gana por recall macro.  \\n"
    "2. **Mejor equilibrio entre clases y mejor fuego:** YOLO26s 768→768 mantiene mayor "
    "recall de fuego, peor-clase, precisión y F1.  \\n"
    "3. **Máximo F1:** YOLOv8s 768→640, empatado en la práctica con YOLO26s 768→768.  \\n"
    "4. **Decisión:** se mantiene YOLO26s 768→768 con umbrales 0,36/0,16, el modelo "
    "elegido en el notebook 09, por su mejor equilibrio entre clases y su mayor recall de fuego."
))
'''),
    ]
    notebook = new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {"display_name": "Python 3 (TFM Docker)", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
    )
    for index, cell in enumerate(notebook.cells):
        cell.id = f"all-models-01pct-{index:02d}"
    return notebook


def main():
    target = ROOT / "notebooks" / NAME
    nbformat.write(build_notebook(), target)
    print(target)


if __name__ == "__main__":
    main()
