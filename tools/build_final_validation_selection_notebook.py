"""Construye el notebook 09 de selección final sobre validación."""
from __future__ import annotations

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


ROOT = Path(__file__).resolve().parents[1]
NAME = "09_DFire_seleccion_final_validacion.ipynb"


def build_notebook():
    def md(value):
        return new_markdown_cell(value.strip())

    def code(value, tag=None):
        cell = new_code_cell(value.strip() + "\n")
        if tag:
            cell.metadata["tags"] = [tag]
        return cell

    cells = [
        md("""# D-Fire · Selección final de modelo en validación

Este cuaderno cierra la comparación entre **YOLO26s 768→768** y
**YOLOv8s 768→640** usando exclusivamente el split de validación congelado.
El test permanece bloqueado y no participa en umbrales ni en la elección.
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

CONFIG_PATH = PROJECT_ROOT / "configs" / "final_validation_selection.yaml"
RUN_SELECTION = False  # True recalcula tablas desde predicciones de validación en caché.
config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
assert config["split"] == "val" and config["test_locked"] is True

if RUN_SELECTION:
    subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "tools" / "run_final_validation_selection.py"),
         "--config", str(CONFIG_PATH)], check=True, cwd=PROJECT_ROOT,
    )

latest = pipeline.read_json(PROJECT_ROOT / config["output_parent"] / "latest.json")
RUN_DIR = PROJECT_ROOT / latest["run_rel"]
summary = pipeline.read_json(RUN_DIR / "run_summary.json")
assert summary["split"] == "val" and summary["test_inference_executed"] is False
assert not any(summary["source_test_flags"])

operating = pd.read_csv(RUN_DIR / "final_operating_points.csv").sort_values("selection_rank")
winner = operating.iloc[0]
display(Markdown(
    f"**Selección:** {winner['label']} · humo **{winner.smoke_threshold:.2f}** · "
    f"fuego **{winner.fire_threshold:.2f}**.  \\n"
    f"Recall macro **{winner.macro_recall:.2%}**, F1 micro **{winner.micro_f1:.2%}** "
    f"y alarmas en **{int(winner.negative_images_with_alarm)}/{int(winner.negative_images)}** "
    f"negativas ({winner.negative_alarm_rate:.3%})."
))
'''),
        md("""## Contexto y método

### Supuestos clave

- La prioridad es maximizar el recall conjunto de humo y fuego después de exigir
  como máximo un **1 % de imágenes negativas con alarma**.
- Con 783 negativas, el límite efectivo es 7 imágenes: 8/783 equivale al 1,02 %.
- Los umbrales se ajustan por clase; las métricas estándar se muestran aparte.
- La comparación usa una sola semilla por arquitectura. La decisión es el
  candidato previo a test, no una estimación definitiva de generalización.
"""),
        md("### 1. Verificar fuentes y trazabilidad"),
        code('''verification = subprocess.run(
    [sys.executable, str(PROJECT_ROOT / "tools" / "verify_final_validation_selection.py")],
    check=True, cwd=PROJECT_ROOT, capture_output=True, text=True,
)
verification_report = json.loads(verification.stdout)
display(pd.DataFrame([verification_report]))

source_rows = []
for source_name, pointer in summary["source_runs"].items():
    source_rows.append({"fuente": source_name, "run_id": pointer["run_id"], "ruta": pointer["run_rel"]})
display(pd.DataFrame(source_rows))
'''),
        md("""## Datos

La población contiene 1.721 imágenes de validación, 783 de ellas sin cajas de
humo ni fuego. Las predicciones se generaron a confianza 0,01 y se reutilizan
offline para evaluar 2.025 parejas de umbrales por modelo. El IoU de acierto es 0,50.
"""),
        code('''class_metrics = pd.read_csv(RUN_DIR / "class_metrics.csv")
size_metrics = pd.read_csv(RUN_DIR / "size_metrics.csv")
errors = pd.read_csv(RUN_DIR / "error_comparison.csv")
standard = pd.read_csv(RUN_DIR / "standard_metrics.csv")
paired = pd.read_csv(RUN_DIR / "paired_comparison_768.csv")
threshold_grid = pd.read_csv(RUN_DIR / "class_threshold_grid_01pct.csv")

coverage = pd.DataFrame([{
    "imágenes": int(winner.images),
    "negativas": int(winner.negative_images),
    "modelos": operating.candidate.nunique(),
    "parejas de umbral": len(threshold_grid),
    "parejas por modelo": len(threshold_grid) // operating.candidate.nunique(),
    "test consultado": summary["test_inference_executed"],
}])
display(coverage)
'''),
        md("## Resultados"),
        md("### 2. Tabla definitiva y presupuesto de falsas alarmas"),
        code('''global_columns = [
    "selection_rank", "label", "smoke_threshold", "fire_threshold",
    "micro_precision", "micro_recall", "micro_f1", "macro_recall",
    "minimum_class_recall", "negative_images_with_alarm", "negative_alarm_rate",
]
global_table = operating[global_columns].rename(columns={
    "selection_rank": "Rango", "label": "Modelo", "smoke_threshold": "Umbral humo",
    "fire_threshold": "Umbral fuego", "micro_precision": "Precisión micro",
    "micro_recall": "Recall micro", "micro_f1": "F1 micro",
    "macro_recall": "Recall macro", "minimum_class_recall": "Recall mínimo",
    "negative_images_with_alarm": "Negativas con alarma",
    "negative_alarm_rate": "Tasa alarma negativa",
})
display(global_table.style.format({
    "Umbral humo": "{:.2f}", "Umbral fuego": "{:.2f}",
    "Precisión micro": "{:.2%}", "Recall micro": "{:.2%}", "F1 micro": "{:.2%}",
    "Recall macro": "{:.2%}", "Recall mínimo": "{:.2%}", "Tasa alarma negativa": "{:.3%}",
}).hide(axis="index"))
display(Image(filename=str(RUN_DIR / "figures" / "01_global_operating_comparison.png"), width=1100))
'''),
        md("""Los dos modelos cumplen exactamente el mismo presupuesto observado:
7/783 negativas. YOLO26s queda seleccionado porque mejora recall macro, recall
de la clase más débil y F1, mientras la precisión permanece prácticamente empatada.
"""),
        md("### 3. Coste del límite del 1 % frente al 2 %"),
        code('''budget_comparison = pd.read_csv(RUN_DIR / "alarm_budget_comparison.csv")
budget_table = budget_comparison[[
    "label", "alarm_budget", "smoke_threshold", "fire_threshold",
    "micro_precision", "micro_recall", "micro_f1", "macro_recall",
    "negative_images_with_alarm", "negative_alarm_rate",
]].rename(columns={
    "label": "Modelo", "alarm_budget": "Presupuesto", "smoke_threshold": "Umbral humo",
    "fire_threshold": "Umbral fuego", "micro_precision": "Precisión micro",
    "micro_recall": "Recall micro", "micro_f1": "F1 micro", "macro_recall": "Recall macro",
    "negative_images_with_alarm": "Negativas con alarma", "negative_alarm_rate": "Tasa observada",
})
display(budget_table.style.format({
    "Presupuesto": "{:.0%}", "Umbral humo": "{:.2f}", "Umbral fuego": "{:.2f}",
    "Precisión micro": "{:.2%}", "Recall micro": "{:.2%}", "F1 micro": "{:.2%}",
    "Recall macro": "{:.2%}", "Tasa observada": "{:.3%}",
}).hide(axis="index"))
display(Image(filename=str(RUN_DIR / "figures" / "04_alarm_budget_comparison.png"), width=1050))
'''),
        md("""El 1 % reduce el recall macro unos tres puntos en ambos modelos,
principalmente porque exige elevar el umbral de humo. A cambio mejora la precisión
y limita las alarmas a 7 imágenes negativas. YOLO26s sigue siendo el candidato
con mayor recall macro bajo ambos presupuestos.
"""),
        md("### 4. Métricas estándar y por clase"),
        code('''standard_table = standard[standard.scope == "all"][[
    "label", "precision", "recall", "mAP50", "mAP50_95", "inference_ms"
]].rename(columns={"label": "Modelo", "precision": "Precisión", "recall": "Recall",
                   "mAP50_95": "mAP50-95", "inference_ms": "Inferencia ms"})
display(standard_table.style.format({
    "Precisión": "{:.2%}", "Recall": "{:.2%}", "mAP50": "{:.2%}",
    "mAP50-95": "{:.2%}", "Inferencia ms": "{:.3f}",
}).hide(axis="index"))

class_table = class_metrics[["label", "class_name", "threshold", "precision", "recall", "f1", "tp", "fp", "fn"]].copy()
class_table["class_name"] = class_table.class_name.map({"smoke": "Humo", "fire": "Fuego"})
class_table = class_table.rename(columns={"label": "Modelo", "class_name": "Clase", "threshold": "Umbral",
    "precision": "Precisión", "recall": "Recall", "f1": "F1", "tp": "TP", "fp": "FP", "fn": "FN"})
display(class_table.style.format({"Umbral": "{:.2f}", "Precisión": "{:.2%}",
                                  "Recall": "{:.2%}", "F1": "{:.2%}"}).hide(axis="index"))
'''),
        md("### 5. Recall por tamaño"),
        code('''size_table = size_metrics[["label", "class_name", "size_band", "gt_boxes", "tp", "fn", "recall"]].copy()
size_table["class_name"] = size_table.class_name.map({"smoke": "Humo", "fire": "Fuego"})
size_table["size_band"] = size_table.size_band.map({"small": "Pequeño", "medium": "Mediano", "large": "Grande"})
size_table = size_table.rename(columns={"label": "Modelo", "class_name": "Clase", "size_band": "Tamaño",
    "gt_boxes": "Cajas GT", "tp": "TP", "fn": "FN", "recall": "Recall"})
display(size_table.style.format({"Recall": "{:.2%}"}).hide(axis="index"))
display(Image(filename=str(RUN_DIR / "figures" / "02_class_and_size_recall.png"), width=1200))
'''),
        md("""YOLO26s mejora las tres bandas de humo y el fuego grande. Empata en
fuego mediano y queda 0,40 puntos porcentuales por debajo en fuego pequeño.
Las cajas pequeñas son el grupo mayoritario para fuego (760/1.155).
"""),
        md("### 6. Falsos positivos y falsos negativos"),
        code('''display(errors.rename(columns={
    "label": "Modelo", "smoke_fp": "FP humo", "smoke_fn": "FN humo",
    "fire_fp": "FP fuego", "fire_fn": "FN fuego",
    "positive_image_fp_boxes": "FP en positivas", "negative_image_fp_boxes": "FP en negativas",
    "negative_images_with_alarm": "Negativas con alarma",
}).drop(columns=["candidate"]).style.hide(axis="index"))
display(Image(filename=str(RUN_DIR / "figures" / "03_fp_fn_comparison.png"), width=1050))

manual_review = pd.read_csv(RUN_DIR / "manual_fp_review_summary.csv")
display(manual_review[["manual_category", "images"]].rename(
    columns={"manual_category": "Categoría manual", "images": "Imágenes"}
).style.hide(axis="index"))
'''),
        md("""La mayoría de las cajas FP aparece en imágenes que sí contienen un
incendio. La inspección dirigida de YOLO26s encontró sobre todo diferencias de
extensión, duplicados, anotación posiblemente incompleta, ambigüedad humo/fuego,
luces artificiales y objetos rojos. La muestra manual es diagnóstica y corresponde
al umbral común 0,16, no exactamente al punto final 0,36/0,16.
"""),
        md("### 7. Evidencia emparejada a resolución común"),
        code('''paired_table = paired[["class_name", "unit", "n", "baseline_rate", "candidate_rate", "gains", "losses", "p_value"]].copy()
paired_table["class_name"] = paired_table.class_name.map({"smoke": "Humo", "fire": "Fuego"})
paired_table["unit"] = paired_table.unit.map({"positive_image": "Imagen positiva", "box": "Caja"})
paired_table = paired_table.rename(columns={"class_name": "Clase", "unit": "Unidad", "baseline_rate": "YOLOv8s",
    "candidate_rate": "YOLO26s", "gains": "Ganancias", "losses": "Pérdidas", "p_value": "p exacta"})
display(paired_table.style.format({"YOLOv8s": "{:.2%}", "YOLO26s": "{:.2%}",
                                   "p exacta": "{:.3g}"}).hide(axis="index"))
'''),
        md("""Esta evidencia complementaria procede del análisis previo con ambos
modelos inferidos a 768 y presupuesto del 2 %. YOLO26s recuperó más casos en las
cuatro comparaciones. Con corrección Bonferroni permanecen tres de cuatro
diferencias; no se utiliza la significación como criterio directo de selección.
"""),
        md("## Takeaways"),
        code('''display(Markdown(
    f"1. **Candidato congelado:** `{summary['selected_experiment_id']}`.  \\n"
    f"2. **Configuración:** 768 px, humo `{summary['selected_smoke_threshold']:.2f}`, "
    f"fuego `{summary['selected_fire_threshold']:.2f}`.  \\n"
    "3. **Criterio satisfecho:** 7/783 negativas con alarma (0,894 %).  \\n"
    "4. **Siguiente paso:** ejecutar esta configuración una sola vez sobre test, "
    "sin reajustar pesos, resolución ni umbrales."
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
        cell.id = f"final-validation-{index:02d}"
    return notebook


def main():
    target = ROOT / "notebooks" / NAME
    nbformat.write(build_notebook(), target)
    print(target)


if __name__ == "__main__":
    main()
