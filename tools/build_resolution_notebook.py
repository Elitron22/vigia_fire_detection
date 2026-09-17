"""Construye el notebook 06: experimento controlado de resolución."""
from __future__ import annotations

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


ROOT = Path(__file__).resolve().parents[1]
NAME = "06_DFire_comparacion_resolucion.ipynb"


def build_notebook():
    def md(value):
        return new_markdown_cell(value.strip())

    def code(value, tag=None):
        cell = new_code_cell(value.strip() + "\n")
        if tag:
            cell.metadata["tags"] = [tag]
        return cell

    cells = [
        md("""# D-Fire · Comparación de resolución 640, 768 y 1024

**Fase 06 del TFM — experimento controlado en validación.**

Este cuaderno compara tres entrenamientos YOLOv8s sobre el mismo dataset y la
misma partición. Se separa la resolución de entrenamiento de la usada al evaluar
mediante una cuadrícula 3×3. El test permanece reservado.
"""),
        code('''from pathlib import Path
import os
import sys
import numpy as np
import pandas as pd
from IPython.display import display, Markdown, Image

roots = [Path(os.environ["TFM_PROJECT_ROOT"])] if os.environ.get("TFM_PROJECT_ROOT") else []
roots += [Path.cwd(), *Path.cwd().parents]
PROJECT_ROOT = next((p for p in roots if (p / "tfm_pipeline.py").is_file()), None)
if PROJECT_ROOT is None:
    raise FileNotFoundError("Abrir el notebook dentro de TFM o definir TFM_PROJECT_ROOT.")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import tfm_pipeline as pipeline

CONFIG_PATH = PROJECT_ROOT / "configs" / "resolution_comparison.yaml"
RUN_COMPARISON = False
RUN_ID = None
''', "parameters"),
        code('''result_parent = PROJECT_ROOT / "artifacts" / "06_resolution_comparison" / "validation"
if RUN_COMPARISON:
    from tools.run_resolution_comparison import run
    RESULT_DIR = run(CONFIG_PATH)
elif RUN_ID:
    if Path(RUN_ID).name != RUN_ID:
        raise ValueError("RUN_ID debe ser un identificador, no una ruta.")
    RESULT_DIR = result_parent / RUN_ID
else:
    latest = pipeline.read_json(result_parent / "latest.json")
    RESULT_DIR = PROJECT_ROOT / latest["run_rel"]
    if pipeline.sha256_file(RESULT_DIR / "run_summary.json") != latest["summary_sha256"]:
        raise ValueError("El índice latest.json no coincide con el resumen.")

summary = pipeline.read_json(RESULT_DIR / "run_summary.json")
if summary["status"] != "complete" or summary["split"] != "val":
    raise ValueError("Se necesita una ejecución completa de validación.")
if summary["test_inference_executed"]:
    raise ValueError("Este análisis no debe utilizar test.")
for relative, expected in summary["output_hashes"].items():
    if pipeline.sha256_file(RESULT_DIR / relative) != expected:
        raise ValueError(f"Artefacto modificado: {relative}")

metrics = pd.read_csv(RESULT_DIR / "threshold_metrics.csv")
candidates = pd.read_csv(RESULT_DIR / "scenario_candidates.csv")
standard = pd.read_csv(RESULT_DIR / "standard_metrics.csv")
image_detection = pd.read_csv(RESULT_DIR / "image_detection_metrics.csv")
sizes = pd.read_csv(RESULT_DIR / "size_metrics.csv")
paired = pd.read_csv(RESULT_DIR / "paired_comparison.csv")
resources = pd.read_csv(RESULT_DIR / "training_resources.csv")
review = pd.read_csv(RESULT_DIR / "review_selection.csv")
error_summary = pd.read_csv(RESULT_DIR / "error_review_summary.csv")
error_examples = pd.read_csv(RESULT_DIR / "error_review_examples.csv")
selected = candidates[(candidates.scenario == "alarm_02pct") & candidates.feasible].copy()

variant_specs = summary["config"]["training_variants"]
ordered_variants = sorted(variant_specs, key=lambda v: int(variant_specs[v]["trained_imgsz"]))
native_keys = [f"{v}_eval{int(variant_specs[v]['trained_imgsz'])}" for v in ordered_variants]
labels = {
    f"{variant}_eval{eval_size}": (
        f"Entrena {int(variant_specs[variant]['trained_imgsz'])} → evalúa {int(eval_size)}"
    )
    for variant in ordered_variants
    for eval_size in summary["config"]["evaluation_sizes"]
}
operating = selected.merge(
    image_detection,
    left_on=["model_key", "threshold"],
    right_on=["configuration", "threshold"],
    validate="one_to_one",
)
native = operating[operating.model_key.isin(native_keys)].set_index("model_key").loc[native_keys]
print(f"Ejecución: {summary['run_id']} · {summary['images_per_configuration']:,} imágenes por configuración")
print(f"Dataset: {summary['dataset_version']} · test consultado: {summary['test_inference_executed']}")
print(f"Resultados: {RESULT_DIR}")
'''),
        md("""## tl;dr"""),
        code('''baseline_key = native_keys[0]
focus_key = f"{summary['config']['focus_variant']}_eval{int(summary['config']['focus_evaluation_size'])}"
best_recall_key = native.fire_recall.idxmax()
best_f1_key = native.micro_f1.idxmax()
base = native.loc[baseline_key]
focus = operating.set_index("model_key").loc[focus_key]
base_std = standard[(standard.configuration == baseline_key) & (standard.scope == "fire")].iloc[0]
focus_std = standard[(standard.configuration == focus_key) & (standard.scope == "fire")].iloc[0]
focus_variant = focus_key.rsplit("_eval", 1)[0]
focus_cost = resources.set_index("variant").loc[focus_variant, "training_time_ratio_vs_640"]

display(Markdown(f"""El modelo entrenado a **768** e inferido a **640** obtiene
**{focus.fire_recall:.1%} de recall de fuego**
y **{focus.micro_f1:.1%} de F1 micro** en su punto operativo, frente a
**{base.fire_recall:.1%}** y **{base.micro_f1:.1%}** a 640. El cambio es
**{focus.fire_recall-base.fire_recall:+.1%}** en recall y
**{focus.micro_f1-base.micro_f1:+.1%}** en F1, con
**{int(focus.negative_images_with_alarm)}/783** negativas con alarma.

El mAP50-95 estándar de fuego cambia de **{base_std.mAP50_95:.3f}** a
**{focus_std.mAP50_95:.3f}** y entrenar a 768 costó **{focus_cost:.2f}×** respecto
a 640. El mayor recall nativo lo consigue **{labels[best_recall_key]}** y el mayor
F1 nativo **{labels[best_f1_key]}**. La decisión se toma con estas métricas y el
coste, no con una sola cifra."""))
'''),
        md("""## Contexto y método

Cada combinación usa las mismas 1.721 imágenes de validación. Se calculan las
métricas estándar de Ultralytics y un barrido de 21 umbrales de confianza. El
punto operativo maximiza el recall medio de humo y fuego con un máximo del 2 %
de imágenes negativas con alguna alarma, equivalente a 15 de 783.

Un acierto exige clase correcta e IoU ≥ 0,50. La supresión NMS usa IoU 0,70.
También se mide detección por imagen, recall de fuego por tamaño y cambios
emparejados respecto a 640. Solo hay una semilla por resolución.
"""),
        md("""## Resultados nativos"""),
        code('''native_table = native[[
    "threshold", "smoke_recall", "fire_recall", "fire_image_recall",
    "micro_precision", "micro_f1", "negative_images_with_alarm", "negative_alarm_rate"
]].copy()
native_table.index = [labels[key] for key in native_table.index]
native_table.index.name = "Configuración"
native_table.columns = [
    "Umbral", "Recall humo", "Recall fuego", "Recall imágenes con fuego",
    "Precisión micro", "F1 micro", "Negativas con alarma", "Tasa de alarma",
]
display(native_table.style.format({
    "Umbral": "{:.2f}", "Recall humo": "{:.1%}", "Recall fuego": "{:.1%}",
    "Recall imágenes con fuego": "{:.1%}", "Precisión micro": "{:.1%}",
    "F1 micro": "{:.1%}", "Negativas con alarma": "{:.0f}", "Tasa de alarma": "{:.2%}",
}))
display(Image(filename=str(RESULT_DIR / "figures" / "01_native_operating_comparison.png"), width=1100,
              alt="Comparación operativa nativa a 640, 768 y 1024"))
'''),
        md("""## Cuadrícula entrenamiento × inferencia"""),
        code('''all_operating = operating[[
    "model_key", "threshold", "smoke_recall", "fire_recall", "fire_image_recall",
    "micro_precision", "micro_f1", "negative_images_with_alarm", "negative_alarm_rate"
]].copy()
all_operating["model_key"] = all_operating.model_key.map(labels)
all_operating = all_operating.sort_values(["fire_recall", "micro_f1"], ascending=False)
all_operating.columns = [
    "Configuración", "Umbral", "Recall humo", "Recall fuego", "Recall imágenes con fuego",
    "Precisión micro", "F1 micro", "Negativas con alarma", "Tasa de alarma",
]
display(all_operating.style.format({
    "Umbral": "{:.2f}", "Recall humo": "{:.1%}", "Recall fuego": "{:.1%}",
    "Recall imágenes con fuego": "{:.1%}", "Precisión micro": "{:.1%}",
    "F1 micro": "{:.1%}", "Negativas con alarma": "{:.0f}", "Tasa de alarma": "{:.2%}",
}).hide(axis="index"))
display(Image(filename=str(RESULT_DIR / "figures" / "02_fire_tradeoff.png"), width=850,
              alt="Recall de fuego frente a falsas alarmas"))
'''),
        md("""## Métricas estándar y coste"""),
        code('''std_fire = standard[standard.scope == "fire"][[
    "configuration", "precision", "recall", "mAP50", "mAP50_95",
    "inference_ms", "peak_cuda_allocated_gib"
]].copy()
std_fire["configuration"] = std_fire.configuration.map(labels)
std_fire.columns = [
    "Configuración", "Precisión", "Recall", "mAP50", "mAP50-95",
    "Inferencia ms/imagen", "Pico CUDA GiB",
]
display(std_fire.style.format({
    "Precisión": "{:.1%}", "Recall": "{:.1%}", "mAP50": "{:.1%}", "mAP50-95": "{:.1%}",
    "Inferencia ms/imagen": "{:.2f}", "Pico CUDA GiB": "{:.2f}",
}).hide(axis="index"))
display(resources.rename(columns={
    "variant": "Entrenamiento", "trained_imgsz": "imgsz", "epochs_completed": "Épocas",
    "best_epoch": "Mejor época", "training_hours": "Horas",
    "best_training_map50_95": "Mejor mAP50-95", "training_time_ratio_vs_640": "Coste relativo",
})[["Entrenamiento", "imgsz", "Épocas", "Mejor época", "Horas", "Mejor mAP50-95", "Coste relativo"]]
 .style.format({"Horas": "{:.2f}", "Mejor mAP50-95": "{:.4f}", "Coste relativo": "{:.2f}×"})
 .hide(axis="index"))
print("Los tiempos de inferencia proceden de val con batch=8 y sirven como comparación relativa.")
display(Image(filename=str(RESULT_DIR / "figures" / "03_factorial_resolution.png"), width=1000,
              alt="Matrices de resolución de entrenamiento e inferencia"))
'''),
        md("""## Fuego por tamaño y comparación emparejada"""),
        code('''display(Image(filename=str(RESULT_DIR / "figures" / "04_fire_recall_by_size.png"), width=850,
              alt="Recall de fuego por tamaño de caja"))
picks = native.threshold.to_dict()
size_native = pd.concat([
    sizes[
        (sizes.model_key == key)
        & np.isclose(sizes.threshold, threshold)
        & (sizes.class_name == "fire")
    ]
    for key, threshold in picks.items()
])
size_table = size_native.pivot(index="size_band", columns="model_key", values="recall")
size_table = size_table.loc[["small", "medium", "large"], native_keys]
size_table.index = ["Pequeño (<1 %)", "Mediano (1–10 %)", "Grande (≥10 %)"]
size_table.columns = [labels[key] for key in native_keys]
display(size_table.style.format("{:.1%}"))

reported_pair_keys = set(native_keys[1:] + [focus_key])
paired_view = paired[paired.candidate_key.isin(reported_pair_keys)].copy()
paired_view["Candidato"] = paired_view.candidate_key.map(labels)
paired_view["Unidad"] = paired_view.unit.map({
    "fire_positive_image": "Imagen con fuego", "fire_box": "Caja de fuego"
})
paired_view = paired_view[[
    "Candidato", "Unidad", "n", "gains", "losses",
    "baseline_rate", "candidate_rate", "p_value"
]]
paired_view.columns = [
    "Candidato", "Unidad", "n", "Recuperaciones", "Pérdidas",
    "Tasa 640", "Tasa candidato", "p exacta",
]
display(paired_view.style.format({
    "Tasa 640": "{:.1%}", "Tasa candidato": "{:.1%}", "p exacta": "{:.4f}",
}).hide(axis="index"))
'''),
        md("""## Revisión cualitativa con inferencia a 640"""),
        code('''display(Image(filename=str(RESULT_DIR / "figures" / "05_changed_fire_examples.png"), width=1100,
              alt="Ejemplos de detecciones de fuego que cambian al entrenar a 768"))
review_view = review.rename(columns={
    "filename": "Imagen", "area_fraction": "Fracción de área", "size_band": "Tamaño",
    "change": "Cambio", "status_baseline": "640 → 640", "status_candidate": "768 → 640",
})
display(review_view[["Imagen", "Fracción de área", "Tamaño", "Cambio", "640 → 640", "768 → 640"]]
        .style.format({"Fracción de área": "{:.3%}"}).hide(axis="index"))
'''),
        md("""Cajas blancas: anotaciones. Azul: predicción emparejada correcta. Naranja:
predicción no emparejada. La selección sirve para entender errores concretos;
no estima su frecuencia en el conjunto.
"""),
        md("""## Desglose de errores"""),
        code('''review_error_keys = list(dict.fromkeys(native_keys + [focus_key]))
native_errors = error_summary[error_summary.configuration.isin(review_error_keys)].copy()
native_errors["configuration"] = native_errors.configuration.map(labels)
native_errors.columns = [
    "Configuración", "Umbral", "FN humo", "FN fuego", "FP humo", "FP fuego",
    "FP en negativas", "FP en positivas", "Negativas con alarma",
    "Positivas con FP", "Imágenes de fuego perdidas", "Imágenes de incidente perdidas",
]
display(native_errors.style.format({"Umbral": "{:.2f}"}).hide(axis="index"))
display(error_examples.head(15).rename(columns={
    "filename": "Imagen", "category": "Categoría", "gt_count": "Cajas GT",
    "pred_count": "Predicciones", "smoke_tp": "TP humo", "fire_tp": "TP fuego",
    "smoke_fp": "FP humo", "fire_fp": "FP fuego", "smoke_fn": "FN humo",
    "fire_fn": "FN fuego", "error_priority": "Errores",
})[["Imagen", "Categoría", "Cajas GT", "Predicciones", "TP humo", "TP fuego",
    "FP humo", "FP fuego", "FN humo", "FN fuego", "Errores"]]
 .style.hide(axis="index"))
'''),
        md("""## Conclusiones y límites"""),
        code('''focus_pair = paired[paired.candidate_key == focus_key].set_index("unit")
display(Markdown(f"""Frente a 640 → 640, el modelo 768 → 640 recupera
**{int(focus_pair.loc['fire_box', 'gains'])}** cajas de fuego y pierde
**{int(focus_pair.loc['fire_box', 'losses'])}**; en imágenes con fuego recupera
**{int(focus_pair.loc['fire_positive_image', 'gains'])}** y pierde
**{int(focus_pair.loc['fire_positive_image', 'losses'])}**.

La evaluación permite decidir la resolución de referencia y el punto operativo
en validación. No demuestra variabilidad entre entrenamientos porque se ha usado
una sola semilla, ni mide todavía falsas alarmas por minuto en vídeo. El umbral
elegido debe congelarse antes de consultar test."""))
'''),
        md("""## Trazabilidad"""),
        code('''print("Configuración:", RESULT_DIR / "config.yaml")
print("Resumen narrativo:", RESULT_DIR / "RESUMEN_RESOLUCION.md")
print("Barrido completo:", RESULT_DIR / "threshold_metrics.csv")
print("Métricas estándar:", RESULT_DIR / "standard_metrics.csv")
print("Comparación emparejada:", RESULT_DIR / "paired_comparison.csv")
print("Resumen de errores:", RESULT_DIR / "error_review_summary.csv")
print("Casos prioritarios:", RESULT_DIR / "error_review_examples.csv")
print("Código ejecutado fijado en:", RESULT_DIR / "code")
print("Verificación independiente: python tools/verify_resolution_comparison.py")
print("Reconstruir notebook: python tools/build_resolution_notebook.py")
'''),
    ]
    notebook = new_notebook(
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
    for index, cell in enumerate(notebook.cells):
        cell.id = f"resolution-{index:02d}"
    return notebook


def main():
    target = ROOT / "notebooks" / NAME
    nbformat.write(build_notebook(), target)
    print(target)


if __name__ == "__main__":
    main()
