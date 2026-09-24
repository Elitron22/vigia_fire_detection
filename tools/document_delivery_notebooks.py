"""Añade una guía de entrega homogénea a los notebooks sin borrar sus salidas.

El script es idempotente: actualiza la celda etiquetada ``delivery-guide`` si ya
existe y la inserta tras la presentación inicial en caso contrario.
"""

from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"
GUIDE_TAG = "delivery-guide"


GUIDES = {
    "01_DFire_preparacion_dataset.ipynb": {
        "role": "Prepara la versión del dataset que usan todos los experimentos: revisa las anotaciones, corrige errores y crea las particiones.",
        "use": "El primero, para ver cómo se prepararon los datos.",
        "mode": "No modifica nada (`RUN_PREPARATION=False`): carga y muestra la versión ya preparada. Necesita el dataset original en `data/D-Fire`.",
        "inputs": "Dataset D-Fire original en `data/D-Fire`.",
        "outputs": "`artifacts/datasets/dfire_seed42_val10_v1/` (incluida en el repositorio): imágenes de cada partición, correcciones y ficheros `data.yaml`.",
        "next": "`02_DFire_entrenamiento_modelos.ipynb`.",
    },
    "02_DFire_entrenamiento_modelos.ipynb": {
        "role": "Entrena los modelos YOLO que se comparan en el TFM.",
        "use": "Para ver la configuración de entrenamiento o volver a entrenar un modelo.",
        "mode": "No entrena (`RUN_TRAINING=False`). Para entrenar hace falta una GPU.",
        "inputs": "Dataset preparado por el notebook 01 y `configs/model_registry.yaml`.",
        "outputs": "Una carpeta por entrenamiento en `artifacts/experiments/<id>/` con pesos, métricas y configuración.",
        "next": "`03_DFire_evaluacion_modelos.ipynb`.",
    },
    "03_DFire_evaluacion_modelos.ipynb": {
        "role": "Calcula las métricas de cada modelo entrenado sobre validación y analiza sus errores.",
        "use": "Para ver la comparación inicial de modelos (Tablas 1 y 2 de la memoria).",
        "mode": "No evalúa (`RUN_STANDARD_EVALUATION=False`, `RUN_ERROR_ANALYSIS=False`): muestra las evaluaciones guardadas.",
        "inputs": "Modelos entrenados en `artifacts/experiments/`.",
        "outputs": "Métricas y análisis de errores dentro de la carpeta de cada experimento.",
        "next": "`05_DFire_barrido_umbrales.ipynb`.",
    },
    "05_DFire_barrido_umbrales.ipynb": {
        "role": "Prueba distintos umbrales de confianza en tres modelos y mide cuántas falsas alarmas produce cada uno.",
        "use": "Para ver cómo cambian el recall y las falsas alarmas al mover el umbral.",
        "mode": "No recalcula (`RUN_SWEEP=False`): muestra el último barrido guardado.",
        "inputs": "Modelos de `artifacts/experiments/` y `configs/threshold_sweep.yaml`.",
        "outputs": "`artifacts/05_threshold_sweep/`, con tablas, figuras y ejemplos de errores.",
        "next": "`06_DFire_comparacion_resolucion.ipynb`.",
    },
    "06_DFire_comparacion_resolucion.ipynb": {
        "role": "Compara YOLOv8s entrenado a 640, 768 y 1024 px, evaluando cada uno a las tres resoluciones.",
        "use": "Para ver el estudio de resolución (Tablas 3 y 4 de la memoria).",
        "mode": "No recalcula (`RUN_COMPARISON=False`): muestra la última comparación guardada.",
        "inputs": "Los tres entrenamientos de YOLOv8s y `configs/resolution_comparison.yaml`.",
        "outputs": "`artifacts/06_resolution_comparison/`, con tablas, figuras y ejemplos.",
        "next": "`08_DFire_optimizacion_hiperparametros_YOLO26s.ipynb` y `09_DFire_seleccion_final_validacion.ipynb`.",
    },
    "07_DFire_pruebas_manual_inferencia.ipynb": {
        "role": "Herramienta opcional para probar un modelo con una imagen o un vídeo propios.",
        "use": "Para hacer una prueba rápida desde Jupyter. No interviene en ninguna decisión del TFM.",
        "mode": "No hace nada hasta indicar un modelo y un fichero.",
        "inputs": "Un modelo (por ejemplo, el final: `artifacts/14_final_model_freeze/final/weights/best.pt`) y una imagen o vídeo.",
        "outputs": "`artifacts/07_manual_inference/<id>/` con el resultado anotado y los parámetros usados.",
        "next": "Para usar el sistema completo, la aplicación web (`docker compose up --build`).",
    },
    "08_DFire_optimizacion_hiperparametros_YOLO26s.ipynb": {
        "role": "Prueba cuatro cambios de hiperparámetros sobre YOLO26s a 768 px con entrenamientos cortos de 50 épocas.",
        "use": "Para ver el ajuste de hiperparámetros (Tabla 6 de la memoria).",
        "mode": "No entrena ni evalúa (`RUN_TRAINING`, `RUN_EVALUATION` y `RUN_FINAL_TRAINING` a `False`).",
        "inputs": "Dataset preparado, entrenamiento de referencia de YOLO26s y `configs/yolo26s_hyperparameter_search.yaml`.",
        "outputs": "Un experimento por ensayo en `artifacts/experiments/`.",
        "next": "`09_DFire_seleccion_final_validacion.ipynb`.",
    },
    "09_DFire_seleccion_final_validacion.ipynb": {
        "role": "Elige el modelo final y sus umbrales usando solo validación.",
        "use": "Para ver por qué se eligió YOLO26s a 768 px con umbrales 0,36 (humo) y 0,16 (fuego) (Secciones 4.3 y 4.5 de la memoria).",
        "mode": "Muestra los resultados guardados (`RUN_SELECTION=False`). No usa test.",
        "inputs": "Predicciones de validación de YOLO26s 768→768 y YOLOv8s 768→640.",
        "outputs": "Resumen de la selección en `artifacts/12_final_validation_selection/`.",
        "next": "`10_DFire_comparacion_todos_modelos_1pct.ipynb`.",
    },
    "10_DFire_comparacion_todos_modelos_1pct.ipynb": {
        "role": "Comprueba la elección aplicando el mismo criterio del 1 % a todas las configuraciones entrenadas.",
        "use": "Para comprobar que la elección no dependía de comparar solo dos candidatos (Figura 5 de la memoria).",
        "mode": "Muestra los resultados guardados. Se recalculan con `tools/run_all_models_01pct_comparison.py`.",
        "inputs": "Predicciones de validación de las 14 configuraciones completas y de los cuatro ensayos de hiperparámetros.",
        "outputs": "Rankings y tablas por clase y tamaño.",
        "next": "Congelación del modelo (`tools/freeze_final_model.py`) y `11_DFire_evaluacion_final_test.ipynb`.",
    },
    "11_DFire_evaluacion_final_test.ipynb": {
        "role": "Muestra la única evaluación del modelo final sobre test.",
        "use": "Para consultar las cifras finales de la memoria (Sección 5).",
        "mode": "Solo lectura: no permite volver a evaluar, cambiar los umbrales ni elegir otro modelo.",
        "inputs": "Modelo congelado y umbrales humo 0,36 / fuego 0,16.",
        "outputs": "Tablas y figuras, copiadas en `results/final_test/`.",
        "next": "`12_DFire_interpretabilidad_modelo.ipynb`.",
    },
    "12_DFire_interpretabilidad_modelo.ipynb": {
        "role": "Analiza en qué zonas de la imagen se fija el modelo final para decidir.",
        "use": "Para ver el análisis de interpretabilidad (Sección 6 de la memoria).",
        "mode": "Muestra los resultados guardados (`RUN_ANALYSIS=False`). Usa validación, no test.",
        "inputs": "Modelo final y seis imágenes de validación.",
        "outputs": "Mapas, métricas y figuras, copiados en `results/interpretability/`.",
        "next": "Fin del proceso experimental.",
    },
}


def guide_markdown(name: str, values: dict[str, str]) -> str:
    return f"""## Guía de lectura y ejecución

| Aspecto | Descripción |
|---|---|
| **Papel en el proyecto** | {values['role']} |
| **Cuándo abrirlo** | {values['use']} |
| **Comportamiento por defecto** | {values['mode']} |
| **Entradas principales** | {values['inputs']} |
| **Salidas principales** | {values['outputs']} |
| **Continuación** | {values['next']} |

> **Para ejecutarlo:** usar el entorno Docker de notebooks (sección 4 del
> `README.md` principal). Volver a ejecutarlo requiere los resultados de los
> notebooks anteriores; ver `notebooks/README.md`.
"""


def replace_in_output(output, old: str, new: str) -> None:
    if "text" in output and isinstance(output["text"], str):
        output["text"] = output["text"].replace(old, new)
    data = output.get("data", {})
    for key, value in list(data.items()):
        if isinstance(value, str):
            data[key] = value.replace(old, new)
        elif isinstance(value, list):
            data[key] = [item.replace(old, new) if isinstance(item, str) else item for item in value]


def remove_obsolete_scene_references(notebook) -> None:
    replacements = (
        ("- El análisis perceptual completo entre escenas se mantiene en el notebook 04.\n", ""),
        ("La auditoría de independencia entre escenas permanece en el notebook 04.\n", ""),
        ('"near_duplicate_audit": "see_notebook_04"', '"near_duplicate_audit": "not_in_delivery_scope"'),
    )
    for cell in notebook.cells:
        for old, new in replacements:
            cell.source = cell.source.replace(old, new)
            if cell.cell_type == "code":
                for output in cell.get("outputs", []):
                    replace_in_output(output, old, new)


def document_notebook(path: Path) -> None:
    notebook = nbformat.read(path, as_version=4)
    remove_obsolete_scene_references(notebook)
    guide = nbformat.v4.new_markdown_cell(
        guide_markdown(path.name, GUIDES[path.name]),
        metadata={"tags": [GUIDE_TAG]},
    )
    existing = next(
        (
            index
            for index, cell in enumerate(notebook.cells)
            if GUIDE_TAG in cell.metadata.get("tags", [])
        ),
        None,
    )
    if existing is None:
        notebook.cells.insert(1, guide)
    else:
        notebook.cells[existing] = guide
    nbformat.validate(notebook)
    nbformat.write(notebook, path)


def main() -> None:
    existing = {path.name for path in NOTEBOOK_DIR.glob("*.ipynb")}
    expected = set(GUIDES)
    if existing != expected:
        missing = sorted(expected - existing)
        unexpected = sorted(existing - expected)
        raise SystemExit(f"Inventario inesperado. Faltan={missing}; sobran={unexpected}")
    for name in sorted(GUIDES):
        document_notebook(NOTEBOOK_DIR / name)
        print(f"Documentado: {name}")


if __name__ == "__main__":
    main()
