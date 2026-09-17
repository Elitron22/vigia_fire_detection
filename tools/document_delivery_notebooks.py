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
        "role": "Núcleo reproducible: prepara la única versión del dataset usada por los experimentos.",
        "use": "Abrir primero para entender las particiones, correcciones y controles de integridad.",
        "mode": "Consulta por defecto (`RUN_PREPARATION=False`); la preparación completa solo se activa al crear deliberadamente otra versión.",
        "inputs": "Dataset oficial en `data/D-Fire`.",
        "outputs": "`artifacts/datasets/dfire_seed42_val10_v1`, con manifiesto, YAML y etiquetas preparadas.",
        "next": "`02_DFire_entrenamiento_modelos.ipynb`.",
    },
    "02_DFire_entrenamiento_modelos.ipynb": {
        "role": "Núcleo reproducible: entrena y registra los checkpoints comparados.",
        "use": "Abrir para conocer la configuración común, la cola de modelos y la trazabilidad de cada entrenamiento.",
        "mode": "No entrena por defecto; requiere activar `RUN_TRAINING=True` y disponer de GPU para una reproducción razonable.",
        "inputs": "Dataset preparado por 01 y `configs/model_registry.yaml`.",
        "outputs": "`artifacts/experiments/<experiment_id>` con configuración, métricas y pesos.",
        "next": "`03_DFire_evaluacion_modelos.ipynb`.",
    },
    "03_DFire_evaluacion_modelos.ipynb": {
        "role": "Núcleo reproducible: evalúa checkpoints sin mezclar validación y test.",
        "use": "Abrir para entender las métricas estándar y el diagnóstico inicial de falsos positivos y negativos.",
        "mode": "Trabaja en validación; el selector de test solo se reserva para la configuración final congelada.",
        "inputs": "Un experimento completo de `artifacts/experiments` y el dataset preparado.",
        "outputs": "Métricas y diagnósticos bajo el directorio de evaluación del experimento.",
        "next": "`05_DFire_barrido_umbrales.ipynb`.",
    },
    "05_DFire_barrido_umbrales.ipynb": {
        "role": "Núcleo analítico: compara umbrales por clase y el coste en falsas alarmas.",
        "use": "Abrir para seguir cómo se pasa de las métricas generales a puntos operativos candidatos.",
        "mode": "Solo consulta por defecto (`RUN_SWEEP=False`); reutiliza la última ejecución verificada.",
        "inputs": "Predicciones de validación de los modelos finalistas y `configs/threshold_sweep.yaml`.",
        "outputs": "`artifacts/05_threshold_sweep/validation/<run_id>` con tablas, figuras y revisión de errores.",
        "next": "`06_DFire_comparacion_resolucion.ipynb`.",
    },
    "06_DFire_comparacion_resolucion.ipynb": {
        "role": "Núcleo analítico: separa el efecto de la resolución de entrenamiento y de inferencia.",
        "use": "Abrir para revisar la cuadrícula 3×3, el recall por tamaño y el coste computacional.",
        "mode": "Solo consulta por defecto (`RUN_COMPARISON=False`).",
        "inputs": "Entrenamientos YOLOv8s a 640, 768 y 1024 px y sus cachés de validación.",
        "outputs": "`artifacts/06_resolution_comparison/validation/<run_id>`.",
        "next": "`08_DFire_optimizacion_hiperparametros_YOLO26s.ipynb` o directamente 09 para la selección final.",
    },
    "07_DFire_pruebas_manual_inferencia.ipynb": {
        "role": "Utilidad opcional: demuestra inferencia sobre una imagen o un vídeo externo.",
        "use": "Abrir únicamente para una prueba manual fuera de la aplicación web; no sustenta la selección del modelo.",
        "mode": "Interactivo y no destructivo; no entrena ni consulta test.",
        "inputs": "Un checkpoint registrado y un archivo elegido por la persona usuaria.",
        "outputs": "`artifacts/07_manual_inference/<run_id>` con el medio anotado y su configuración.",
        "next": "Para la demostración principal se recomienda `docker compose up --build`.",
    },
    "08_DFire_optimizacion_hiperparametros_YOLO26s.ipynb": {
        "role": "Experimento complementario: documenta una búsqueda manual acotada que no sustituyó al entrenamiento final controlado.",
        "use": "Abrir para comprobar qué alternativas se probaron y por qué no se presentan como una optimización exhaustiva.",
        "mode": "Todas las ejecuciones costosas están desactivadas por defecto; una repetición completa requiere GPU.",
        "inputs": "Baseline YOLO26s, dataset preparado y configuración de los ensayos.",
        "outputs": "Experimentos cortos y evaluación comparativa; la receta final congelada sigue siendo `controlled_768`.",
        "next": "`09_DFire_seleccion_final_validacion.ipynb`.",
    },
    "09_DFire_seleccion_final_validacion.ipynb": {
        "role": "Núcleo decisional: fija checkpoint y umbrales antes de abrir test.",
        "use": "Abrir para entender la comparación final entre YOLO26s 768→768 y YOLOv8s 768→640.",
        "mode": "Cuaderno de resultados sobre validación; no ejecuta inferencia en test.",
        "inputs": "Artefactos verificados de los dos candidatos y el presupuesto máximo del 1 % de negativas con alarma.",
        "outputs": "Resumen de selección, métricas por clase/tamaño y punto operativo congelado.",
        "next": "`10_DFire_comparacion_todos_modelos_1pct.ipynb` y congelación del modelo.",
    },
    "10_DFire_comparacion_todos_modelos_1pct.ipynb": {
        "role": "Comprobación de cobertura: aplica el mismo criterio final al inventario completo de modelos.",
        "use": "Abrir para verificar que la decisión no depende solo de los dos finalistas inicialmente comparados.",
        "mode": "Solo consulta; utiliza validación y mantiene test bloqueado.",
        "inputs": "Cachés verificadas de 14 configuraciones completas y cuatro ensayos cortos separados.",
        "outputs": "Rankings de sensibilidad y F1, métricas por clase/tamaño y comparación de puntos operativos.",
        "next": "`11_DFire_evaluacion_final_test.ipynb` tras congelar el checkpoint.",
    },
    "11_DFire_evaluacion_final_test.ipynb": {
        "role": "Resultado final: presenta la única evaluación del modelo congelado sobre test.",
        "use": "Abrir para consultar las cifras finales que se citan en la memoria.",
        "mode": "Solo lectura: deliberadamente no permite reevaluar, buscar umbrales ni cambiar de modelo.",
        "inputs": "Checkpoint congelado y umbrales humo 0,36 / fuego 0,16.",
        "outputs": "`results/final_test` con tablas, figuras, resumen y manifiesto de ejecución.",
        "next": "No usar test para nuevas decisiones; cualquier análisis posterior vuelve a validación o a datos externos.",
    },
    "12_DFire_interpretabilidad_modelo.ipynb": {
        "role": "Análisis final: estudia qué regiones sostienen predicciones seleccionadas del YOLO26s.",
        "use": "Abrir para revisar Eigen-CAM, oclusión, fidelidad y las limitaciones de las explicaciones.",
        "mode": "Solo consulta sobre validación; no cambia el modelo ni sus umbrales.",
        "inputs": "Checkpoint final y seis casos dirigidos del split de validación.",
        "outputs": "`results/interpretability` con mapas, métricas y resumen.",
        "next": "Fin del flujo experimental documentado.",
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

> **Entorno:** abrir desde la raíz del repositorio o definir
> `TFM_PROJECT_ROOT`. La reproducción completa de entrenamientos necesita el
> dataset D-Fire y una GPU; la lectura de resultados guardados no.
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
