"""Resume y valida los cuatro entrenamientos base sin ejecutar inferencia nueva."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import tfm_pipeline as pipeline


OUTPUT_ROOT = (
    PROJECT_ROOT
    / "artifacts"
    / "04_model_comparison"
    / "reports"
    / "20260831_post_training_review"
)


def collect_models() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    experiment_rows = pipeline.list_experiments(PROJECT_ROOT).to_dict("records")
    model_rows: list[dict[str, object]] = []
    inputs: list[dict[str, object]] = []
    for experiment in experiment_rows:
        results_path = Path(experiment["training_run_dir"]) / "results.csv"
        if not results_path.exists():
            continue
        results = pd.read_csv(results_path)
        results.columns = [column.strip() for column in results.columns]
        metric = "metrics/mAP50-95(B)"
        best = results.loc[results[metric].idxmax()]
        final = results.iloc[-1]
        checkpoint = Path(experiment["best_model"])
        model = YOLO(str(checkpoint))
        class_names = {int(key): str(value).lower() for key, value in model.names.items()}
        if class_names != pipeline.CLASS_NAMES:
            raise ValueError(f"Clases inesperadas en {checkpoint}: {class_names}")
        parameter_count = sum(parameter.numel() for parameter in model.model.parameters())
        end_to_end = bool(getattr(model.model.model[-1], "end2end", False))
        standard_summary = Path(experiment["experiment_root"]) / "evaluation" / "test_evaluation_summary.json"
        legacy_summary = experiment.get("legacy_evaluation_summary")
        test_evaluated = standard_summary.exists() or (
            isinstance(legacy_summary, str) and Path(legacy_summary).exists()
        )
        model_rows.append(
            {
                "model": experiment["model_key"],
                "family": experiment["model_family"],
                "scale": experiment["model_scale"],
                "experiment_id": experiment["experiment_id"],
                "best_val_map5095": float(best[metric]),
                "best_val_map50": float(best["metrics/mAP50(B)"]),
                "best_val_precision": float(best["metrics/precision(B)"]),
                "best_val_recall": float(best["metrics/recall(B)"]),
                "best_epoch": int(best["epoch"]),
                "epochs_run": int(final["epoch"]),
                "epochs_after_best": int(final["epoch"] - best["epoch"]),
                "final_val_map5095": float(final[metric]),
                "training_hours": float(final["time"]) / 3600,
                "checkpoint_mb": checkpoint.stat().st_size / 1024**2,
                "parameters_m": parameter_count / 1_000_000,
                "end_to_end": end_to_end,
                "test_evaluated": test_evaluated,
                "dataset_version": experiment["dataset_version"],
                "seed": int(experiment["seed"]),
                "profile": experiment["profile"],
            }
        )
        inputs.append(
            {
                "model": experiment["model_key"],
                "experiment_id": experiment["experiment_id"],
                "results_csv": pipeline.project_relative(results_path, PROJECT_ROOT),
                "experiment_json": pipeline.project_relative(
                    Path(experiment["experiment_root"]) / "experiment.json", PROJECT_ROOT
                ),
                "best_checkpoint": pipeline.project_relative(checkpoint, PROJECT_ROOT),
            }
        )
    expected = {"yolov8n", "yolov8s", "yolo26n", "yolo26s"}
    found = {str(row["model"]) for row in model_rows}
    if found != expected:
        raise ValueError(f"Se esperaban {sorted(expected)} y se encontraron {sorted(found)}")
    return sorted(model_rows, key=lambda row: float(row["best_val_map5095"]), reverse=True), inputs


def source_definitions() -> list[dict[str, object]]:
    return [
        {
            "id": "comparison",
            "label": "Comparación derivada de los cuatro results.csv y sus best.pt",
            "path": "artifacts/04_model_comparison/reports/20260831_post_training_review/comparison_snapshot.json",
            "query": {
                "engine": "sqlite",
                "sql": "SELECT json_extract(value, '$.model') AS model, json_extract(value, '$.family') AS family, json_extract(value, '$.scale') AS scale, json_extract(value, '$.best_val_map5095') AS best_val_map5095, json_extract(value, '$.best_val_map50') AS best_val_map50, json_extract(value, '$.best_val_precision') AS best_val_precision, json_extract(value, '$.best_val_recall') AS best_val_recall, json_extract(value, '$.best_epoch') AS best_epoch, json_extract(value, '$.epochs_run') AS epochs_run, json_extract(value, '$.training_hours') AS training_hours, json_extract(value, '$.checkpoint_mb') AS checkpoint_mb, json_extract(value, '$.parameters_m') AS parameters_m, json_extract(value, '$.end_to_end') AS end_to_end, json_extract(value, '$.test_evaluated') AS test_evaluated FROM json_each(?) ORDER BY best_val_map5095 DESC",
                "description": "Proyección reproducible sobre snapshot.datasets.models, derivado sin nueva inferencia de los resultados de entrenamiento y checkpoints guardados.",
                "filters": [
                    "Modelos: yolov8n, yolov8s, yolo26n y yolo26s",
                    "Dataset: dfire_seed42_val10_v1",
                    "Semilla: 42",
                    "Perfil: controlled",
                ],
                "metric_definitions": {
                    "best_val_map5095": "Máximo mAP de validación promediado entre IoU 0,50 y 0,95 durante el entrenamiento.",
                    "checkpoint_mb": "Tamaño del archivo best.pt en MiB.",
                    "training_hours": "Tiempo acumulado registrado en results.csv hasta la última época ejecutada.",
                },
            },
        },
        {
            "id": "protocol",
            "label": "Registro del perfil controlado",
            "path": "TFM/configs/model_registry.yaml",
        },
        {
            "id": "evaluation_notebook",
            "label": "Notebook de evaluación estándar y análisis operativo",
            "path": "TFM/notebooks/03_DFire_evaluacion_modelos.ipynb",
        },
    ]


def build_artifact(models: list[dict[str, object]]) -> dict[str, object]:
    generated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    title = "D-Fire · Qué hacer tras entrenar los cuatro modelos"
    sources = source_definitions()
    return {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": title,
            "generatedAt": generated_at,
            "sources": sources,
            "charts": [
                {
                    "id": "validation_map",
                    "title": "Mejor mAP50–95 de validación",
                    "description": "Máximo observado durante cada entrenamiento; mismo dataset, semilla y perfil base.",
                    "showDescription": True,
                    "type": "bar",
                    "dataset": "models",
                    "sourceId": "comparison",
                    "encodings": {
                        "x": {"field": "model", "type": "nominal", "label": "Modelo"},
                        "y": {
                            "field": "best_val_map5095",
                            "type": "quantitative",
                            "label": "mAP50–95 de validación",
                            "format": "percent",
                        },
                    },
                    "valueFormat": "percent",
                },
                {
                    "id": "checkpoint_size",
                    "title": "Tamaño del checkpoint best.pt",
                    "description": "Tamaño en MiB; no sustituye a una medición de latencia o memoria en Raspberry Pi.",
                    "showDescription": True,
                    "type": "bar",
                    "dataset": "models",
                    "sourceId": "comparison",
                    "encodings": {
                        "x": {"field": "model", "type": "nominal", "label": "Modelo"},
                        "y": {
                            "field": "checkpoint_mb",
                            "type": "quantitative",
                            "label": "Tamaño de best.pt (MiB)",
                            "format": "number",
                        },
                    },
                    "valueFormat": "number",
                },
            ],
            "tables": [
                {
                    "id": "model_table",
                    "title": "Entrenamientos base completados",
                    "description": "Validación global del mejor checkpoint; aún faltan métricas por clase y evaluación operativa homogénea.",
                    "showDescription": True,
                    "dataset": "models",
                    "sourceId": "comparison",
                    "defaultSort": {"field": "best_val_map5095", "direction": "desc"},
                    "columns": [
                        {"field": "model", "label": "Modelo"},
                        {"field": "scale", "label": "Escala"},
                        {"field": "best_val_map5095", "label": "mAP50–95 val", "format": "percent"},
                        {"field": "best_val_map50", "label": "mAP50 val", "format": "percent"},
                        {"field": "best_val_precision", "label": "Precisión val", "format": "percent"},
                        {"field": "best_val_recall", "label": "Recall val", "format": "percent"},
                        {"field": "best_epoch", "label": "Mejor época", "format": "number"},
                        {"field": "epochs_run", "label": "Épocas", "format": "number"},
                        {"field": "training_hours", "label": "Entrenamiento (h)", "format": "number"},
                        {"field": "checkpoint_mb", "label": "best.pt (MiB)", "format": "number"},
                        {"field": "parameters_m", "label": "Parámetros (M)", "format": "number"},
                        {"field": "end_to_end", "label": "End-to-end"},
                        {"field": "test_evaluated", "label": "Test evaluado"},
                    ],
                }
            ],
            "blocks": [
                {"id": "title", "type": "markdown", "body": f"# {title}"},
                {
                    "id": "summary",
                    "type": "markdown",
                    "body": (
                        "## Recomendación técnica\n\n"
                        "**La fase de entrenamiento base está cerrada, pero todavía no elegiría un ganador ni ejecutaría el test de los tres modelos nuevos.** "
                        "Primero completaría en validación las métricas por clase, la tasa de falsas alarmas sobre imágenes negativas y la latencia con el modo de inferencia nativo de cada arquitectura. Después elegiría un finalista small y uno nano, comprobaría estabilidad con semillas adicionales y congelaría el umbral operativo antes del test final.\n\n"
                        "YOLOv8s lidera el mAP50–95 de validación con 46,520 %, seguido de YOLO26s con 46,158 %. Entre los nano, YOLO26n alcanza 44,911 % y YOLOv8n 44,683 %. Las diferencias de 0,362 y 0,228 puntos porcentuales son demasiado pequeñas para tratarlas como concluyentes con una sola semilla."
                    ),
                    "sourceId": "comparison",
                },
                {
                    "id": "performance_intro",
                    "type": "markdown",
                    "body": (
                        "## Validación sitúa a los modelos por pares, no define un ganador absoluto\n\n"
                        "El gráfico compara el máximo mAP50–95 global observado durante cada entrenamiento. Es útil para cribar arquitecturas, pero no informa todavía de humo frente a fuego, falsas alarmas ni variación entre semillas."
                    ),
                },
                {"id": "performance_chart", "type": "chart", "chartId": "validation_map"},
                {
                    "id": "performance_interpretation",
                    "type": "markdown",
                    "body": (
                        "YOLOv8s supera a YOLO26s en 0,362 puntos de mAP50–95 y también muestra 1,146 puntos más de precisión y 0,826 de recall global. "
                        "En nano, YOLO26n gana 0,228 puntos de mAP50–95, pero YOLOv8n conserva 1,109 puntos más de mAP50, 1,007 de precisión y 0,917 de recall. Por eso la elección nano depende del error operativo y del dispositivo, no de una sola columna."
                    ),
                    "sourceId": "comparison",
                },
                {
                    "id": "footprint_intro",
                    "type": "markdown",
                    "body": (
                        "## YOLO26 reduce el tamaño, pero aún no se ha medido su ventaja de despliegue\n\n"
                        "El tamaño de `best.pt` es una primera aproximación al coste de almacenamiento. No permite inferir FPS, latencia, RAM ni temperatura en Raspberry Pi; esas variables requieren exportar y medir."
                    ),
                },
                {"id": "footprint_chart", "type": "chart", "chartId": "checkpoint_size"},
                {
                    "id": "footprint_interpretation",
                    "type": "markdown",
                    "body": (
                        "YOLO26s ocupa 19,36 MiB frente a 21,46 MiB de YOLOv8s y tiene aproximadamente 9,95 M frente a 11,14 M de parámetros. YOLO26n ocupa 5,12 MiB frente a 5,94 MiB de YOLOv8n. "
                        "Sin embargo, con este perfil YOLO26s tardó 3,52 h y YOLO26n 2,44 h, más que sus equivalentes YOLOv8; el tiempo de entrenamiento no equivale a velocidad de inferencia."
                    ),
                    "sourceId": "comparison",
                },
                {"id": "model_table_block", "type": "table", "tableId": "model_table"},
                {
                    "id": "scope",
                    "type": "markdown",
                    "body": (
                        "## Alcance y definición de la comparación\n\n"
                        "Se revisaron cuatro ejecuciones completas: YOLOv8n, YOLOv8s, YOLO26n y YOLO26s. Todas están registradas con `dfire_seed42_val10_v1`, semilla 42 y perfil `controlled`: 640 px, batch 16, AdamW, `lr0=0.001`, máximo 100 épocas y `patience=20`. "
                        "Los tres experimentos nuevos contienen el mismo SHA-256 del manifiesto; el descriptor legado de YOLOv8s no guarda ese campo, aunque la auditoría previa lo vinculó a la misma versión.\n\n"
                        "mAP50–95 es la media de Average Precision entre IoU 0,50 y 0,95. Los valores de precisión y recall de esta tabla son los registrados por Ultralytics durante validación, no los obtenidos con confianza fija 0,25."
                    ),
                    "sourceId": "protocol",
                },
                {
                    "id": "design",
                    "type": "markdown",
                    "body": (
                        "## Diseño experimental verificado\n\n"
                        "Los resultados se extrajeron de cada `results.csv`; para cada modelo se seleccionó la fila con mayor mAP50–95 de validación. El tamaño y el número de parámetros se calcularon desde `best.pt`, y se verificó el mapeo `0=smoke`, `1=fire`. "
                        "YOLOv8 utiliza cabeza con NMS y los checkpoints YOLO26 están marcados como end-to-end. Esta diferencia debe conservarse y documentarse en el benchmark de despliegue; el parámetro NMS IoU no es aplicable al modo nativo de YOLO26."
                    ),
                    "sourceId": "comparison",
                },
                {
                    "id": "limitations",
                    "type": "markdown",
                    "body": (
                        "## La incertidumbre impide interpretar décimas como superioridad\n\n"
                        "Solo hay una semilla por arquitectura. No existen todavía métricas por clase con `best.pt`, intervalos de variabilidad, tasa de falsas alarmas homogénea ni benchmark de inferencia comparable. YOLO26s llegó a la época 100 y su mejor época fue la 82; no alcanzó las 20 épocas de paciencia, pero su métrica final había retrocedido. Extender solo ese modelo rompería la comparación controlada y no está justificado por esta curva.\n\n"
                        "Solo YOLOv8s tiene test guardado. Ese resultado previo no debe utilizarse para ajustar los modelos nuevos ni para elegir su umbral."
                    ),
                    "sourceId": "comparison",
                },
                {
                    "id": "next_steps",
                    "type": "markdown",
                    "body": (
                        "## Hoja de ruta inmediata\n\n"
                        "1. Añadir al notebook de evaluación un modo `split=val` y registrar explícitamente `end2end`/NMS.\n"
                        "2. Evaluar los cuatro `best.pt` en validación: métricas globales y por clase, velocidad, matriz de confusión y análisis de negativos.\n"
                        "3. Seleccionar el umbral operativo exclusivamente en validación, priorizando recall y limitando la tasa de imágenes negativas con alarma.\n"
                        "4. Elegir un candidato small entre YOLOv8s/YOLO26s y uno nano entre YOLOv8n/YOLO26n. Medir exportación, latencia, memoria y tamaño en el entorno objetivo.\n"
                        "5. Repetir los finalistas con al menos dos semillas adicionales antes de atribuir importancia a diferencias inferiores a un punto.\n"
                        "6. Solo entonces realizar una optimización acotada —si aporta una pregunta clara— y congelar pesos, umbral y modo de inferencia.\n"
                        "7. Ejecutar el test una sola vez para cada configuración final y generar la comparación y el análisis de errores definitivos."
                    ),
                    "sourceId": "evaluation_notebook",
                },
                {
                    "id": "questions",
                    "type": "markdown",
                    "body": (
                        "## Preguntas que decidirán el modelo final\n\n"
                        "¿Qué recall mínimo se exige para fuego y humo? ¿Qué porcentaje de imágenes negativas con alarma es aceptable? ¿La Raspberry Pi concreta puede mantener la latencia objetivo con YOLO26s? "
                        "¿Se comparará el modo end-to-end nativo de YOLO26 o también su cabeza tradicional con NMS? Estas decisiones deben fijarse antes de mirar el test de los modelos nuevos."
                    ),
                },
            ],
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": {"models": models},
        },
        "sources": sources,
    }


def main() -> None:
    models, inputs = collect_models()
    artifact = build_artifact(models)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "comparison_snapshot.json").write_text(
        json.dumps(models, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUTPUT_ROOT / "artifact.json").write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUTPUT_ROOT / "report_notes.json").write_text(
        json.dumps(
            {
                "audience": "technical",
                "question": "Decidir el siguiente paso tras completar los cuatro entrenamientos base.",
                "inputs": inputs,
                "required_structure_mapping": {
                    "title": "title",
                    "technical_summary": "summary",
                    "key_findings": [
                        "performance_intro",
                        "performance_chart",
                        "footprint_intro",
                        "footprint_chart",
                        "model_table_block",
                    ],
                    "scope_data_definitions": "scope",
                    "methodology_and_design": "design",
                    "limitations_robustness": "limitations",
                    "recommended_next_steps": "next_steps",
                    "further_questions": "questions",
                },
                "chart_map": [
                    {
                        "section": "Validación",
                        "question": "¿Qué modelo alcanzó mayor mAP50-95?",
                        "family": "Comparison & Ranking",
                        "type": "bar",
                        "fields": ["model", "best_val_map5095"],
                        "supported_claim": "Los modelos small lideran, con diferencias pequeñas dentro de cada escala.",
                        "palette_policy": "single-root preferred",
                    },
                    {
                        "section": "Huella",
                        "question": "¿Qué tamaño tiene el checkpoint de cada modelo?",
                        "family": "Comparison & Ranking",
                        "type": "bar",
                        "fields": ["model", "checkpoint_mb"],
                        "supported_claim": "YOLO26 produce checkpoints menores dentro de cada escala.",
                        "palette_policy": "single-root preferred",
                        "repeat_reason": "Misma forma por tratarse de una segunda comparación categórica de magnitud con unidades distintas.",
                    },
                ],
                "validation_status": "Cálculos reproducidos desde results.csv; falta inferencia homogénea por clase y múltiples semillas.",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(OUTPUT_ROOT)


if __name__ == "__main__":
    main()
