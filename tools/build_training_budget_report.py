"""Construye el diagnóstico reproducible de duración e hiperparámetros del baseline."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = (
    PROJECT_ROOT
    / "artifacts"
    / "01_dfire_yolov8s_baseline"
    / "runs"
    / "yolov8s_dfire_baseline_seed42-2"
)
RESULTS_CSV = RUN_ROOT / "results.csv"
ARGS_YAML = RUN_ROOT / "args.yaml"
REGISTRY_YAML = PROJECT_ROOT / "configs" / "model_registry.yaml"
OUTPUT_ROOT = (
    PROJECT_ROOT
    / "artifacts"
    / "01_dfire_yolov8s_baseline"
    / "reports"
    / "20260830_training_budget_review"
)


def load_rows() -> list[dict[str, float | int]]:
    with RESULTS_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        raw_rows = list(csv.DictReader(handle))

    rows: list[dict[str, float | int]] = []
    running_best = float("-inf")
    for raw in raw_rows:
        clean = {key.strip(): value for key, value in raw.items()}
        map5095 = float(clean["metrics/mAP50-95(B)"])
        running_best = max(running_best, map5095)
        rows.append(
            {
                "epoch": int(clean["epoch"]),
                "elapsed_seconds": float(clean["time"]),
                "map5095": map5095,
                "map50": float(clean["metrics/mAP50(B)"]),
                "precision": float(clean["metrics/precision(B)"]),
                "recall": float(clean["metrics/recall(B)"]),
                "train_box": float(clean["train/box_loss"]),
                "val_box": float(clean["val/box_loss"]),
                "running_best": running_best,
            }
        )
    if not rows:
        raise RuntimeError(f"No hay filas en {RESULTS_CSV}")
    return rows


def summarize(rows: list[dict[str, float | int]]) -> dict[str, float | int]:
    best = max(rows, key=lambda row: float(row["map5095"]))
    final = rows[-1]
    best_value = float(best["map5095"])

    threshold_epochs: dict[int, int] = {}
    for percentage in (90, 95, 99):
        threshold = best_value * percentage / 100
        threshold_epochs[percentage] = int(
            next(row["epoch"] for row in rows if float(row["running_best"]) >= threshold)
        )

    epoch_50 = next(row for row in rows if row["epoch"] == 50)
    min_val_box = min(rows, key=lambda row: float(row["val_box"]))
    return {
        "epochs_completed": len(rows),
        "best_epoch": int(best["epoch"]),
        "best_map5095": best_value,
        "final_epoch": int(final["epoch"]),
        "final_map5095": float(final["map5095"]),
        "epochs_without_improvement": int(final["epoch"]) - int(best["epoch"]),
        "epoch_90pct_best": threshold_epochs[90],
        "epoch_95pct_best": threshold_epochs[95],
        "epoch_99pct_best": threshold_epochs[99],
        "running_best_epoch_50": float(epoch_50["running_best"]),
        "gain_after_epoch_50": best_value - float(epoch_50["running_best"]),
        "gain_after_epoch_50_percentage_points": 100
        * (best_value - float(epoch_50["running_best"])),
        "train_box_drop_50_to_final_fraction": (
            float(epoch_50["train_box"]) - float(final["train_box"])
        )
        / float(epoch_50["train_box"]),
        "val_box_change_50_to_final_fraction": (
            float(final["val_box"]) - float(epoch_50["val_box"])
        )
        / float(epoch_50["val_box"]),
        "min_val_box_epoch": int(min_val_box["epoch"]),
        "min_val_box": float(min_val_box["val_box"]),
        "elapsed_hours": float(final["elapsed_seconds"]) / 3600,
    }


def milestone_rows(
    rows: list[dict[str, float | int]], epochs: tuple[int, ...] = (25, 50, 77, 97)
) -> list[dict[str, float | int | str]]:
    result = []
    for epoch in epochs:
        row = next(item for item in rows if item["epoch"] == epoch)
        result.append(
            {
                "epoch": epoch,
                "budget_label": f"Época {epoch}",
                "map5095_at_epoch": float(row["map5095"]),
                "best_so_far": float(row["running_best"]),
                "train_box": float(row["train_box"]),
                "val_box": float(row["val_box"]),
                "elapsed_hours": float(row["elapsed_seconds"]) / 3600,
            }
        )
    return result


def source_definitions() -> list[dict[str, object]]:
    return [
        {
            "id": "training_trend",
            "label": "Historial completo del entrenamiento YOLOv8s",
            "path": "runs/yolov8s_dfire_baseline_seed42-2/results.csv",
            "query": {
                "engine": "sqlite",
                "sql": "SELECT CAST(json_extract(value, '$.epoch') AS INTEGER) AS epoch, json_extract(value, '$.map5095') AS map5095, json_extract(value, '$.map50') AS map50, json_extract(value, '$.train_box') AS train_box, json_extract(value, '$.val_box') AS val_box, json_extract(value, '$.running_best') AS running_best FROM json_each(?) ORDER BY epoch",
                "description": "Proyección reproducible de las 97 filas de results.csv incluidas en snapshot.datasets.trend.",
                "filters": ["Ejecución yolov8s_dfire_baseline_seed42-2", "Métrica de selección: mAP50-95 de validación"],
                "metric_definitions": {
                    "map5095": "mAP de detección promediado entre IoU 0,50 y 0,95 sobre validación.",
                    "running_best": "Máximo mAP50-95 observado desde la época 1 hasta la época indicada.",
                },
            },
        },
        {
            "id": "training_milestones",
            "label": "Hitos de presupuesto calculados desde el historial completo",
            "path": "runs/yolov8s_dfire_baseline_seed42-2/results.csv",
            "query": {
                "engine": "sqlite",
                "sql": "SELECT CAST(json_extract(value, '$.epoch') AS INTEGER) AS epoch, json_extract(value, '$.budget_label') AS budget_label, json_extract(value, '$.map5095_at_epoch') AS map5095_at_epoch, json_extract(value, '$.best_so_far') AS best_so_far, json_extract(value, '$.train_box') AS train_box, json_extract(value, '$.val_box') AS val_box, json_extract(value, '$.elapsed_hours') AS elapsed_hours FROM json_each(?) ORDER BY epoch",
                "description": "Selección determinista de las épocas 25, 50, 77 y 97 incluida en snapshot.datasets.milestones.",
                "filters": ["Hitos: épocas 25, 50, 77 y 97"],
            },
        },
        {
            "id": "training_args",
            "label": "Argumentos registrados por Ultralytics",
            "path": "runs/yolov8s_dfire_baseline_seed42-2/args.yaml",
        },
        {
            "id": "controlled_profile",
            "label": "Perfil controlado para la comparación de arquitecturas",
            "path": "TFM/configs/model_registry.yaml",
        },
    ]


def build_artifact(
    rows: list[dict[str, float | int]],
    milestones: list[dict[str, float | int | str]],
    summary: dict[str, float | int],
) -> dict[str, object]:
    generated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    sources = source_definitions()
    title = "D-Fire · Duración e hiperparámetros tras el primer YOLOv8s"
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
                    "id": "validation_curve",
                    "title": "mAP50–95 de validación por época",
                    "description": "97 épocas observadas; la curva se aproxima a su máximo antes de terminar.",
                    "showDescription": True,
                    "type": "line",
                    "dataset": "trend",
                    "sourceId": "training_trend",
                    "encodings": {
                        "x": {"field": "epoch", "type": "quantitative", "label": "Época"},
                        "y": {
                            "field": "map5095",
                            "type": "quantitative",
                            "label": "mAP50–95 de validación",
                            "format": "percent",
                        },
                    },
                    "valueFormat": "percent",
                },
                {
                    "id": "budget_comparison",
                    "title": "Mejor mAP50–95 alcanzado con cada presupuesto",
                    "description": "Comparación acumulada en cuatro hitos de la misma ejecución; no son entrenamientos independientes.",
                    "showDescription": True,
                    "type": "bar",
                    "dataset": "milestones",
                    "sourceId": "training_milestones",
                    "encodings": {
                        "x": {"field": "budget_label", "type": "nominal", "label": "Presupuesto máximo observado"},
                        "y": {
                            "field": "best_so_far",
                            "type": "quantitative",
                            "label": "Mejor mAP50–95 acumulado",
                            "format": "percent",
                        },
                    },
                    "valueFormat": "percent",
                },
            ],
            "tables": [
                {
                    "id": "milestone_table",
                    "title": "Hitos de entrenamiento",
                    "description": "Valores exactos en épocas seleccionadas de la ejecución YOLOv8s.",
                    "showDescription": True,
                    "dataset": "milestones",
                    "sourceId": "training_milestones",
                    "defaultSort": {"field": "epoch", "direction": "asc"},
                    "columns": [
                        {"field": "epoch", "label": "Época", "format": "number"},
                        {"field": "map5095_at_epoch", "label": "mAP50–95 en la época", "format": "percent"},
                        {"field": "best_so_far", "label": "Mejor acumulado", "format": "percent"},
                        {"field": "train_box", "label": "Train box loss", "format": "number"},
                        {"field": "val_box", "label": "Val box loss", "format": "number"},
                        {"field": "elapsed_hours", "label": "Tiempo acumulado (h)", "format": "number"},
                    ],
                }
            ],
            "blocks": [
                {"id": "title", "type": "markdown", "body": f"# {title}"},
                {
                    "id": "technical_summary",
                    "type": "markdown",
                    "body": (
                        "## Recomendación técnica\n\n"
                        "**No cambiaría todavía los hiperparámetros del perfil controlado y mantendría `epochs=100` con `patience=20` para YOLOv8n y YOLO26n.** "
                        "Las 100 épocas son un límite máximo, no una obligación: la parada temprana ya detuvo este baseline en la época 97 y conservó `best.pt` de la época 77.\n\n"
                        "El YOLOv8s alcanzó el 99 % de su mejor mAP50–95 en la época 49. Desde el mejor valor acumulado disponible en la época 50 hasta el máximo de la 77 solo ganó 0,371 puntos porcentuales. "
                        "Esto desaconseja aumentar el presupuesto por encima de 100, pero una sola arquitectura y una sola semilla no justifican imponer 50, 77 u 80 épocas a todos los modelos."
                    ),
                    "sourceId": "training_trend",
                },
                {
                    "id": "curve_intro",
                    "type": "markdown",
                    "body": (
                        "## La validación se saturó, aunque la época exacta no es transferible\n\n"
                        "La curva muestra el mAP50–95 medido en validación después de cada época. Debe leerse como evidencia de rendimientos decrecientes para este YOLOv8s: no como una predicción de cuándo convergerán YOLOv8n o YOLO26n."
                    ),
                },
                {"id": "curve", "type": "chart", "chartId": "validation_curve"},
                {
                    "id": "curve_interpretation",
                    "type": "markdown",
                    "body": (
                        "El máximo fue 46,52 % en la época 77. El final quedó en 46,285 %, tras exactamente 20 épocas sin mejorar, por lo que `patience=20` funcionó como estaba previsto. "
                        "Entre las épocas 50 y 97, la pérdida de cajas de entrenamiento bajó un 15,39 %, mientras la de validación aumentó un 0,55 %: señal de saturación y sobreajuste leve, no de un colapso."
                    ),
                    "sourceId": "training_trend",
                },
                {
                    "id": "budget_intro",
                    "type": "markdown",
                    "body": (
                        "## Más épocas aportaron una mejora pequeña después de la época 50\n\n"
                        "La barra utiliza el mejor valor alcanzado hasta cada hito, evitando que la oscilación de una época concreta oculte el progreso acumulado. Los hitos pertenecen al mismo entrenamiento y no estiman variabilidad entre ejecuciones."
                    ),
                },
                {"id": "budget", "type": "chart", "chartId": "budget_comparison"},
                {"id": "milestones", "type": "table", "tableId": "milestone_table"},
                {
                    "id": "budget_interpretation",
                    "type": "markdown",
                    "body": (
                        "A la época 50 ya se había obtenido 46,149 % como mejor valor acumulado, frente a 46,52 % en la 77. El ahorro de terminar siempre en 50 épocas sería importante, pero podría perder el pequeño margen final y rompería la igualdad de presupuesto entre arquitecturas. "
                        "Para la comparación principal conviene conservar la parada temprana; un perfil de exploración corto sería un experimento distinto."
                    ),
                    "sourceId": "training_milestones",
                },
                {
                    "id": "scope",
                    "type": "markdown",
                    "body": (
                        "## Alcance, datos y definición de la métrica\n\n"
                        "Se ha revisado únicamente la ejecución `yolov8s_dfire_baseline_seed42-2`: 97 épocas registradas sobre el split de entrenamiento/validación ya preparado. "
                        "La métrica de selección es mAP50–95 de validación, es decir, la precisión media de detección promediada entre umbrales IoU 0,50 y 0,95. El test no se utiliza para decidir épocas ni hiperparámetros.\n\n"
                        "La comparación relevante no es la métrica de una época aislada, sino el máximo de validación alcanzado bajo un presupuesto y el checkpoint `best.pt` que Ultralytics conserva."
                    ),
                    "sourceId": "training_trend",
                },
                {
                    "id": "model_specification",
                    "type": "markdown",
                    "body": (
                        "## Especificación que debe permanecer controlada\n\n"
                        "El baseline empleó YOLOv8s, imágenes de 640 píxeles, batch 16, AdamW, `lr0=0.001`, scheduler coseno, AMP, `seed=42`, modo determinista, máximo de 100 épocas y `patience=20`. "
                        "Mantener esos valores para YOLOv8n y YOLO26n permite atribuir las diferencias iniciales principalmente a la arquitectura. Cambiarlos ahora mezclaría el efecto del modelo con el del entrenamiento."
                    ),
                    "sourceId": "training_args",
                },
                {
                    "id": "methodology",
                    "type": "markdown",
                    "body": (
                        "## Método de diagnóstico\n\n"
                        "Se leyó `results.csv` completo, se identificó el máximo de mAP50–95 de validación y se calculó la primera época que alcanzó el 90 %, 95 % y 99 % de ese máximo. "
                        "También se compararon las pérdidas de cajas de entrenamiento y validación entre la época 50 y el final. Los cálculos están reproducidos por `tools/build_training_budget_report.py` y la instantánea numérica está incrustada en este informe."
                    ),
                    "sourceId": "training_trend",
                },
                {
                    "id": "limitations",
                    "type": "markdown",
                    "body": (
                        "## Lo que esta curva no permite concluir\n\n"
                        "Solo existe una ejecución, una semilla y una arquitectura en este diagnóstico; no hay intervalo de incertidumbre. La época 77 puede depender del ruido de validación y no es un hiperparámetro óptimo universal. "
                        "Tampoco puede afirmarse que otro modelo necesite menos épocas solo por tener menos parámetros.\n\n"
                        "El 99 % del mejor valor es un umbral descriptivo, no un criterio de parada usado durante el entrenamiento. La decisión sobre el modelo final debe incluir métricas por clase, falsas alarmas, velocidad y coste de despliegue, no solo mAP50–95."
                    ),
                },
                {
                    "id": "next_steps",
                    "type": "markdown",
                    "body": (
                        "## Siguiente decisión experimental\n\n"
                        "1. Entrenar YOLOv8n y YOLO26n con el mismo perfil controlado: `epochs=100`, `patience=20`, batch 16, 640 px, AdamW y `lr0=0.001`.\n"
                        "2. Comparar cada `best.pt` en la misma validación y, solo una vez cerrado el protocolo, en el test.\n"
                        "3. No aumentar las épocas: el baseline ya presenta rendimientos decrecientes.\n"
                        "4. Seleccionar uno o dos finalistas antes de optimizar. En ellos probar primero una búsqueda acotada en validación —por ejemplo, tasa de aprendizaje, resolución y aumentos— manteniendo una variable o un diseño de búsqueda documentado.\n"
                        "5. Si los tres modelos vuelven a saturarse antes de 70–80 épocas, entonces sí considerar para ensayos futuros `epochs=80–90` y `patience=15`, sin reescribir retrospectivamente esta comparación."
                    ),
                    "sourceId": "controlled_profile",
                },
                {
                    "id": "questions",
                    "type": "markdown",
                    "body": (
                        "## Preguntas que resolverán los siguientes modelos\n\n"
                        "¿Los modelos nano alcanzan su mejor validación antes o después que YOLOv8s? ¿El pequeño margen posterior a la época 50 se repite entre semillas? "
                        "¿Una mayor resolución mejora objetos pequeños lo suficiente para compensar su coste en Raspberry Pi? Estas respuestas determinarán si merece la pena reducir el presupuesto o abrir una optimización de hiperparámetros."
                    ),
                },
            ],
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": {"trend": rows, "milestones": milestones},
        },
        "sources": sources,
    }


def main() -> None:
    for required in (RESULTS_CSV, ARGS_YAML, REGISTRY_YAML):
        if not required.exists():
            raise FileNotFoundError(required)

    rows = load_rows()
    summary = summarize(rows)
    milestones = milestone_rows(rows)
    artifact = build_artifact(rows, milestones, summary)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "analysis_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUTPUT_ROOT / "artifact.json").write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUTPUT_ROOT / "report_notes.json").write_text(
        json.dumps(
            {
                "audience": "technical",
                "question": "Decidir si cambiar hiperparámetros o presupuesto antes de comparar arquitecturas.",
                "required_structure_mapping": {
                    "title": "title",
                    "technical_summary": "technical_summary",
                    "key_findings": ["curve_intro", "curve", "budget_intro", "budget", "milestones"],
                    "scope_and_definitions": "scope",
                    "model_specification": "model_specification",
                    "methodology": "methodology",
                    "limitations_and_robustness": "limitations",
                    "recommended_next_steps": "next_steps",
                    "further_questions": "questions",
                },
                "chart_map": [
                    {
                        "section": "Saturación de validación",
                        "question": "¿Cuándo dejó de mejorar materialmente el modelo?",
                        "family": "Trend",
                        "type": "line",
                        "fields": ["epoch", "map5095"],
                        "takeaway": "La mayor parte del rendimiento se alcanzó antes de la época 50.",
                        "palette_policy": "single-root preferred",
                    },
                    {
                        "section": "Rendimientos decrecientes",
                        "question": "¿Qué rendimiento permitió cada presupuesto observado?",
                        "family": "Comparison & Ranking",
                        "type": "bar",
                        "fields": ["budget_label", "best_so_far"],
                        "takeaway": "El margen adicional tras la época 50 fue pequeño.",
                        "palette_policy": "single-root preferred",
                    },
                ],
                "robustness": "Cálculos deterministas sobre las 97 filas; limitación principal: una sola semilla y arquitectura.",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(OUTPUT_ROOT)


if __name__ == "__main__":
    main()
