"""Umbrales por clase y revisión dirigida de falsos positivos en validación."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tfm_pipeline as pipeline
from tfm_thresholds import evaluate_threshold, load_predictions
from tools.run_threshold_sweep import get_prediction_cache


def fingerprint(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def load_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1 or config.get("split") != "val":
        raise ValueError("El análisis requiere schema_version=1 y split=val.")
    if set(config.get("profiles", {})) != {"sensitivity", "balanced"}:
        raise ValueError("Se requieren los perfiles sensitivity y balanced.")
    for field in ("smoke_thresholds", "fire_thresholds"):
        values = np.asarray(config[field], dtype=float)
        if not len(values) or np.any(np.diff(values) <= 0) or np.any(values < config["prediction_confidence"]):
            raise ValueError(f"Malla no válida: {field}")
    if not 0 < config["alarm_budget"] < 1 or not 0 <= config["recall_tolerance"] < 1:
        raise ValueError("Presupuesto de alarmas o tolerancia no válidos.")
    return config


def one_class_metrics(result: dict, class_name: str) -> dict[str, float | int]:
    return {name: result[f"{class_name}_{name}"] for name in ("tp", "fp", "fn", "precision", "recall", "f1")}


def build_grid(
    profile_key: str,
    payloads: list[dict],
    manifest: pd.DataFrame,
    config: dict,
    eval_config: dict,
) -> tuple[pd.DataFrame, dict[float, pd.DataFrame]]:
    records = {row["filename"]: row for row in manifest.to_dict("records")}
    thresholds = sorted(set(config["smoke_thresholds"]) | set(config["fire_thresholds"]))
    results: dict[float, dict] = {}
    image_tables: dict[float, pd.DataFrame] = {}
    for threshold in thresholds:
        result, images, _, _, _ = evaluate_threshold(
            payloads, records, threshold, eval_config, profile_key
        )
        results[threshold] = result
        image_tables[threshold] = images.set_index("filename").sort_index()

    rows = []
    for smoke_threshold in config["smoke_thresholds"]:
        smoke = one_class_metrics(results[smoke_threshold], "smoke")
        smoke_images = image_tables[smoke_threshold]
        for fire_threshold in config["fire_thresholds"]:
            fire = one_class_metrics(results[fire_threshold], "fire")
            fire_images = image_tables[fire_threshold]
            tp = int(smoke["tp"] + fire["tp"])
            fp = int(smoke["fp"] + fire["fp"])
            fn = int(smoke["fn"] + fire["fn"])
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            negative = smoke_images.gt_count == 0
            negative_alarm = (
                (smoke_images.loc[negative, "smoke_fp"] > 0)
                | (fire_images.loc[negative, "fire_fp"] > 0)
            )
            rows.append({
                "profile": profile_key,
                "smoke_threshold": smoke_threshold,
                "fire_threshold": fire_threshold,
                **{f"smoke_{key}": value for key, value in smoke.items()},
                **{f"fire_{key}": value for key, value in fire.items()},
                "micro_tp": tp, "micro_fp": fp, "micro_fn": fn,
                "micro_precision": precision, "micro_recall": recall, "micro_f1": f1,
                "macro_recall": (float(smoke["recall"]) + float(fire["recall"])) / 2,
                "minimum_class_recall": min(float(smoke["recall"]), float(fire["recall"])),
                "negative_images": int(negative.sum()),
                "negative_images_with_alarm": int(negative_alarm.sum()),
                "negative_alarm_rate": float(negative_alarm.mean()),
                "positive_image_fp_boxes": int(smoke_images.loc[~negative, "smoke_fp"].sum()
                                               + fire_images.loc[~negative, "fire_fp"].sum()),
                "negative_image_fp_boxes": int(smoke_images.loc[negative, "smoke_fp"].sum()
                                               + fire_images.loc[negative, "fire_fp"].sum()),
            })
    return pd.DataFrame(rows), image_tables


def pick_row(table: pd.DataFrame, sort_columns: list[str], ascending: list[bool]) -> pd.Series:
    if table.empty:
        raise AssertionError("No existe ningún punto que cumpla las restricciones.")
    return table.sort_values(sort_columns, ascending=ascending, kind="stable").iloc[0]


def recommendations(grid: pd.DataFrame, profiles: dict, config: dict) -> pd.DataFrame:
    rows = []
    for profile_key, table in grid.groupby("profile", sort=False):
        profile = profiles[profile_key]
        fixed = table[
            np.isclose(table.smoke_threshold, profile["baseline_smoke_threshold"])
            & np.isclose(table.fire_threshold, profile["baseline_fire_threshold"])
        ]
        if len(fixed) != 1:
            raise AssertionError(f"No se encontró el punto original de {profile_key}.")
        fixed = fixed.iloc[0]
        feasible = table[table.negative_images_with_alarm <= config["alarm_budget"] * table.negative_images + 1e-12]
        guarded = feasible[
            (feasible.smoke_recall >= fixed.smoke_recall - config["recall_tolerance"])
            & (feasible.fire_recall >= fixed.fire_recall - config["recall_tolerance"])
        ]
        fire_guarded = feasible[feasible.fire_recall >= fixed.fire_recall - config["recall_tolerance"]]
        selections = {
            "original": fixed,
            "max_recall_budget": pick_row(
                feasible, ["macro_recall", "minimum_class_recall", "micro_f1", "micro_precision"],
                [False, False, False, False],
            ),
            "max_f1_budget": pick_row(
                feasible, ["micro_f1", "macro_recall", "micro_precision"], [False, False, False]
            ),
            "precision_recall_guardrail": pick_row(
                guarded, ["micro_precision", "micro_f1", "macro_recall"], [False, False, False]
            ),
            "f1_fire_guardrail": pick_row(
                fire_guarded, ["micro_f1", "micro_precision", "macro_recall"], [False, False, False]
            ),
        }
        for scenario, selected in selections.items():
            rows.append({**selected.to_dict(), "scenario": scenario,
                         "recall_tolerance": config["recall_tolerance"],
                         "alarm_budget": config["alarm_budget"]})
    return pd.DataFrame(rows)


def fp_review(
    payloads: list[dict], manifest: pd.DataFrame, config: dict, eval_config: dict, output: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    records = {row["filename"]: row for row in manifest.to_dict("records")}
    _, images, _, _, detections = evaluate_threshold(
        payloads, records, 0.16, eval_config, "sensitivity", details=True
    )
    false_positives = detections[detections.status == "fp"].copy()
    false_positives["area_fraction"] = (
        (false_positives.x2 - false_positives.x1) * (false_positives.y2 - false_positives.y1)
    )
    false_positives["image_scope"] = np.where(false_positives.is_negative_image, "negative", "positive")
    false_positives["confidence_band"] = pd.cut(
        false_positives.confidence, [0.16, 0.25, 0.50, 0.75, 1.01],
        labels=["0.16-0.25", "0.25-0.50", "0.50-0.75", "0.75-1.00"], include_lowest=True,
    ).astype(str)
    summary = (
        false_positives.groupby(["image_scope", "class_name", "fp_reason"], dropna=False)
        .agg(false_positive_boxes=("filename", "size"), images=("filename", "nunique"),
             mean_confidence=("confidence", "mean"), max_confidence=("confidence", "max"))
        .reset_index()
    )

    positive = false_positives[~false_positives.is_negative_image].copy()
    reason_order = ["no_overlap", "localization", "wrong_class", "weak_overlap", "duplicate"]
    candidates = []
    quota = max(2, int(np.ceil(config["gallery_images"] / max(1, len(reason_order)))))
    for reason in reason_order:
        subset = positive[positive.fp_reason == reason].sort_values(
            ["confidence", "area_fraction"], ascending=[False, False]
        )
        candidates.append(subset.drop_duplicates("filename").head(quota))
    selected = pd.concat(candidates, ignore_index=True).drop_duplicates("filename")
    if len(selected) < config["gallery_images"]:
        used = set(selected.filename)
        fill = positive[~positive.filename.isin(used)].sort_values(
            ["confidence", "area_fraction"], ascending=[False, False]
        ).drop_duplicates("filename").head(config["gallery_images"] - len(selected))
        selected = pd.concat([selected, fill], ignore_index=True)
    selected = selected.head(config["gallery_images"]).copy()
    selected.insert(0, "review_id", [f"FP{i:02d}" for i in range(1, len(selected) + 1)])
    selected["manual_category"] = ""
    selected["observation"] = ""
    selected["review_status"] = "pending"

    payload_by_filename = {item["filename"]: item for item in payloads}
    figure_dir = output / "figures"
    figure_dir.mkdir(exist_ok=True)
    font = ImageFont.load_default()
    panels = []
    for row in selected.itertuples():
        source = Image.open(records[row.filename]["image_path"]).convert("RGB")
        draw = ImageDraw.Draw(source)
        width, height = source.size
        truth = payload_by_filename[row.filename]["ground_truth"]
        for box in truth:
            x1, y1, x2, y2 = box["xyxy"]
            draw.rectangle((x1 * width, y1 * height, x2 * width, y2 * height), outline="#FFFFFF", width=max(2, width // 400))
        image_detections = false_positives[false_positives.filename == row.filename]
        for box in image_detections.itertuples():
            draw.rectangle((box.x1 * width, box.y1 * height, box.x2 * width, box.y2 * height),
                           outline="#FF6B35", width=max(3, width // 300))
        target_width = 440
        ratio = target_width / source.width
        resized = source.resize((target_width, max(1, round(source.height * ratio))))
        header = Image.new("RGB", (target_width, 52), "#202428")
        header_draw = ImageDraw.Draw(header)
        header_draw.text((8, 6), f"{row.review_id} · {row.filename}", fill="white", font=font)
        header_draw.text((8, 28), f"FP {row.class_name} {row.confidence:.2f} · {row.fp_reason}", fill="#FFB08F", font=font)
        panel = Image.new("RGB", (target_width, 52 + resized.height), "white")
        panel.paste(header, (0, 0)); panel.paste(resized, (0, 52)); panels.append(panel)

    per_page = 9
    for page_index in range(0, len(panels), per_page):
        page_panels = panels[page_index:page_index + per_page]
        cell_height = max(panel.height for panel in page_panels)
        canvas = Image.new("RGB", (440 * 3, cell_height * 3), "white")
        for index, panel in enumerate(page_panels):
            canvas.paste(panel, ((index % 3) * 440, (index // 3) * cell_height))
        canvas.save(figure_dir / f"03_fp_review_page_{page_index // per_page + 1}.png", quality=92)
    return summary, selected


def build_figures(grid: pd.DataFrame, recommendations_table: pd.DataFrame, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    figures = output / "figures"; figures.mkdir(exist_ok=True)
    for profile, table in grid.groupby("profile", sort=False):
        pivot = table.pivot(index="smoke_threshold", columns="fire_threshold", values="micro_precision")
        fig, ax = plt.subplots(figsize=(9, 7))
        image = ax.imshow(pivot.values, origin="lower", aspect="auto", cmap="YlGnBu", vmin=.60, vmax=.85)
        ax.set_xticks(range(len(pivot.columns))[::2], [f"{x:.2f}" for x in pivot.columns[::2]], rotation=45)
        ax.set_yticks(range(len(pivot.index))[::2], [f"{x:.2f}" for x in pivot.index[::2]])
        ax.set(xlabel="Umbral de fuego", ylabel="Umbral de humo", title=f"Precisión micro · {profile}")
        fig.colorbar(image, ax=ax, format=PercentFormatter(1), label="Precisión")
        fig.tight_layout(); fig.savefig(figures / f"01_precision_grid_{profile}.png", dpi=170); plt.close(fig)

    scenarios = ["original", "max_recall_budget", "max_f1_budget", "precision_recall_guardrail"]
    selected = recommendations_table[recommendations_table.scenario.isin(scenarios)].copy()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    for ax, (profile, table) in zip(axes, selected.groupby("profile", sort=False)):
        x = np.arange(len(table)); width = .18
        for offset, (column, label) in enumerate([
            ("smoke_recall", "Recall humo"), ("fire_recall", "Recall fuego"),
            ("micro_precision", "Precisión"), ("micro_f1", "F1"),
        ]):
            ax.bar(x + (offset - 1.5) * width, table[column], width, label=label)
        ax.set_xticks(x, [name.replace("_", "\n") for name in table.scenario], fontsize=8)
        ax.set_title(profile); ax.grid(axis="y", color="#E1E4E6"); ax.yaxis.set_major_formatter(PercentFormatter(1))
    axes[0].set_ylim(0, 1); axes[1].legend(frameon=False, loc="lower left")
    fig.suptitle("Perfiles con umbrales independientes por clase")
    fig.tight_layout(); fig.savefig(figures / "02_recommendation_comparison.png", dpi=170); plt.close(fig)


def write_summary(output: Path, recommendations_table: pd.DataFrame, fp_summary: pd.DataFrame, config: dict) -> None:
    lines = [
        "# Umbrales por clase y revisión de falsos positivos", "",
        "Análisis offline sobre validación. No se ha consultado test.", "",
        "## Perfiles conservados", "",
        "- Sensibilidad: YOLO26s 768 -> 768, humo 0,16 y fuego 0,16.",
        "- Equilibrado: YOLOv8s 768 -> 640, humo 0,30 y fuego 0,30.", "",
        "## Alternativas encontradas", "",
        "| Perfil | Escenario | Humo | Fuego | Recall humo | Recall fuego | Precisión | F1 | Alarmas |", 
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in recommendations_table.itertuples():
        lines.append(
            f"| {row.profile} | {row.scenario} | {row.smoke_threshold:.2f} | {row.fire_threshold:.2f} | "
            f"{row.smoke_recall:.2%} | {row.fire_recall:.2%} | {row.micro_precision:.2%} | "
            f"{row.micro_f1:.2%} | {int(row.negative_images_with_alarm)}/{int(row.negative_images)} |"
        )
    lines += ["", "## Revisión geométrica de falsos positivos YOLO26s sensible", "",
              "Las categorías geométricas describen la relación con las anotaciones; la interpretación visual se completa en `fp_review_candidates.csv`.", ""]
    for row in fp_summary.sort_values("false_positive_boxes", ascending=False).itertuples():
        lines.append(
            f"- {row.image_scope} · {row.class_name} · {row.fp_reason}: {row.false_positive_boxes} cajas en {row.images} imágenes."
        )
    lines += ["", "## Criterio", "",
              f"El escenario `precision_recall_guardrail` maximiza precisión sin permitir que el recall de ninguna clase caiga más de {config['recall_tolerance']:.0%} frente al perfil original. ",
              "Las categorías nubes, niebla, luces, reflejos o anotación posiblemente incompleta requieren inspección visual y no se infieren solo por geometría.", ""]
    (output / "RESUMEN_UMBRALES_POR_CLASE.md").write_text("\n".join(lines), encoding="utf-8")


def run(config_path: Path) -> Path:
    config = load_config(config_path)
    contract = pipeline.validate_prepared_dataset(ROOT, config["dataset_version"])
    manifest = pipeline.rebased_manifest(contract)
    manifest = manifest[manifest.split == "val"].copy().reset_index(drop=True)
    run_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + fingerprint(config)[:8]
    parent = ROOT / "artifacts" / "10_class_threshold_fp_review" / "validation"
    output = parent / run_id; output.mkdir(parents=True)
    shutil.copy2(config_path, output / "config.yaml")
    code_dir = output / "code"; code_dir.mkdir(); shutil.copy2(Path(__file__), code_dir / Path(__file__).name)
    summary = {
        "schema_version": 1, "status": "running", "run_id": run_id,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "split": "val",
        "dataset_version": config["dataset_version"], "test_inference_executed": False,
        "manifest_sha256": pipeline.sha256_file(contract["manifest_path"]), "config": config,
    }
    pipeline.write_json_atomic(output / "run_summary.json", summary)
    grids, caches, experiments = [], {}, {}
    try:
        for profile_key, profile in config["profiles"].items():
            experiment = pipeline.resolve_experiment(experiment_id=profile["experiment_id"], root=ROOT)
            if experiment["status"] != "complete" or experiment["dataset_version"] != config["dataset_version"]:
                raise ValueError(f"Experimento no válido: {profile['experiment_id']}")
            experiments[profile_key] = experiment
            eval_config = {**config, "imgsz": profile["imgsz"]}
            cache = get_prediction_cache(experiment, contract, eval_config, offline=True)
            caches[profile_key] = cache
            payloads = load_predictions(ROOT / cache["predictions_rel"], manifest, config["prediction_confidence"])
            grid, _ = build_grid(profile_key, payloads, manifest, config, eval_config)
            grids.append(grid)
        grid = pd.concat(grids, ignore_index=True)
        recommendation_table = recommendations(grid, config["profiles"], config)

        sensitive_config = {**config, "imgsz": config["profiles"]["sensitivity"]["imgsz"]}
        sensitive_payloads = load_predictions(
            ROOT / caches["sensitivity"]["predictions_rel"], manifest, config["prediction_confidence"]
        )
        fp_summary, fp_candidates = fp_review(sensitive_payloads, manifest, config, sensitive_config, output)
        grid.to_csv(output / "class_threshold_grid.csv", index=False)
        recommendation_table.to_csv(output / "profile_recommendations.csv", index=False)
        fp_summary.to_csv(output / "fp_geometric_summary.csv", index=False)
        fp_candidates.to_csv(output / "fp_review_candidates.csv", index=False)
        build_figures(grid, recommendation_table, output)
        write_summary(output, recommendation_table, fp_summary, config)

        expected_rows = len(config["profiles"]) * len(config["smoke_thresholds"]) * len(config["fire_thresholds"])
        if len(grid) != expected_rows or set(grid.negative_images) != {783}:
            raise AssertionError("La cuadrícula no tiene la cobertura esperada.")
        if len(fp_candidates) != config["gallery_images"] or fp_candidates.filename.duplicated().any():
            raise AssertionError("Muestra visual incompleta o duplicada.")
        summary.update(
            status="complete", completed_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
            grid_points=len(grid), review_images=len(fp_candidates), inputs=caches,
            output_hashes={str(path.relative_to(output)): pipeline.sha256_file(path)
                           for path in output.rglob("*") if path.is_file() and path.name != "run_summary.json"},
        )
        pipeline.write_json_atomic(output / "run_summary.json", summary)
        pipeline.write_json_atomic(parent / "latest.json", {
            "run_id": run_id, "run_rel": pipeline.project_relative(output, ROOT),
            "summary_sha256": pipeline.sha256_file(output / "run_summary.json"),
        })
        print(f"Análisis completo: {output}")
        print(recommendation_table[["profile", "scenario", "smoke_threshold", "fire_threshold", "smoke_recall", "fire_recall", "micro_precision", "micro_f1", "negative_images_with_alarm"]].to_string(index=False))
        return output
    except BaseException as exc:
        summary.update(status="incomplete", error=f"{type(exc).__name__}: {exc}")
        pipeline.write_json_atomic(output / "run_summary.json", summary)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "class_threshold_and_fp_review.yaml")
    args = parser.parse_args(); run(args.config)


if __name__ == "__main__":
    main()
