"""Genera interpretabilidad del YOLO26s congelado usando solo validación."""
from __future__ import annotations

import argparse
import datetime as dt
import gc
import json
from pathlib import Path
import shutil
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
import torch
import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tfm_pipeline as pipeline
from tfm_interpretability import (
    deletion_curve, heatmap_mass_inside, mask_tiles, minmax,
    multiscale_eigencam, occlusion_sensitivity, pointing_game, predict_arrays,
    result_predictions, select_cases, target_score,
)


def load_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1 or config.get("split") != "val":
        raise ValueError("La interpretabilidad solo admite schema_version=1 y split=val")
    if config.get("test_locked") is not True:
        raise ValueError("Test debe permanecer bloqueado")
    if config["model"]["experiment_id"] != "yolo26s_dfire_seed42_20260912T165304Z":
        raise ValueError("Se esperaba el YOLO26s final")
    if int(config["model"]["imgsz"]) != 768:
        raise ValueError("La resolución final debe ser 768")
    point = config["operating_point"]
    if float(point["smoke_threshold"]) != .36 or float(point["fire_threshold"]) != .16:
        raise ValueError("Los umbrales deben ser 0,36/0,16")
    cases = [tuple(case) for case in config["sample_design"]["cases"]]
    expected = {(mode, cls) for mode in ("true_positive", "negative_false_alarm", "borderline_false_negative")
                for cls in ("smoke", "fire")}
    if set(cases) != expected or len(cases) != 6:
        raise ValueError("El diseño debe cubrir acierto, falsa alarma y omisión para ambas clases")
    return config


def capture_eigencam(model, image, target_box, class_id, config, device):
    layers = [int(value) for value in config["eigen_cam"]["feature_layers"]]
    activations, hooks = {}, []
    for layer_index in layers:
        def hook(_, __, output, index=layer_index):
            if not torch.is_tensor(output):
                raise TypeError(f"La capa {index} no produjo un tensor espacial")
            activations[index] = output.detach().float().cpu().numpy()
        hooks.append(model.model.model[layer_index].register_forward_hook(hook))
    try:
        result = predict_arrays(model, [image], imgsz=int(config["model"]["imgsz"]),
                                conf=float(config["occlusion"]["inference_confidence"]),
                                batch=1, device=device)[0]
    finally:
        for handle in hooks: handle.remove()
    score = target_score(result_predictions(result), class_id, target_box,
                         float(config["occlusion"]["target_match_iou"]))
    combined, layer_maps = multiscale_eigencam(activations, image.shape[:2], int(config["model"]["imgsz"]))
    return combined, layer_maps, float(score)


def add_box(axis, box, width, height, *, color, label, linestyle="-"):
    x1, y1, x2, y2 = box
    axis.add_patch(Rectangle((x1*width, y1*height), (x2-x1)*width, (y2-y1)*height,
                             fill=False, edgecolor=color, linewidth=2.2, linestyle=linestyle))
    axis.text(x1*width + 3, max(14, y1*height + 14), label, color=color, fontsize=9,
              bbox={"facecolor": "#15181B", "alpha": .72, "pad": 1.5, "edgecolor": "none"})


def plot_case(image, row, target_box, gt_box, eigen_map, occlusion_map, top_masked,
              output: Path, layer_shapes: dict[int, tuple[int, ...]]):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]
    positive_occlusion = minmax(np.maximum(occlusion_map, 0))
    positive_occlusion = cv2.resize(positive_occlusion, (width, height), interpolation=cv2.INTER_NEAREST)
    masked_rgb = cv2.cvtColor(top_masked, cv2.COLOR_BGR2RGB)
    labels = {"true_positive": "Acierto", "negative_false_alarm": "Falsa alarma",
              "borderline_false_negative": "Omisión limítrofe"}
    class_label = "humo" if row.class_name == "smoke" else "fuego"
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.8))
    for axis in axes: axis.axis("off")
    axes[0].imshow(rgb); axes[0].set_title("Imagen y objetivo", loc="left")
    if gt_box is not None: add_box(axes[0], gt_box, width, height, color="#F2F2F2", label="GT", linestyle="--")
    add_box(axes[0], target_box, width, height, color="#00C2FF", label=f"Pred. {class_label}")
    axes[1].imshow(rgb); axes[1].imshow(eigen_map, cmap="inferno", alpha=.48, vmin=0, vmax=1)
    axes[1].set_title("Eigen-CAM multiescala", loc="left")
    add_box(axes[1], target_box, width, height, color="#00C2FF", label="Objetivo")
    axes[2].imshow(rgb); axes[2].imshow(positive_occlusion, cmap="magma", alpha=.55, vmin=0, vmax=1)
    axes[2].set_title("Sensibilidad por oclusión", loc="left")
    add_box(axes[2], target_box, width, height, color="#00C2FF", label="Objetivo")
    axes[3].imshow(masked_rgb); axes[3].set_title("Región más influyente oculta", loc="left")
    add_box(axes[3], target_box, width, height, color="#00C2FF", label="Objetivo")
    shapes = ", ".join(f"L{layer}: {shape[-2]}×{shape[-1]}" for layer, shape in layer_shapes.items())
    fig.suptitle(f"{labels[row.case]} de {class_label} · {row.filename} · confianza {row.fresh_target_confidence:.3f}",
                 x=.045, ha="left", fontsize=16)
    fig.text(.045, .015, f"Capas: {shapes}. Azul: predicción objetivo; GT discontinua. "
             "Eigen-CAM es activación no específica de clase; oclusión mide caída relativa de la confianza.",
             fontsize=9, color="#454B50")
    fig.subplots_adjust(left=.02, right=.99, top=.84, bottom=.09, wspace=.05)
    target = output / f"{int(row.order):02d}_{row.case}_{row.class_name}_{Path(row.filename).stem}.png"
    fig.savefig(target, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return target


def build_summary_figures(output: Path, curves: pd.DataFrame, metrics: pd.DataFrame,
                          case_figures: list[Path]) -> list[Path]:
    figures = output / "figures"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "white", "savefig.facecolor": "white"})
    aggregate = (curves.groupby(["filename", "strategy", "fraction"], as_index=False)
                 .score_retained.mean().groupby(["strategy", "fraction"]).score_retained
                 .agg(["mean", "std"]).reset_index())
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    styles = {"important": ("#D95F35", "Regiones más importantes", "o"),
              "random": ("#65727C", "Regiones aleatorias", "s")}
    for strategy, (color, label, marker) in styles.items():
        subset = aggregate[aggregate.strategy == strategy]
        ax.plot(subset.fraction, subset["mean"], color=color, marker=marker, linewidth=2, label=label)
        lower = np.maximum(0, subset["mean"] - subset["std"].fillna(0))
        upper = subset["mean"] + subset["std"].fillna(0)
        ax.fill_between(subset.fraction, lower, upper, color=color, alpha=.13)
    upper_limit = max(1.08, float((aggregate["mean"] + aggregate["std"].fillna(0)).max()) * 1.06)
    ax.set(
        xlabel="Fracción de regiones ocultas",
        ylabel="Confianza objetivo retenida",
        xlim=(0, 1),
        ylim=(0, upper_limit),
        title="Prueba de eliminación sobre seis casos de validación",
    )
    ax.xaxis.set_major_formatter(PercentFormatter(1)); ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(axis="y", color="#DCE1E4"); ax.legend(frameon=False)
    fig.text(.12, .01, "Media ± desviación entre seis casos dirigidos; menor confianza al ocultar regiones importantes indica mayor fidelidad.", fontsize=9)
    fig.tight_layout(rect=(0,.04,1,1)); path = figures / "07_deletion_curve.png"
    fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig)

    labels = [f"{row.class_name[:1].upper()} · {row.case.replace('_',' ')}" for row in metrics.itertuples()]
    x = np.arange(len(metrics)); width=.36
    fig, ax = plt.subplots(figsize=(11,5.6))
    top_drop = 1 - metrics.top25_score_retained.to_numpy()
    random_drop = 1 - metrics.random25_score_retained.to_numpy()
    ax.bar(x-width/2, top_drop, width, color="#D95F35", label="Regiones importantes")
    ax.bar(x+width/2, random_drop, width, color="#65727C", label="Aleatorias")
    ax.axhline(0, color="#202428", linewidth=.8)
    ax.set_xticks(x, labels, rotation=18, ha="right")
    ax.set_ylabel("Caída relativa de confianza al ocultar 25 %")
    ax.yaxis.set_major_formatter(PercentFormatter(1)); ax.grid(axis="y", color="#DCE1E4")
    ax.set_title("Comprobación local de fidelidad por caso", loc="left", fontsize=15); ax.legend(frameon=False)
    fig.tight_layout(); path2 = figures / "08_faithfulness_by_case.png"
    fig.savefig(path2, dpi=180, bbox_inches="tight"); plt.close(fig)

    from PIL import Image, ImageDraw
    thumbnails=[]
    for path in case_figures:
        picture=Image.open(path).convert("RGB"); picture.thumbnail((720,220)); thumbnails.append(picture.copy()); picture.close()
    canvas=Image.new("RGB",(740,240*len(thumbnails)),"white"); draw=ImageDraw.Draw(canvas)
    for index,picture in enumerate(thumbnails):
        canvas.paste(picture,(10,index*240+20)); draw.text((10,index*240+3),f"Caso {index+1}",fill="black")
    contact=figures/"09_interpretability_contact_sheet.png"; canvas.save(contact); canvas.close()
    return [path, path2, contact]


def markdown_table(frame: pd.DataFrame) -> str:
    def cell(value):
        if pd.isna(value): return "—"
        if isinstance(value,(float,np.floating)): return f"{float(value):.4f}"
        return str(value).replace("|","\\|").replace("\n"," ")
    headers=list(map(str,frame.columns))
    lines=["| "+" | ".join(headers)+" |","|"+"|".join("---" for _ in headers)+"|"]
    lines.extend("| "+" | ".join(cell(value) for value in row)+" |" for row in frame.itertuples(index=False,name=None))
    return "\n".join(lines)


def write_summary(output: Path, selection: pd.DataFrame, metrics: pd.DataFrame,
                  curves: pd.DataFrame, config: dict) -> Path:
    better = int(metrics.top25_better_than_random.sum())
    median_drop = float((1 - metrics.top25_score_retained).median())
    random_drop = float((1 - metrics.random25_score_retained).median())
    pointing = int(metrics.occlusion_pointing_game.sum())
    eigen_pointing = int(metrics.eigencam_pointing_game.sum())
    text = f"""# Interpretabilidad del YOLO26s final

## tl;dr

Se explicaron seis casos dirigidos del conjunto de **validación**: acierto,
falsa alarma negativa y omisión limítrofe para humo y fuego. Al ocultar el 25 %
de las regiones con mayor sensibilidad, la caída mediana de confianza fue
{median_drop:.1%}, frente a {random_drop:.1%} al ocultar regiones aleatorias.
La estrategia dirigida redujo más la confianza en {better}/6 casos. El máximo
del mapa de oclusión cayó dentro de la caja objetivo en {pointing}/6 casos y el
de Eigen-CAM en {eigen_pointing}/6.

## Método

- Modelo congelado: **YOLO26s 768→768**, hash `{config['model']['weights_sha256']}`.
- Punto operativo: humo 0,36 y fuego 0,16; no se modifican pesos ni umbrales.
- Eigen-CAM multiescala: primera componente de las activaciones de las capas
  {config['eigen_cam']['feature_layers']}, normalizadas, deshecho el letterbox y promediadas.
- Oclusión: rejilla {config['occlusion']['grid_size']}×{config['occlusion']['grid_size']};
  cada región se sustituye por su versión desenfocada y se mide la caída relativa
  de la misma detección (clase y solapamiento IoU≥0,30).
- Fidelidad: se ocultan progresivamente las regiones ordenadas por importancia y
  se comparan con {config['occlusion']['random_repeats']} selecciones aleatorias.

## Diseño de casos

La selección es dirigida y determinista. Cubre tipos de comportamiento, pero no
es una muestra aleatoria ni permite estimar la frecuencia de las explicaciones.
Las omisiones escogidas son limítrofes: existía una predicción coincidente a
confianza baja, pero quedaba por debajo del umbral operativo.

{markdown_table(selection[['order','case','class_name','filename','cached_target_confidence','operating_threshold']])}

## Fidelidad por caso

{markdown_table(metrics[['order','case','class_name','filename','fresh_target_confidence','max_single_occlusion_drop',
          'top25_score_retained','random25_score_retained','occlusion_mass_in_target',
          'occlusion_pointing_game','eigencam_mass_in_target','eigencam_pointing_game']])}

## Interpretación responsable

Los mapas muestran asociaciones locales del modelo, no una explicación causal del
incendio ni una garantía de robustez. Eigen-CAM es **no específico de clase**:
indica actividad multiescala, no prueba que una zona cause la etiqueta humo o
fuego. La oclusión sí es específica para una detección, pero depende del tamaño
de rejilla, del desenfoque elegido y puede introducir imágenes fuera de distribución.
La prueba de eliminación es local y solo usa seis casos.

Las falsas alarmas ayudan a formular hipótesis visuales sobre texturas o colores,
pero no demuestran su causa sin un estudio adicional. Las omisiones limítrofes
explican evidencia débil del modelo y no representan los FN persistentes sin
ninguna detección asociable.

## Alcance

No se ha consultado `test`, no se han buscado nuevos umbrales y estos resultados
no cambian la selección final. El análisis satisface el requisito de incluir mapas
de activación y pruebas de oclusión sobre humo, fuego, aciertos, falsas alarmas y
omisiones, con una comprobación cuantitativa de fidelidad.
"""
    target=output/"RESUMEN_INTERPRETABILIDAD.md"; target.write_text(text,encoding="utf-8"); return target


def run(config_path: str | Path) -> Path:
    config_path=Path(config_path)
    if not config_path.is_absolute(): config_path=ROOT/config_path
    config=load_config(config_path.resolve())
    parent=ROOT/config["output_parent"]; parent.mkdir(parents=True,exist_ok=True)
    config_hash=pipeline.sha256_file(config_path)
    latest_path=parent/"latest.json"
    if latest_path.exists():
        latest=json.loads(latest_path.read_text(encoding="utf-8")); existing=ROOT/latest["run_rel"]
        if (existing/"run_summary.json").is_file():
            summary=json.loads((existing/"run_summary.json").read_text(encoding="utf-8"))
            if summary.get("status")=="complete" and summary.get("config_sha256")==config_hash:
                print(f"Interpretabilidad ya completa; se reutiliza: {existing}"); return existing

    weights=ROOT/config["model"]["weights"]
    if pipeline.sha256_file(weights)!=config["model"]["weights_sha256"]: raise ValueError("Hash de pesos distinto")
    cache_summary_path=ROOT/config["validation_cache"]["summary"]
    cache_predictions=ROOT/config["validation_cache"]["predictions"]
    cache_summary=json.loads(cache_summary_path.read_text(encoding="utf-8"))
    if cache_summary.get("split")!="val" or cache_summary["protocol"]["imgsz"]!=768:
        raise ValueError("La caché no corresponde a validación a 768")
    if cache_summary.get("model_sha256")!=config["model"]["weights_sha256"] or cache_summary.get("completed_images")!=1721:
        raise ValueError("La caché no coincide con pesos o cobertura")
    contract=pipeline.validate_prepared_dataset(ROOT,config["dataset_version"])
    staged=pipeline.stage_prepared_dataset(contract,workers=8)
    manifest=pipeline.rebased_manifest(contract,staged)
    val=manifest[manifest.split=="val"].copy().reset_index(drop=True)
    thresholds={0:float(config["operating_point"]["smoke_threshold"]),1:float(config["operating_point"]["fire_threshold"])}
    selection,_=select_cases(cache_predictions,val,config["sample_design"]["cases"],thresholds,
                             float(config["operating_point"]["match_iou"]))
    run_id=dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ_")+config_hash[:8]
    output=parent/run_id; output.mkdir(exist_ok=False); (output/"figures/cases").mkdir(parents=True); (output/"maps").mkdir()
    selection.to_csv(output/"selected_cases.csv",index=False)
    shutil.copy2(config_path,output/"config.yaml"); code=output/"code"; code.mkdir()
    for path in (Path(__file__),ROOT/"tfm_interpretability.py",ROOT/"tools/verify_model_interpretability.py"):
        if path.exists(): shutil.copy2(path,code/path.name)

    model=YOLO(str(weights)); pipeline.validate_class_mapping(model.names)
    device=0 if torch.cuda.is_available() else "cpu"
    metric_rows=[]; all_curves=[]; case_figures=[]
    try:
        for row in selection.itertuples(index=False):
            print(f"[{row.order}/6] {row.case} · {row.class_name} · {row.filename}",flush=True)
            image=cv2.imread(str(row.image_path))
            if image is None: raise FileNotFoundError(row.image_path)
            target_box=json.loads(row.target_box); gt_box=json.loads(row.gt_box) if row.gt_box else None
            eigen_map,layer_maps,fresh_score=capture_eigencam(model,image,target_box,int(row.class_id),config,device)
            if fresh_score<=0: raise RuntimeError(f"No se reprodujo el objetivo: {row.filename}")
            occ=occlusion_sensitivity(model,image,int(row.class_id),target_box,
                grid_size=int(config["occlusion"]["grid_size"]),imgsz=int(config["model"]["imgsz"]),
                conf=float(config["occlusion"]["inference_confidence"]),
                match_iou=float(config["occlusion"]["target_match_iou"]),
                batch=int(config["occlusion"]["batch"]),device=device)
            curves=deletion_curve(model,image,int(row.class_id),target_box,occ["heatmap"],occ["baseline_score"],
                fractions=list(map(float,config["occlusion"]["deletion_fractions"])),
                repeats=int(config["occlusion"]["random_repeats"]),seed=int(config["sample_design"]["seed"])+int(row.order),
                grid_size=int(config["occlusion"]["grid_size"]),imgsz=int(config["model"]["imgsz"]),
                conf=float(config["occlusion"]["inference_confidence"]),
                match_iou=float(config["occlusion"]["target_match_iou"]),
                batch=int(config["occlusion"]["batch"]),device=device)
            curves.insert(0,"filename",row.filename); curves.insert(1,"order",row.order); all_curves.append(curves)
            top_index=int(np.argmax(occ["relative_drops"])); top_masked=mask_tiles(image,occ["blurred"],occ["boxes"],[top_index])
            selected_fraction=min(config["occlusion"]["deletion_fractions"],key=lambda value:abs(float(value)-.25))
            top25=float(curves[(curves.strategy=="important") & np.isclose(curves.fraction,selected_fraction)].score_retained.iloc[0])
            random25=float(curves[(curves.strategy=="random") & np.isclose(curves.fraction,selected_fraction)].score_retained.mean())
            positive_occ=np.maximum(occ["heatmap"],0)
            metric_rows.append({"order":row.order,"case":row.case,"class_name":row.class_name,"filename":row.filename,
                "cached_target_confidence":row.cached_target_confidence,"fresh_target_confidence":fresh_score,
                "occlusion_baseline_confidence":occ["baseline_score"],
                "max_single_occlusion_drop":float(occ["relative_drops"].max()),
                "mean_positive_occlusion_drop":float(positive_occ.mean()),
                "top25_score_retained":top25,"random25_score_retained":random25,
                "top25_better_than_random":bool(top25<random25),
                "occlusion_mass_in_target":heatmap_mass_inside(positive_occ,target_box),
                "occlusion_pointing_game":pointing_game(positive_occ,target_box),
                "eigencam_mass_in_target":heatmap_mass_inside(eigen_map,target_box),
                "eigencam_pointing_game":pointing_game(eigen_map,target_box)})
            np.savez_compressed(output/"maps"/f"{int(row.order):02d}_{Path(row.filename).stem}.npz",
                eigencam=eigen_map,occlusion=occ["heatmap"],target_box=np.asarray(target_box),
                **{f"layer_{index}":value for index,value in layer_maps.items()})
            enriched=pd.Series({**row._asdict(),"fresh_target_confidence":fresh_score})
            case_figures.append(plot_case(image,enriched,target_box,gt_box,eigen_map,occ["heatmap"],top_masked,
                output/"figures/cases",{index:value.shape for index,value in layer_maps.items()}))
    finally:
        del model; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    metrics=pd.DataFrame(metric_rows); curves=pd.concat(all_curves,ignore_index=True)
    metrics.to_csv(output/"interpretability_metrics.csv",index=False); curves.to_csv(output/"deletion_curves.csv",index=False)
    summary_figures=build_summary_figures(output,curves,metrics,case_figures)
    report=write_summary(output,selection,metrics,curves,config)
    run_summary={"schema_version":1,"status":"complete","run_id":run_id,"created_at_utc":dt.datetime.now(dt.timezone.utc).isoformat(),
        "split":"val","test_inference_executed":False,"threshold_search_executed":False,
        "model":config["model"],"operating_point":config["operating_point"],"config_sha256":config_hash,
        "validation_cache_summary_sha256":pipeline.sha256_file(cache_summary_path),"selected_cases":len(selection),
        "methods":["multiscale_eigencam","occlusion_sensitivity","deletion_faithfulness"],
        "figures":[pipeline.project_relative(path,ROOT) for path in case_figures+summary_figures],
        "report_rel":pipeline.project_relative(report,ROOT),
        "output_hashes":{pipeline.project_relative(path,output):pipeline.sha256_file(path)
                         for path in output.rglob("*") if path.is_file() and path.name!="run_summary.json"}}
    pipeline.write_json_atomic(output/"run_summary.json",run_summary)
    pipeline.write_json_atomic(latest_path,{"run_id":run_id,"run_rel":pipeline.project_relative(output,ROOT),
        "run_summary_sha256":pipeline.sha256_file(output/"run_summary.json")})
    print(f"Interpretabilidad: {output}"); return output


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--config",default="configs/model_interpretability.yaml")
    args=parser.parse_args(); run(args.config)


if __name__=="__main__": main()
