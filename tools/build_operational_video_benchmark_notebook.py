"""Construye el notebook 09 para el benchmark temporal de vídeo."""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.build_pipeline_notebooks import code, markdown, notebook


OUTPUT = ROOT / "notebooks" / "09_DFire_evaluacion_operativa_video.ipynb"


def benchmark_notebook():
    return notebook([
        markdown(
            """
# D-Fire — benchmark operativo de vídeo

**Fase 09 del TFM · evaluación temporal reproducible.**

Este cuaderno compara los checkpoints congelados de YOLO26s y YOLOv8s a nivel
de evento. Mide tiempo hasta detección, continuidad, episodios falsos por hora y
latencia. La inferencia se guarda a confianza mínima y los perfiles temporales
se calculan después offline. El conjunto de test de imágenes permanece cerrado.
"""
        ),
        markdown("## Parámetros de ejecución"),
        code(
            '''
RUN_BENCHMARK = False
OFFLINE = False
CONFIG_PATH = "configs/operational_video_benchmark.yaml"
RUN_ID = None  # None carga la última ejecución completa.
''',
            tag="parameters",
        ),
        code(
            '''
from pathlib import Path
import json
import os
import subprocess
import sys

import pandas as pd
import yaml
from IPython.display import display, Markdown, Image

PROJECT_ROOT = Path(os.environ.get("TFM_PROJECT_ROOT", "/workspace/TFM"))
if not PROJECT_ROOT.is_dir():
    PROJECT_ROOT = Path("C:/Users/elitr/Documents/UPM Data/TFM")
PROJECT_ROOT = PROJECT_ROOT.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

config_path = PROJECT_ROOT / CONFIG_PATH
config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
if not config.get("test_locked"):
    raise ValueError("El benchmark debe mantener test_locked=true.")
output_parent = PROJECT_ROOT / config["output_parent"]
display(Markdown(f"**Proyecto:** `{PROJECT_ROOT}`  \\\n+**Configuración:** `{config_path.relative_to(PROJECT_ROOT)}`"))
''',
        ),
        markdown("## tl;dr"),
        code(
            '''
if RUN_BENCHMARK:
    command = [sys.executable, str(PROJECT_ROOT / "tools/run_operational_video_benchmark.py"),
               "--config", str(config_path)]
    if OFFLINE:
        command.append("--offline")
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)

if RUN_ID is None:
    latest = json.loads((output_parent / "latest.json").read_text(encoding="utf-8"))
    run_dir = PROJECT_ROOT / latest["run_rel"]
else:
    run_dir = output_parent / RUN_ID

from tools.verify_operational_video_benchmark import verify
verification = verify(run_dir)
run_metadata = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
selection = pd.read_csv(run_dir / "pilot_selection.csv")
best = selection.iloc[0]

display(Markdown(
    f"**Ejecución:** `{run_metadata['run_id']}` · "
    f"**{run_metadata['configurations']} configuraciones**, "
    f"{run_metadata['videos']} vídeos.  \\\n+Mejor rango piloto: `{best.model_key} / {best.profile_key} / {best.temporal_rule}`; "
    f"recall de eventos {best.event_recall:.1%}, "
    f"FA/h {best.false_alarms_per_hour:.2f} y TTD mediano "
    f"{best.median_time_to_detection_s:.2f} s.  \\\n+**Selección final permitida:** {'sí' if run_metadata['final_selection_allowed'] else 'no; el corpus sigue siendo piloto'}."
))
''',
        ),
        markdown(
            """
## Contexto y métodos

### Supuestos clave

- Una alarma se activa por clase con una regla M-de-N; la alerta global es humo
  o fuego.
- Un episodio falso completo cuenta una vez, aunque dure varios fotogramas.
- Las predicciones de ambos modelos se muestrean a 5 FPS por timestamp.
- Los umbrales proceden únicamente de validación de imágenes.
- Los roles `independent_operational` y `dfire_domain_diagnostic_only` se
  mantienen separados en los agregados.
- Los intervalos de recall son Wilson 95 %; TTD y falsas alarmas/h usan bootstrap
  por vídeo. Con el corpus piloto deben interpretarse como descriptivos.
"""
        ),
        code(
            '''
profiles = []
for model_key, model in config["models"].items():
    for profile_key, profile in model["profiles"].items():
        profiles.append({
            "Modelo": model["label"], "Perfil": profile["label"],
            "Umbral humo": profile["smoke_threshold"],
            "Umbral fuego": profile["fire_threshold"],
        })
display(pd.DataFrame(profiles))

rules = pd.DataFrame([
    {"Regla": rule["label"], "Ventana": rule["window_frames"],
     "Mínimo de aciertos": rule["minimum_hits"],
     "Cierre tras ausencias": rule["clear_after_misses"]}
    for rule in config["temporal_rules"].values()
])
display(rules)
''',
        ),
        markdown("## Datos"),
        code(
            '''
manifest = pd.read_json(PROJECT_ROOT / config["corpus_manifest"])
coverage = (
    manifest.groupby(["evaluation_role", "event_label"], dropna=False)
    .agg(videos=("id", "count"), minutos=("duration_seconds", lambda x: x.sum() / 60))
    .reset_index()
)
display(coverage)
display(Markdown(
    f"Fuente: `{config['corpus_manifest']}` · SHA-256 "
    f"`{run_metadata['corpus_manifest_sha256'][:16]}…`."
))
''',
        ),
        markdown("## Resultados"),
        code(
            '''
columns = ["pilot_rank", "model_key", "profile_key", "temporal_rule",
           "event_recall", "median_time_to_detection_s", "false_alarms_per_hour",
           "negative_videos_with_alarm", "pareto_efficient"]
display(selection[columns].head(18).style.format({
    "event_recall": "{:.1%}", "median_time_to_detection_s": "{:.2f}",
    "false_alarms_per_hour": "{:.2f}",
}))
''',
        ),
        code(
            '''
figure_alts = {
    "01_false_alarms_vs_detection_time.png": "Falsas alarmas por hora frente a tiempo hasta detección",
    "02_configuration_comparison.png": "Matrices de episodios y tiempo en falsa alarma",
    "03_inference_latency.png": "Latencia mediana y percentil 95 por modelo",
}
for filename, alt in figure_alts.items():
    display(Image(filename=str(run_dir / "figures" / filename), width=1050, alt=alt))
''',
        ),
        markdown("### Resultados por vídeo y por rol"),
        code(
            '''
per_video = pd.read_csv(run_dir / "per_video_metrics.csv")
benchmark_summary = pd.read_csv(run_dir / "benchmark_summary.csv")
timing = pd.read_csv(run_dir / "timing_summary.csv")
display(benchmark_summary.head(12))
display(timing.style.format("{:.3f}", subset=timing.select_dtypes("number").columns))
''',
        ),
        markdown("## Conclusiones"),
        code(
            '''
if run_metadata["final_selection_allowed"]:
    message = (
        "El corpus alcanza los mínimos declarados. Debe revisarse la frontera de "
        "Pareto y congelar una configuración antes del test."
    )
else:
    message = (
        f"Este resultado valida el pipeline, pero no permite elegir un ganador final: "
        f"solo hay {run_metadata['negative_minutes']:.2f} minutos negativos y "
        f"{run_metadata['independent_negative_videos']} vídeos negativos independientes. "
        "Hay que ampliar y congelar el holdout externo antes de la selección."
    )
display(Markdown(message))
display(Markdown(f"Informe completo: `{run_dir / 'RESUMEN_BENCHMARK_VIDEO.md'}`"))
''',
        ),
    ])


def main() -> None:
    nbformat.write(benchmark_notebook(), OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
