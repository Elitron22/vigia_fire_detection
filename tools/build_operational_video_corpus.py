"""Build the documented operational video corpus without model inference.

The command is idempotent: existing non-empty files are reused.  It downloads
Commons originals, copies selected D-Fire diagnostic clips, probes all media,
and writes an auditable CSV/JSON manifest plus attribution notes.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
import urllib.parse
import urllib.request
import urllib.error
from collections import Counter
from pathlib import Path

from catalog_operational_videos import probe, write_contact_sheets


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def commons_download_url(filename: str) -> str:
    encoded = urllib.parse.quote(filename, safe="")
    return f"https://commons.wikimedia.org/wiki/Special:Redirect/file/{encoded}"


def download(url: str, target: Path, retries: int = 5) -> bool:
    if target.is_file() and target.stat().st_size > 0:
        print(f"[reuse] {target.name}", flush=True)
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "TFM-D-Fire-operational-corpus/1.0 (academic use)"},
            )
            with urllib.request.urlopen(request, timeout=90) as response, partial.open("wb") as handle:
                content_type = response.headers.get("Content-Type", "")
                if "text/html" in content_type.lower():
                    raise RuntimeError(f"Respuesta HTML inesperada para {target.name}")
                shutil.copyfileobj(response, handle, length=1024 * 1024)
            partial.replace(target)
            print(f"[download] {target.name} ({target.stat().st_size / 1024 / 1024:.1f} MiB)", flush=True)
            return True
        except urllib.error.HTTPError as error:
            partial.unlink(missing_ok=True)
            if attempt == retries:
                raise
            retry_after = error.headers.get("Retry-After")
            if error.code == 429:
                wait_seconds = min(55, int(retry_after) if retry_after and retry_after.isdigit() else 20 * attempt)
                print(f"[rate-limit] pausa de {wait_seconds} s antes de reintentar", flush=True)
            else:
                wait_seconds = min(30, 2**attempt)
            time.sleep(wait_seconds)
        except Exception:
            partial.unlink(missing_ok=True)
            if attempt == retries:
                raise
            time.sleep(min(30, 2**attempt))
    return False


def materialize(source: dict[str, object], destination: Path, root: Path) -> bool:
    target = destination / str(source["target_filename"])
    if source["source_kind"] == "wikimedia_commons":
        url = str(source.get("download_url") or commons_download_url(str(source["commons_filename"])))
        return download(url, target)
    local_path = root / str(source["local_path"])
    if not local_path.is_file():
        raise FileNotFoundError(local_path)
    if target.is_file() and target.stat().st_size == local_path.stat().st_size:
        print(f"[reuse] {target.name}", flush=True)
        return False
    shutil.copy2(local_path, target)
    print(f"[copy] {target.name}", flush=True)
    return True


def display(value: object) -> str:
    if value is None:
        return "—"
    if isinstance(value, list):
        return ", ".join(str(item) for item in value) or "—"
    return str(value)


def write_readme(config: dict[str, object], records: list[dict[str, object]], output_dir: Path) -> None:
    counts = Counter(str(record["event_label"]) for record in records)
    independent = sum(record["evaluation_role"] == "independent_operational" for record in records)
    total_seconds = sum(float(record["duration_seconds"]) for record in records)
    rows = []
    for record in records:
        expected = (
            f"desde {float(record['expected_alert_start_s']):.1f} s"
            if record["expected_alert_start_s"] is not None
            else "sin alerta"
        )
        rows.append(
            "| {id} | {event} | {scenario} | {duration:.1f} s | {resolution} | {fps:.3f} | {expected} | {role} |".format(
                id=record["id"],
                event=record["event_label"],
                scenario=display(record["scenario_tags"]),
                duration=float(record["duration_seconds"]),
                resolution=f"{record['width']}×{record['height']}",
                fps=float(record["fps"]),
                expected=expected,
                role=record["evaluation_role"],
            )
        )

    text = f"""# Corpus operativo de vídeo D-Fire v1

Corpus reproducible para evaluar el comportamiento temporal del detector sin
mezclar estos vídeos con el entrenamiento. Contiene **{len(records)} vídeos**
({counts['positive']} positivos y {counts['negative']} negativos), con
**{independent} vídeos externos independientes** y una duración total de
**{total_seconds / 60:.2f} minutos**.

## Uso correcto

- Los vídeos con `evaluation_role=independent_operational` pueden utilizarse para
  el análisis exploratorio externo.
- Los vídeos con `evaluation_role=dfire_domain_diagnostic_only` solo sirven como
  diagnóstico: podrían estar relacionados con las imágenes D-Fire utilizadas en
  entrenamiento y no demuestran generalización.
- El instante esperado de alerta es una referencia semántica aproximada, no una
  anotación cuadro a cuadro. Debe congelarse antes de ejecutar el modelo final.
- Los vídeos no deben incorporarse al entrenamiento ni utilizarse para elegir
  hiperparámetros. El resultado final debe informar por separado ambos roles.

## Inventario

| ID | Etiqueta | Escenario | Duración | Resolución | FPS | Alerta esperada | Rol |
|---|---|---|---:|---:|---:|---|---|
{chr(10).join(rows)}

## Archivos

- `videos/`: copias locales inmutables de trabajo.
- `source_originals/`: originales AV1 conservados cuando la copia operativa usa
  un transcodificado VP9 compatible; no forman parte del recuento ni de las métricas.
- `manifest.csv` y `manifest.json`: procedencia, licencia, metadatos, SHA-256 y
  expectativa de alerta de cada vídeo.
- `ATTRIBUTIONS.md`: créditos y condiciones de reutilización.
- `contact_sheet_*.jpg`: tres fotogramas de control por vídeo, sin inferencia.

## Métricas previstas

Para negativos: alarmas, vídeos con al menos una alarma y falsas alarmas por hora.
Para positivos: detección del evento, tiempo hasta primera alerta, continuidad de
la alerta y clase detectada. Las reglas temporales deben ser idénticas para todos
los modelos comparados.
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def write_attributions(records: list[dict[str, object]], output_dir: Path) -> None:
    lines = [
        "# Procedencia y atribuciones",
        "",
        "Consultar siempre la página enlazada antes de publicar o redistribuir los vídeos.",
        "Los estados `pending` no invalidan por sí solos la licencia declarada, pero",
        "aconsejan volver a comprobar la página antes de entregar la memoria.",
        "",
    ]
    for record in records:
        lines.extend(
            [
                f"## {record['id']}",
                "",
                f"- Archivo: `{record['relative_path']}`",
                f"- Fuente: [{record['source_page_url']}]({record['source_page_url']})",
                f"- Autoría: {record['author']}",
                f"- Licencia: [{record['license']}]({record['license_url']})",
                f"- Estado de revisión: `{record['license_review_status']}`",
                f"- Nota: {record['notes']}",
                "",
            ]
        )
    (output_dir / "ATTRIBUTIONS.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=project_root() / "configs" / "operational_video_sources.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root() / "data" / "operational_videos_v1",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = project_root()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    active_sources = [source for source in config["sources"] if source.get("enabled", True)]
    output_dir = args.output_dir.resolve()
    video_dir = output_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)

    for source in active_sources:
        fetched = materialize(source, video_dir, root)
        if fetched and source["source_kind"] == "wikimedia_commons":
            time.sleep(8)

    positions = [0.1, 0.5, 0.9]
    entries = []
    records: list[dict[str, object]] = []
    for source in active_sources:
        path = video_dir / str(source["target_filename"])
        media, frames = probe(path, positions)
        record = {**source, **media}
        record["relative_path"] = path.relative_to(root).as_posix()
        record["scenario_tags"] = list(source["scenario_tags"])
        record["expected_classes"] = list(source["expected_classes"])
        records.append(record)
        entries.append((path, record, frames))

    csv_records = []
    for record in records:
        csv_record = dict(record)
        csv_record["scenario_tags"] = ";".join(record["scenario_tags"])
        csv_record["expected_classes"] = ";".join(record["expected_classes"])
        csv_record.pop("commons_filename", None)
        csv_record.pop("local_path", None)
        csv_records.append(csv_record)

    fields = list(dict.fromkeys(key for record in csv_records for key in record))
    with (output_dir / "manifest.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(csv_records)
    (output_dir / "manifest.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_contact_sheets(entries, output_dir, positions, columns=4)
    write_readme(config, records, output_dir)
    write_attributions(records, output_dir)
    print(f"Corpus listo: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
