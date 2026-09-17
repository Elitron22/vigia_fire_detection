"""Verify the operational video corpus structure, hashes and decodability."""

from __future__ import annotations

import json
from pathlib import Path

from catalog_operational_videos import probe


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "operational_videos_v1"


def main() -> None:
    records = json.loads((CORPUS / "manifest.json").read_text(encoding="utf-8"))
    assert records, "El manifiesto está vacío"
    assert len({record["id"] for record in records}) == len(records), "Hay ID duplicados"
    assert len({record["relative_path"] for record in records}) == len(records), "Hay rutas duplicadas"

    positive_count = 0
    negative_count = 0
    independent_count = 0
    all_tags: set[str] = set()
    for record in records:
        path = ROOT / record["relative_path"]
        assert path.is_file() and path.stat().st_size > 0, f"Falta {path}"
        media, frames = probe(path, [0.1, 0.5, 0.9])
        assert media["sha256"] == record["sha256"], f"SHA-256 distinto en {path.name}"
        assert len(frames) == 3, f"No se decodifican tres muestras de {path.name}"
        assert float(media["duration_seconds"]) > 0, f"Duración no válida en {path.name}"
        for field in ("source_page_url", "author", "license", "license_url", "content"):
            assert record.get(field), f"Falta {field} en {record['id']}"

        all_tags.update(record["scenario_tags"])
        if record["event_label"] == "positive":
            positive_count += 1
            assert record["expected_alert_start_s"] is not None
            assert record["expected_classes"]
        elif record["event_label"] == "negative":
            negative_count += 1
            assert record["expected_alert_start_s"] is None
            assert not record["expected_classes"]
        else:
            raise AssertionError(f"Etiqueta desconocida: {record['event_label']}")
        independent_count += record["evaluation_role"] == "independent_operational"

    required_tag_groups = [
        {"wildfire", "prescribed_burn"},
        {"smoke"},
        {"fog", "low_cloud"},
        {"night", "reflection"},
        {"red_objects", "colored_object"},
        {"clean_negative"},
    ]
    for alternatives in required_tag_groups:
        assert all_tags.intersection(alternatives), f"Escenario sin cubrir: {alternatives}"

    summary = {
        "status": "ok",
        "videos": len(records),
        "positive": positive_count,
        "negative": negative_count,
        "independent_operational": independent_count,
        "total_duration_seconds": round(sum(float(r["duration_seconds"]) for r in records), 3),
        "scenario_tags": sorted(all_tags),
    }
    (CORPUS / "verification.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
