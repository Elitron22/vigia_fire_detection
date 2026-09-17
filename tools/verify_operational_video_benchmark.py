"""Verifica cobertura, trazabilidad y bloqueo de test del benchmark de vídeo."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PARENT = ROOT / "artifacts" / "13_operational_video_benchmark" / "pilot"


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def resolve_run(parent: Path, run_id: str | None) -> Path:
    if run_id:
        return parent / run_id
    pointer = json.loads((parent / "latest.json").read_text(encoding="utf-8"))
    return ROOT / pointer["run_rel"]


def verify(run_dir: Path) -> dict[str, object]:
    metadata_path = run_dir / "run_summary.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["status"] == "complete", metadata.get("error")
    assert metadata["test_inference_executed"] is False
    config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    assert config["test_locked"] is True

    per_video = pd.read_csv(run_dir / "per_video_metrics.csv")
    summary = pd.read_csv(run_dir / "benchmark_summary.csv")
    selection = pd.read_csv(run_dir / "pilot_selection.csv")
    timing = pd.read_csv(run_dir / "timing_summary.csv")
    expected_configurations = sum(
        len(model["profiles"]) for model in config["models"].values()
    ) * len(config["temporal_rules"])
    assert metadata["configurations"] == expected_configurations
    assert len(selection) == expected_configurations
    assert per_video.configuration_id.nunique() == expected_configurations
    assert len(per_video) == expected_configurations * metadata["videos"]
    assert set(per_video.model_key) == set(config["models"])
    assert set(per_video.temporal_rule) == set(config["temporal_rules"])
    assert set(summary.evaluation_role).issuperset({"all", *config["reporting_roles"]})
    assert len(timing) == len(config["models"])
    assert (per_video.sampled_frames > 0).all()
    assert (per_video.duration_seconds > 0).all()
    assert (per_video.false_alarm_episodes >= 0).all()
    assert (per_video.false_alarm_active_seconds >= 0).all()
    assert (selection.event_recall.dropna().between(0, 1)).all()
    assert (selection.false_alarms_per_hour.dropna() >= 0).all()
    assert selection.pilot_rank.tolist() == list(range(1, expected_configurations + 1))

    positive = per_video.event_label == "positive"
    negative = per_video.event_label == "negative"
    assert per_video.loc[positive, "event_detected"].notna().all()
    assert (per_video.loc[negative, "false_alarm_episodes"] == per_video.loc[negative, "alarm_episodes"]).all()
    assert (per_video.loc[positive, "false_alarm_episodes"] == 0).all()

    cache_index = json.loads((run_dir / "cache_index.json").read_text(encoding="utf-8"))
    assert set(cache_index) == set(config["models"])
    for model_key, cache in cache_index.items():
        assert cache["status"] == "complete"
        frame_path = ROOT / cache["frame_scores_rel"]
        box_path = ROOT / cache["box_predictions_rel"]
        assert frame_path.is_file() and box_path.is_file(), model_key
        frames = pd.read_csv(frame_path)
        assert frames.video_id.nunique() == metadata["videos"]
        assert len(frames) == cache["sampled_frames"]

    for relative, expected_hash in metadata["output_hashes"].items():
        path = ROOT / relative
        assert path.is_file(), path
        assert sha256_file(path) == expected_hash, f"Hash distinto: {path}"

    summary_path = ROOT / metadata["summary_rel"]
    assert summary_path.is_file()
    text = summary_path.read_text(encoding="utf-8")
    assert "test de imágenes no se cargó" in text
    assert "piloto" in text.lower()
    if metadata["independent_negative_videos"] == 0:
        assert metadata["final_selection_allowed"] is False

    result = {
        "status": "passed", "run_id": metadata["run_id"],
        "configurations": expected_configurations, "videos": metadata["videos"],
        "per_video_rows": len(per_video), "summary_rows": len(summary),
        "test_inference_executed": False,
        "final_selection_allowed": metadata["final_selection_allowed"],
    }
    (run_dir / "verification_report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    run_dir = resolve_run(args.parent.resolve(), args.run_id)
    print(json.dumps(verify(run_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
