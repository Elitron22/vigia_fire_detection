"""Regla temporal y registro auditable de incidentes."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import threading
import uuid

from .config import AlertSettings


@dataclass(frozen=True)
class AlertEvent:
    event_type: str
    class_name: str
    timestamp_seconds: float
    confidence: float
    source_id: str
    state: str
    event_id: str
    created_utc: str

    @classmethod
    def create(
        cls,
        event_type: str,
        class_name: str,
        timestamp_seconds: float,
        confidence: float,
        source_id: str,
        state: str,
    ) -> "AlertEvent":
        return cls(
            event_type=event_type,
            class_name=class_name,
            timestamp_seconds=float(timestamp_seconds),
            confidence=float(confidence),
            source_id=source_id,
            state=state,
            event_id=uuid.uuid4().hex,
            created_utc=datetime.now(timezone.utc).isoformat(),
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class TemporalAlertEngine:
    """Exige presencia continua y evita repetir alertas durante el cooldown."""

    def __init__(self, settings: AlertSettings, source_id: str) -> None:
        self.settings = settings
        self.source_id = source_id
        self._classes: dict[str, dict[str, float | str | None]] = {
            name: {
                "state": "idle",
                "pending_since": None,
                "last_seen": None,
                "absent_since": None,
                "cooldown_until": None,
                "max_confidence": 0.0,
            }
            for name in settings.minimum_confidence
        }

    def update(self, timestamp_seconds: float, detections: list[dict[str, object]]) -> list[AlertEvent]:
        if not self.settings.enabled:
            return []
        timestamp = float(timestamp_seconds)
        best = {name: 0.0 for name in self._classes}
        for detection in detections:
            name = str(detection.get("class_name", ""))
            if name in best:
                best[name] = max(best[name], float(detection.get("confidence", 0.0)))

        events: list[AlertEvent] = []
        for name, data in self._classes.items():
            confidence = best[name]
            present = confidence >= self.settings.minimum_confidence[name]
            state = str(data["state"])
            cooldown_until = data["cooldown_until"]

            if state == "cooldown" and cooldown_until is not None and timestamp >= float(cooldown_until):
                data.update(state="idle", pending_since=None, absent_since=None, max_confidence=0.0)
                state = "idle"

            if present:
                last_seen = data["last_seen"]
                if state == "pending" and last_seen is not None and timestamp - float(last_seen) > self.settings.maximum_gap_seconds:
                    data["pending_since"] = timestamp
                    data["max_confidence"] = confidence
                data["last_seen"] = timestamp
                data["absent_since"] = None
                data["max_confidence"] = max(float(data["max_confidence"] or 0.0), confidence)

                if state == "idle":
                    data.update(state="pending", pending_since=timestamp, max_confidence=confidence)
                pending_since = data["pending_since"]
                if data["state"] == "pending" and pending_since is not None and timestamp - float(pending_since) >= self.settings.hold_seconds:
                    data["state"] = "active"
                    event = AlertEvent.create(
                        "triggered", name, timestamp, float(data["max_confidence"]), self.source_id, "active"
                    )
                    events.append(event)
                continue

            if state == "pending":
                last_seen = data["last_seen"]
                if last_seen is None or timestamp - float(last_seen) > self.settings.maximum_gap_seconds:
                    data.update(state="idle", pending_since=None, last_seen=None, max_confidence=0.0)
            elif state == "active":
                if data["absent_since"] is None:
                    data["absent_since"] = timestamp
                absent_since = data["absent_since"]
                if absent_since is not None and timestamp - float(absent_since) >= self.settings.clear_seconds:
                    events.append(AlertEvent.create("cleared", name, timestamp, 0.0, self.source_id, "cooldown"))
                    data.update(
                        state="cooldown",
                        pending_since=None,
                        last_seen=None,
                        absent_since=None,
                        max_confidence=0.0,
                        cooldown_until=timestamp + self.settings.cooldown_seconds,
                    )
        return events

    def snapshot(self) -> dict[str, dict[str, float | str | None]]:
        return {name: dict(values) for name, values in self._classes.items()}


class EventStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()

    def append(self, event: AlertEvent, notification: dict[str, object] | None = None) -> None:
        record = event.to_dict()
        if notification is not None:
            record["notification"] = notification
        line = json.dumps(record, ensure_ascii=False)
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(line + "\n")

    def recent(self, limit: int = 50) -> list[dict[str, object]]:
        if not self.path.is_file():
            return []
        with self._lock:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        records: list[dict[str, object]] = []
        for line in lines[-max(1, min(limit, 500)) :]:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return list(reversed(records))
