from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from fastapi.testclient import TestClient
import numpy as np

from fire_app.alerting import AlertEvent, EventStore, TemporalAlertEngine
from fire_app.config import AlertSettings, TelegramSettings, load_settings
from fire_app.main import create_app
from fire_app.inference import InferenceResult, VideoResult, operating_keep_indices
from fire_app.model_registry import ModelRegistry
from fire_app.telegram import TelegramNotifier


def detection(class_name: str, confidence: float) -> dict[str, object]:
    return {"class_name": class_name, "confidence": confidence}


class TemporalAlertEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = AlertSettings(
            enabled=True,
            hold_seconds=2.0,
            clear_seconds=1.0,
            cooldown_seconds=3.0,
            maximum_gap_seconds=0.6,
            minimum_confidence={"smoke": 0.3, "fire": 0.4},
        )

    def test_trigger_requires_continuous_presence(self) -> None:
        engine = TemporalAlertEngine(self.settings, "test")
        self.assertEqual(engine.update(0.0, [detection("fire", 0.8)]), [])
        for timestamp in (0.5, 1.0, 1.5):
            self.assertEqual(engine.update(timestamp, [detection("fire", 0.8)]), [])
        events = engine.update(2.0, [detection("fire", 0.9)])
        self.assertEqual([(event.event_type, event.class_name) for event in events], [("triggered", "fire")])
        self.assertEqual(events[0].confidence, 0.9)
        self.assertEqual(engine.update(2.5, [detection("fire", 0.7)]), [])

    def test_gap_resets_pending_interval(self) -> None:
        engine = TemporalAlertEngine(self.settings, "test")
        engine.update(0.0, [detection("smoke", 0.8)])
        engine.update(1.0, [])
        self.assertEqual(engine.snapshot()["smoke"]["state"], "idle")
        engine.update(1.1, [detection("smoke", 0.8)])
        self.assertEqual(engine.update(2.2, [detection("smoke", 0.8)]), [])

    def test_clear_and_cooldown_prevent_duplicate_alert(self) -> None:
        engine = TemporalAlertEngine(self.settings, "test")
        engine.update(0.0, [detection("fire", 0.8)])
        for timestamp in (0.5, 1.0, 1.5):
            engine.update(timestamp, [detection("fire", 0.8)])
        self.assertEqual(len(engine.update(2.0, [detection("fire", 0.8)])), 1)
        engine.update(2.5, [])
        cleared = engine.update(3.5, [])
        self.assertEqual([event.event_type for event in cleared], ["cleared"])
        self.assertEqual(engine.update(4.0, [detection("fire", 0.9)]), [])
        engine.update(6.5, [])
        self.assertEqual(engine.snapshot()["fire"]["state"], "idle")

    def test_class_threshold_is_applied(self) -> None:
        engine = TemporalAlertEngine(replace(self.settings, hold_seconds=0.0), "test")
        self.assertEqual(engine.update(0.0, [detection("fire", 0.39)]), [])
        self.assertEqual(engine.snapshot()["fire"]["state"], "idle")
        self.assertEqual(len(engine.update(1.0, [detection("smoke", 0.31)])), 1)


class ModelRegistryTests(unittest.TestCase):
    def test_only_complete_models_with_weights_are_listed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            complete = root / "artifacts" / "experiments" / "complete"
            incomplete = root / "artifacts" / "experiments" / "running"
            weights = complete / "train" / "weights" / "best.pt"
            weights.parent.mkdir(parents=True)
            weights.write_bytes(b"weights")
            incomplete.mkdir(parents=True)
            (complete / "experiment.json").write_text(json.dumps({
                "experiment_id": "complete", "status": "complete", "model_key": "yolov8s",
                "completed_utc": "2026-01-02T00:00:00Z", "train_config": {"imgsz": 768},
                "best_model_rel": "artifacts/experiments/complete/train/weights/best.pt",
            }), encoding="utf-8")
            (incomplete / "experiment.json").write_text(json.dumps({
                "experiment_id": "running", "status": "running", "best_model_rel": "missing.pt",
            }), encoding="utf-8")
            models = ModelRegistry(root).discover()
            self.assertEqual([model.experiment_id for model in models], ["complete"])
            self.assertEqual(models[0].trained_imgsz, 768)

    def test_environment_model_supports_ncnn_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "model" / "yolo26s_768_ncnn_model"
            model.mkdir(parents=True)
            (model / "model.ncnn.param").write_text("mock", encoding="utf-8")
            with mock.patch.dict("os.environ", {
                "TFM_APP_MODEL_PATH": str(model),
                "TFM_APP_MODEL_ID": "yolo26s_final_rpi5",
            }, clear=False):
                discovered = ModelRegistry(root).discover()
            self.assertEqual(len(discovered), 1)
            self.assertEqual(discovered[0].backend, "ncnn")
            self.assertTrue(discovered[0].frozen)


class OperatingThresholdTests(unittest.TestCase):
    def test_class_specific_thresholds_filter_visible_detections(self) -> None:
        keep = operating_keep_indices(
            {0: "smoke", 1: "fire"},
            [0, 0, 1, 1],
            [0.35, 0.36, 0.15, 0.16],
            {"smoke": 0.36, "fire": 0.16},
        )
        self.assertEqual(keep, [1, 3])


class TelegramNotifierTests(unittest.TestCase):
    def test_test_message_is_safe_in_dry_run(self) -> None:
        notifier = TelegramNotifier(TelegramSettings(mode="dry_run", timeout_seconds=2.0))
        with mock.patch("fire_app.telegram.httpx.post") as post:
            result = notifier.send_test()
        self.assertEqual(result["status"], "dry_run")
        post.assert_not_called()

    def test_live_test_message_uses_configured_chat(self) -> None:
        notifier = TelegramNotifier(TelegramSettings(mode="live", timeout_seconds=2.0))
        response = mock.Mock()
        response.json.return_value = {"result": {"message_id": 42}}
        with mock.patch.dict("os.environ", {
            "TELEGRAM_BOT_TOKEN": "secret-token",
            "TELEGRAM_CHAT_ID": "12345",
        }, clear=False), mock.patch("fire_app.telegram.httpx.post", return_value=response) as post:
            result = notifier.send_test()
        self.assertEqual(result, {"status": "sent", "telegram_message_id": 42})
        self.assertEqual(post.call_args.kwargs["data"]["chat_id"], "12345")
        self.assertEqual(post.call_args.kwargs["data"]["parse_mode"], "HTML")
        self.assertNotIn("secret-token", post.call_args.kwargs["data"].values())

    def test_alert_message_is_formatted_and_escapes_source(self) -> None:
        notifier = TelegramNotifier(TelegramSettings(mode="live", timeout_seconds=2.0))
        event = AlertEvent.create("triggered", "fire", 3.2, 0.8139, "camara_<principal>&1", "active")
        response = mock.Mock()
        response.json.return_value = {"result": {"message_id": 43}}
        with mock.patch.dict("os.environ", {
            "TELEGRAM_BOT_TOKEN": "secret-token",
            "TELEGRAM_CHAT_ID": "12345",
        }, clear=False), mock.patch("fire_app.telegram.httpx.post", return_value=response) as post:
            result = notifier.send_alert(event, b"jpeg")
        self.assertEqual(result["status"], "sent")
        data = post.call_args.kwargs["data"]
        self.assertEqual(data["parse_mode"], "HTML")
        self.assertIn("ALERTA DE FUEGO", data["caption"])
        self.assertIn("81.4%", data["caption"])
        self.assertIn("camara_&lt;principal&gt;&amp;1", data["caption"])
        self.assertNotIn("camara_<principal>&1", data["caption"])


class ConfigurationAndApiTests(unittest.TestCase):
    def test_application_health_and_model_list(self) -> None:
        with mock.patch.dict("os.environ", {"TFM_TELEGRAM_MODE": "dry_run"}, clear=False):
            settings = load_settings()
        with tempfile.TemporaryDirectory() as temporary:
            temp = Path(temporary)
            settings = replace(settings, output_directory=temp / "output", event_log=temp / "events.jsonl")
            with TestClient(create_app(settings)) as client:
                health = client.get("/api/health")
                models = client.get("/api/models")
                public_config = client.get("/api/config")
                self.assertEqual(health.status_code, 200)
                self.assertEqual(health.json()["status"], "ok")
                self.assertGreaterEqual(len(models.json()["models"]), 1)
                self.assertNotIn("weights", models.json()["models"][0])
                self.assertEqual(public_config.json()["telegram"]["mode"], "dry_run")

    def test_image_detection_sends_one_alert_with_annotated_evidence(self) -> None:
        settings = load_settings()
        prediction = InferenceResult(
            annotated_bgr=np.zeros((32, 32, 3), dtype=np.uint8),
            detections=[
                detection("smoke", 0.91),
                detection("fire", 0.80),
                detection("fire", 0.70),
            ],
            processing_ms=12.0,
        )
        with tempfile.TemporaryDirectory() as temporary:
            temp = Path(temporary)
            settings = replace(settings, output_directory=temp / "output", event_log=temp / "events.jsonl")
            with mock.patch("fire_app.main.decode_image", return_value=np.zeros((32, 32, 3), dtype=np.uint8)), \
                    mock.patch("fire_app.main.ModelManager.predict_frame", return_value=prediction), \
                    mock.patch("fire_app.main.TelegramNotifier.send_alert", return_value={"status": "sent"}) as send:
                with TestClient(create_app(settings)) as client:
                    response = client.post(
                        "/api/analyze/image",
                        content=b"mock-image",
                        headers={"X-Filename": "incendio.jpg"},
                    )
            self.assertEqual(response.status_code, 200)
            self.assertEqual(len(response.json()["events"]), 1)
            self.assertEqual(response.json()["events"][0]["class_name"], "fire")
            send.assert_called_once()
            records = EventStore(settings.event_log).recent()
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["notification"]["status"], "sent")

    def test_video_suppresses_additional_notifications_for_same_source(self) -> None:
        settings = load_settings()

        def fake_process_video(*args, **kwargs):
            output_path = Path(args[2])
            output_path.write_bytes(b"mock-video")
            events = [
                AlertEvent.create("triggered", "smoke", 3.0, 0.8, "incendio.mp4", "active"),
                AlertEvent.create("triggered", "fire", 20.0, 0.9, "incendio.mp4", "active"),
            ]
            for event in events:
                kwargs["on_event"](event, b"mock-jpeg")
            return VideoResult(
                frames_read=100,
                frames_inferred=50,
                fps=25.0,
                duration_seconds=4.0,
                events=events,
                detections_total=10,
                detections_by_class={
                    "smoke": {"count": 7, "max_confidence": 0.8},
                    "fire": {"count": 3, "max_confidence": 0.9},
                },
                processing_ms_per_frame=12.5,
                output_path=str(output_path),
                media_type="video/mp4",
            )

        with tempfile.TemporaryDirectory() as temporary:
            temp = Path(temporary)
            settings = replace(settings, output_directory=temp / "output", event_log=temp / "events.jsonl")
            with mock.patch("fire_app.main.process_video", side_effect=fake_process_video), \
                    mock.patch("fire_app.main.TelegramNotifier.send_alert", return_value={"status": "sent"}) as send:
                with TestClient(create_app(settings)) as client:
                    response = client.post(
                        "/api/analyze/video",
                        content=b"mock-video",
                        headers={"X-Filename": "incendio.mp4"},
                    )
            self.assertEqual(response.status_code, 200)
            self.assertEqual(
                json.loads(response.headers["X-TFM-Detection-Summary"]),
                {
                    "smoke": {"count": 7, "max_confidence": 0.8},
                    "fire": {"count": 3, "max_confidence": 0.9},
                },
            )
            self.assertEqual(float(response.headers["X-TFM-Processing-Ms-Per-Frame"]), 12.5)
            send.assert_called_once()
            records = EventStore(settings.event_log).recent()
            self.assertEqual(len(records), 2)
            self.assertEqual(
                {record["notification"]["status"] for record in records},
                {"sent", "suppressed"},
            )

    def test_event_store_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = EventStore(Path(temporary) / "events.jsonl")
            engine = TemporalAlertEngine(AlertSettings(
                enabled=True, hold_seconds=0, clear_seconds=1, cooldown_seconds=1,
                maximum_gap_seconds=1, minimum_confidence={"smoke": 0.3, "fire": 0.3},
            ), "test")
            event = engine.update(0, [detection("fire", 0.8)])[0]
            store.append(event, {"status": "dry_run"})
            records = store.recent()
            self.assertEqual(records[0]["event_id"], event.event_id)
            self.assertEqual(records[0]["notification"]["status"], "dry_run")

    def test_locked_deployment_rejects_runtime_inference_changes(self) -> None:
        settings = load_settings()
        locked = replace(
            settings,
            inference=replace(settings.inference, allow_runtime_overrides=False),
        )
        with tempfile.TemporaryDirectory() as temporary:
            temp = Path(temporary)
            locked = replace(locked, output_directory=temp / "output", event_log=temp / "events.jsonl")
            with TestClient(create_app(locked)) as client:
                response = client.post(
                    "/api/analyze/image?confidence=0.50",
                    content=b"not-needed-because-parameters-are-rejected-first",
                    headers={"X-Filename": "sample.jpg"},
                )
            self.assertEqual(response.status_code, 409)
            self.assertIn("bloquea", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()
