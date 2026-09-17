"""Notificaciones de Telegram con modo de simulación por defecto."""

from __future__ import annotations

from html import escape
import os

import httpx

from .alerting import AlertEvent
from .config import TelegramSettings


class TelegramNotifier:
    def __init__(self, settings: TelegramSettings) -> None:
        self.settings = settings

    def public_status(self) -> dict[str, object]:
        configured = bool(os.environ.get("TELEGRAM_BOT_TOKEN") and os.environ.get("TELEGRAM_CHAT_ID"))
        return {
            "mode": self.settings.mode,
            "credentials_configured": configured,
            "ready": self.settings.mode == "live" and configured,
        }

    def _send(self, method: str, *, data: dict[str, str], files: dict[str, tuple[str, bytes, str]] | None = None) -> dict[str, object]:
        token = os.environ.get("TELEGRAM_BOT_TOKEN")
        chat_id = os.environ.get("TELEGRAM_CHAT_ID")
        if not token or not chat_id:
            return {"status": "error", "message": "Faltan TELEGRAM_BOT_TOKEN o TELEGRAM_CHAT_ID."}
        try:
            response = httpx.post(
                f"https://api.telegram.org/bot{token}/{method}",
                data={"chat_id": chat_id, **data},
                files=files,
                timeout=self.settings.timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
            return {"status": "sent", "telegram_message_id": payload.get("result", {}).get("message_id")}
        except (httpx.HTTPError, ValueError) as exc:
            return {"status": "error", "message": f"{type(exc).__name__}: {exc}"}

    def send_test(self) -> dict[str, object]:
        """Comprueba credenciales y destino enviando un mensaje identificable."""
        if self.settings.mode == "disabled":
            return {"status": "disabled", "message": "Telegram está desactivado."}
        if self.settings.mode == "dry_run":
            return {"status": "dry_run", "message": "Telegram está en simulación; no se envió el mensaje."}
        return self._send(
            "sendMessage",
            data={
                "parse_mode": "HTML",
                "text": (
                    "✅ <b>Vigía conectado</b>\n\n"
                    "El canal de alertas está listo para avisar cuando se detecte "
                    "<b>humo</b> o <b>fuego</b>.\n\n"
                    "📷 Los avisos incluirán una captura anotada."
                ),
            },
        )

    def send_alert(self, event: AlertEvent, jpeg_bytes: bytes | None = None) -> dict[str, object]:
        """Envía una captura; en dry_run solo deja constancia en el evento."""
        if self.settings.mode == "disabled":
            return {"status": "disabled"}
        if self.settings.mode == "dry_run":
            return {"status": "dry_run", "message": "Alerta simulada; no se contactó con Telegram."}

        is_fire = event.class_name == "fire"
        icon = "🔥" if is_fire else "🌫️"
        label = "Fuego" if is_fire else "Humo"
        moment = "Imagen estática" if event.timestamp_seconds == 0 else f"{event.timestamp_seconds:.1f} s"
        caption = (
            f"🚨 <b>ALERTA DE {label.upper()}</b>\n"
            "━━━━━━━━━━━━━━\n"
            f"{icon} <b>Detección:</b> {label}\n"
            f"🎯 <b>Confianza máxima:</b> {event.confidence:.1%}\n"
            f"🕒 <b>Instante:</b> {moment}\n"
            f"📍 <b>Origen:</b> <code>{escape(event.source_id)}</code>\n\n"
            "⚠️ Revisa la captura adjunta y verifica la situación."
        )
        if jpeg_bytes:
            return self._send(
                "sendPhoto",
                data={"caption": caption, "parse_mode": "HTML"},
                files={"photo": ("alerta.jpg", jpeg_bytes, "image/jpeg")},
            )
        return self._send("sendMessage", data={"text": caption, "parse_mode": "HTML"})
