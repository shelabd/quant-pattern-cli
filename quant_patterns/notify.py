"""Telegram delivery for scheduled qpat output.

Credentials live in ~/.qpat/config.json:

    qpat config set telegram-bot-token <TOKEN>
    qpat config set telegram-chat-id <CHAT_ID>

stdlib-only (urllib) — no new dependencies.
"""

from __future__ import annotations

import json
import urllib.request

from .macro_calendar import load_config


class TelegramError(RuntimeError):
    """Missing credentials or a failed sendMessage call."""


def telegram_credentials() -> tuple[str, str]:
    cfg = load_config()
    token = cfg.get("telegram_bot_token")
    chat_id = cfg.get("telegram_chat_id")
    if not token or not chat_id:
        raise TelegramError(
            "Telegram not configured. Set credentials with: "
            "qpat config set telegram-bot-token <TOKEN> and "
            "qpat config set telegram-chat-id <CHAT_ID>"
        )
    return str(token), str(chat_id)


def send_telegram(text: str, timeout: int = 15) -> None:
    """Send ``text`` to the configured chat. Raises TelegramError on failure."""
    token, chat_id = telegram_credentials()
    req = urllib.request.Request(
        f"https://api.telegram.org/bot{token}/sendMessage",
        data=json.dumps({"chat_id": chat_id, "text": text}).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode())
    except Exception as e:
        raise TelegramError(f"Telegram send failed: {e}") from e
    if not body.get("ok"):
        raise TelegramError(f"Telegram API error: {body.get('description', body)}")
