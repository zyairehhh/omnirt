"""Structured logging helpers."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        data: Dict[str, Any] = {}
        for key in ("run_id", "model", "backend", "elapsed_ms", "error"):
            value = getattr(record, key, None)
            if value is not None:
                data[key] = value

        payload: Dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "stage": getattr(record, "stage", None),
            "event": record.getMessage(),
            "data": data,
        }
        return json.dumps(payload, ensure_ascii=False)


_LOGGER: Optional[logging.Logger] = None


def get_logger() -> logging.Logger:
    global _LOGGER
    if _LOGGER is not None:
        return _LOGGER

    logger = logging.getLogger("omnirt")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        log_path = Path(os.getenv("OMNIRT_LOG_PATH", "omnirt.log"))
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(log_path, encoding="utf-8")
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)

    _LOGGER = logger
    return logger
