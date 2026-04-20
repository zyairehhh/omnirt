"""Execution event contracts shared by executors and the engine."""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, Optional

from omnirt.core.types import StageEventRecord


EventCallback = Callable[[StageEventRecord], None]


def now_ms() -> int:
    return int(time.time() * 1000)


def build_event(event: str, stage: str, *, data: Optional[Dict[str, Any]] = None) -> StageEventRecord:
    return StageEventRecord(
        event=event,
        stage=stage,
        timestamp_ms=now_ms(),
        data=dict(data or {}),
    )


def emit_event(
    callback: Optional[EventCallback],
    event: str,
    stage: str,
    *,
    data: Optional[Dict[str, Any]] = None,
) -> StageEventRecord:
    record = build_event(event, stage, data=data)
    if callback is not None:
        callback(record)
    return record
