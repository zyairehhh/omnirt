"""Helpers for emitting server-sent events."""

from __future__ import annotations

import json

from omnirt.core.types import StageEventRecord


def encode_sse_event(event: StageEventRecord) -> str:
    payload = json.dumps(event.__dict__, ensure_ascii=False)
    return f"event: {event.event}\ndata: {payload}\n\n"
