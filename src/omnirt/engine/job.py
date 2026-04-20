"""Job metadata models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from omnirt.core.types import GenerateRequest, GenerateResult, StageEventRecord


@dataclass
class JobRecord:
    id: str
    request: GenerateRequest
    backend: str
    state: str = "queued"
    result: Any = None
    error: Optional[str] = None
    trace_id: Optional[str] = None
    worker_id: Optional[str] = None
    enqueued_at_ms: int = 0
    started_at_ms: Optional[int] = None
    finished_at_ms: Optional[int] = None
    execution_mode: Optional[str] = None
    events: List[StageEventRecord] = field(default_factory=list)

    @property
    def queue_wait_ms(self) -> Optional[float]:
        if self.started_at_ms is None:
            return None
        return max(float(self.started_at_ms - self.enqueued_at_ms), 0.0)

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "id": self.id,
            "state": self.state,
            "backend": self.backend,
            "request": self.request.to_dict(),
            "error": self.error,
            "trace_id": self.trace_id,
            "worker_id": self.worker_id,
            "enqueued_at_ms": self.enqueued_at_ms,
            "started_at_ms": self.started_at_ms,
            "finished_at_ms": self.finished_at_ms,
            "execution_mode": self.execution_mode,
            "events": [event.__dict__ for event in self.events],
        }
        if self.result is not None:
            payload["result"] = self.result.to_dict()
        return payload

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "JobRecord":
        result_payload = payload.get("result")
        return cls(
            id=str(payload["id"]),
            request=GenerateRequest.from_dict(payload["request"]),
            backend=str(payload["backend"]),
            state=str(payload.get("state", "queued")),
            result=GenerateResult.from_dict(result_payload) if isinstance(result_payload, dict) else result_payload,
            error=payload.get("error"),
            trace_id=payload.get("trace_id"),
            worker_id=payload.get("worker_id"),
            enqueued_at_ms=int(payload.get("enqueued_at_ms", 0) or 0),
            started_at_ms=payload.get("started_at_ms"),
            finished_at_ms=payload.get("finished_at_ms"),
            execution_mode=payload.get("execution_mode"),
            events=[StageEventRecord.from_dict(item) for item in payload.get("events", [])],
        )
