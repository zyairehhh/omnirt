"""Lightweight OTEL-style trace recorder used by the in-process server."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import threading
import time
import uuid
from typing import Any, Dict, List, Optional

from omnirt.core.types import GenerateRequest, StageEventRecord


@dataclass
class TraceSpan:
    span_id: str
    name: str
    stage: str
    started_at_ms: int
    ended_at_ms: Optional[int] = None
    status: str = "ok"
    attributes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TraceRecord:
    trace_id: str
    job_id: str
    task: str
    model: str
    created_at_ms: int
    worker_id: Optional[str] = None
    state: str = "queued"
    error: Optional[str] = None
    events: List[StageEventRecord] = field(default_factory=list)
    spans: List[TraceSpan] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "job_id": self.job_id,
            "task": self.task,
            "model": self.model,
            "created_at_ms": self.created_at_ms,
            "worker_id": self.worker_id,
            "state": self.state,
            "error": self.error,
            "events": [event.__dict__ for event in self.events],
            "spans": [span.to_dict() for span in self.spans],
        }


class TraceRecorder:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._traces: Dict[str, TraceRecord] = {}
        self._spans_by_trace: Dict[str, Dict[tuple[str, str], TraceSpan]] = {}

    def start_trace(self, *, job_id: str, request: GenerateRequest) -> str:
        trace_id = uuid.uuid4().hex
        with self._lock:
            self._traces[trace_id] = TraceRecord(
                trace_id=trace_id,
                job_id=job_id,
                task=request.task,
                model=request.model,
                created_at_ms=int(time.time() * 1000),
            )
            self._spans_by_trace[trace_id] = {}
        return trace_id

    def set_worker(self, trace_id: str, worker_id: str | None) -> None:
        if not worker_id:
            return
        with self._lock:
            trace = self._traces.get(trace_id)
            if trace is not None:
                trace.worker_id = worker_id

    def observe_event(self, trace_id: str, event: StageEventRecord) -> None:
        with self._lock:
            trace = self._traces.get(trace_id)
            if trace is None:
                return
            trace.events.append(event)
            key = (event.stage, event.event)
            if event.event.endswith("start"):
                span = TraceSpan(
                    span_id=uuid.uuid4().hex[:16],
                    name=f"{event.stage}.{event.event}",
                    stage=event.stage,
                    started_at_ms=event.timestamp_ms,
                    attributes=dict(event.data),
                )
                trace.spans.append(span)
                self._spans_by_trace[trace_id][(event.stage, "active")] = span
            elif event.event in {"stage_end", "job_finished", "job_cancelled"}:
                span = self._spans_by_trace[trace_id].pop((event.stage, "active"), None)
                if span is not None:
                    span.ended_at_ms = event.timestamp_ms
                    if event.event == "job_cancelled":
                        span.status = "cancelled"
                    span.attributes.update(dict(event.data))
                if event.event == "job_finished":
                    trace.state = "succeeded"
                elif event.event == "job_cancelled":
                    trace.state = "cancelled"
            elif event.event in {"stage_error", "job_failed"}:
                span = self._spans_by_trace[trace_id].pop((event.stage, "active"), None)
                if span is not None:
                    span.ended_at_ms = event.timestamp_ms
                    span.status = "error"
                    span.attributes.update(dict(event.data))
                trace.error = str(event.data.get("error")) if event.data else trace.error
                if event.event == "job_failed":
                    trace.state = "failed"
            elif event.event == "job_started":
                trace.state = "running"
            else:
                _ = key

    def finish_trace(self, trace_id: str, *, state: str, error: str | None = None) -> None:
        with self._lock:
            trace = self._traces.get(trace_id)
            if trace is None:
                return
            trace.state = state
            if error:
                trace.error = error

    def get_trace(self, trace_id: str) -> Dict[str, Any] | None:
        with self._lock:
            trace = self._traces.get(trace_id)
            return trace.to_dict() if trace is not None else None
