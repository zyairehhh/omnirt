"""Lightweight OTEL-style trace recorder with optional OTLP export."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import threading
import time
import urllib.request
import uuid
from typing import Any, Dict, Iterable, List, Optional

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


class OtlpExporter:
    def __init__(
        self,
        *,
        endpoint: str,
        service_name: str = "omnirt",
        headers: Optional[Dict[str, str]] = None,
        timeout_s: float = 5.0,
    ) -> None:
        self.endpoint = endpoint
        self.service_name = service_name
        self.headers = dict(headers or {})
        self.timeout_s = timeout_s

    def export_trace(self, trace: Dict[str, Any]) -> None:
        payload = self._build_payload(trace)
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"content-type": "application/json", **self.headers},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_s):
            return None

    def _build_payload(self, trace: Dict[str, Any]) -> Dict[str, Any]:
        resource_attrs = [
            {"key": "service.name", "value": {"stringValue": self.service_name}},
            {"key": "omnirt.model", "value": {"stringValue": str(trace["model"])}},
            {"key": "omnirt.task", "value": {"stringValue": str(trace["task"])}},
        ]
        scope_spans = []
        for span in trace.get("spans", []):
            scope_spans.append(
                {
                    "traceId": trace["trace_id"],
                    "spanId": span["span_id"],
                    "name": span["name"],
                    "startTimeUnixNano": int(span["started_at_ms"]) * 1_000_000,
                    "endTimeUnixNano": int(span.get("ended_at_ms") or span["started_at_ms"]) * 1_000_000,
                    "attributes": [
                        {"key": str(key), "value": {"stringValue": str(value)}}
                        for key, value in dict(span.get("attributes") or {}).items()
                    ],
                    "status": {"message": str(span.get("status", "ok")).lower()},
                }
            )
        return {
            "resourceSpans": [
                {
                    "resource": {"attributes": resource_attrs},
                    "scopeSpans": [
                        {
                            "scope": {"name": "omnirt", "version": "1.0.0"},
                            "spans": scope_spans,
                        }
                    ],
                }
            ]
        }


class TraceRecorder:
    def __init__(self, *, exporters: Optional[Iterable[OtlpExporter]] = None) -> None:
        self._lock = threading.RLock()
        self._traces: Dict[str, TraceRecord] = {}
        self._spans_by_trace: Dict[str, Dict[tuple[str, str], TraceSpan]] = {}
        self._exporters = list(exporters or [])

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

    def finish_trace(self, trace_id: str, *, state: str, error: str | None = None) -> None:
        trace_payload = None
        with self._lock:
            trace = self._traces.get(trace_id)
            if trace is None:
                return
            trace.state = state
            if error:
                trace.error = error
            trace_payload = trace.to_dict()
        for exporter in self._exporters:
            try:
                exporter.export_trace(trace_payload)
            except Exception:
                continue

    def get_trace(self, trace_id: str) -> Dict[str, Any] | None:
        with self._lock:
            trace = self._traces.get(trace_id)
            return trace.to_dict() if trace is not None else None
