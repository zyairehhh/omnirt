from __future__ import annotations

import json

from omnirt.core.types import GenerateRequest, StageEventRecord
from omnirt.telemetry.otel import OtlpExporter, TraceRecorder


def test_trace_recorder_exports_completed_trace(monkeypatch) -> None:
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_urlopen(request, timeout=0):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["headers"] = dict(request.header_items())
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    exporter = OtlpExporter(endpoint="http://collector:4318/v1/traces")
    recorder = TraceRecorder(exporters=[exporter])
    trace_id = recorder.start_trace(
        job_id="job-1",
        request=GenerateRequest(task="text2image", model="dummy", inputs={"prompt": "hello"}),
    )
    recorder.set_worker(trace_id, "omnirt-worker-0")
    recorder.observe_event(
        trace_id,
        StageEventRecord(event="stage_start", stage="denoise", timestamp_ms=10, data={"step": 0}),
    )
    recorder.observe_event(
        trace_id,
        StageEventRecord(event="stage_end", stage="denoise", timestamp_ms=20, data={"step": 1}),
    )
    recorder.finish_trace(trace_id, state="succeeded")

    assert captured["url"] == "http://collector:4318/v1/traces"
    assert captured["timeout"] == 5.0
    assert captured["payload"]["resourceSpans"][0]["scopeSpans"][0]["spans"][0]["name"] == "denoise.stage_start"
