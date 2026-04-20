# Telemetry

`middleware.telemetry` writes every generate run to two structured surfaces:

- **`RunReport`** (returned inside `GenerateResult`) — stage timings, resolved config, peak memory, `backend_timeline` (compile / kernel_override / fallback), final latent statistics, error
- **Structured logs** (stdout or your configured sink) — event stream, consumable by any log collector

## `RunReport` field reference

| Field | Type | Description |
|---|---|---|
| `timings` | `dict[stage_name, seconds]` | five-stage timings: `prepare_conditions` / `prepare_latents` / `denoise_loop` / `decode` / `export` |
| `memory` | `dict[str, int]` | peak VRAM (`peak_bytes`) and per-stage samples |
| `backend_timeline` | `list[BackendEvent]` | each compile / kernel_override / fallback outcome |
| `config_resolved` | `dict` | final config after preset merge; essential for reproducibility |
| `latent_stats` | `dict` | final-latent statistics (cross-backend parity) |
| `error` | `ErrorInfo?` | field-structured error on failure |

Definitions in [src/omnirt/telemetry/report.py](https://github.com/datascale-ai/omnirt/blob/main/src/omnirt/telemetry/report.py).

## Minimal example

```python
from omnirt import generate
from omnirt.requests import text2image

result = generate(text2image(model="sd15", prompt="a lighthouse"))
report = result.report
print(report.timings)                  # {'prepare_conditions': 0.41, 'denoise_loop': 2.83, ...}
print(report.memory["peak_bytes"])     # 8_765_432_123
for event in report.backend_timeline:
    print(event.stage, event.kind, event.ok, event.reason)
```

## Streaming events

In the engine / server layer, telemetry also pushes stage events via `attach_stream_events` onto `GenerateResult.stream_events`; this is what the SSE `/v1/jobs/{id}/events` endpoint consumes (see [Dispatch & Queue](dispatch_queue.md)).

## Wiring to an external observability stack

Telemetry is in-process structured data today — no built-in Prometheus / OTLP exporter — but the shape is regular enough for a one-file adapter:

=== "Prometheus adapter"

    ```python
    from prometheus_client import Histogram
    STAGE_HIST = Histogram("omnirt_stage_seconds", "pipeline stage timing",
                           ["task", "model", "stage"])

    def on_run_complete(req, result):
        for stage, secs in result.report.timings.items():
            STAGE_HIST.labels(task=req.task, model=req.model, stage=stage).observe(secs)
    ```

=== "OTLP adapter"

    ```python
    from opentelemetry import metrics
    meter = metrics.get_meter("omnirt")
    stage_hist = meter.create_histogram("omnirt.stage.seconds")

    def on_run_complete(req, result):
        for stage, secs in result.report.timings.items():
            stage_hist.record(secs, attributes={
                "task": req.task, "model": req.model, "stage": stage})
    ```

## Debugging backend fallbacks

`RunReport.backend_timeline` is the first place to look when an Ascend run seems slow:

```python
for ev in result.report.backend_timeline:
    if not ev.ok:
        print(f"[{ev.stage}] {ev.kind} failed: {ev.reason}")
```

Each entry records the stage, action (`compile` / `kernel_override` / `fallback`), outcome, and reason. Details in [Architecture](../../developer_guide/architecture.md) under the backend-layer section.

## Related

- [Python API](../serving/python_api.md) — reading `report` off `GenerateResult`
- [HTTP Server](../serving/http_server.md) — subscribing via SSE
- [Dispatch & Queue](dispatch_queue.md) — engine-level metrics across requests
