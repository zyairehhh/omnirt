# Dispatch & Queue

OmniRT's `engine` layer handles queueing, concurrency, and dynamic batching for multi-request scenarios. The plumbing lives in `omnirt.dispatch`:

| Component | Role | Source |
|---|---|---|
| `JobQueue` | thread-safe job queue (FIFO + priority) | [src/omnirt/dispatch/queue.py](https://github.com/datascale-ai/omnirt/blob/main/src/omnirt/dispatch/queue.py) |
| `Worker` | pulls jobs, invokes pipelines, writes results | [src/omnirt/dispatch/worker.py](https://github.com/datascale-ai/omnirt/blob/main/src/omnirt/dispatch/worker.py) |
| `RequestBatcher` | merges same-task same-model requests inside a time window | [src/omnirt/dispatch/batcher.py](https://github.com/datascale-ai/omnirt/blob/main/src/omnirt/dispatch/batcher.py) |
| `policies` | scheduling policies per task / backend | [src/omnirt/dispatch/policies.py](https://github.com/datascale-ai/omnirt/blob/main/src/omnirt/dispatch/policies.py) |

## What you see from the FastAPI server

The HTTP server exposes these components through `create_app()` parameters (see [HTTP Server](../serving/http_server.md)):

| Parameter | Meaning |
|---|---|
| `max_concurrency` | max in-flight requests in the engine |
| `pipeline_cache_size` | LRU cache for loaded pipelines |
| `batch_window_ms` | batch-aggregation window in ms; `0` disables |
| `max_batch_size` | batch-size cap |

## Typical configurations

=== "Single-card, low latency"

    ```python
    create_app(
        max_concurrency=1,
        pipeline_cache_size=1,
        batch_window_ms=0,
        max_batch_size=1,
    )
    ```

    Minimize per-request latency. Best for **interactive** traffic or VRAM-tight Ascend hosts.

=== "Single-card, high throughput (same model)"

    ```python
    create_app(
        max_concurrency=4,
        pipeline_cache_size=1,
        batch_window_ms=20,
        max_batch_size=4,
    )
    ```

    Short wait inside the batch window in exchange for 2–3× throughput. Only helps for tasks that actually batch (e.g. `text2image` on the same model).

=== "Mixed models"

    ```python
    create_app(
        max_concurrency=2,
        pipeline_cache_size=4,   # keep up to 4 pipelines warm
        batch_window_ms=0,
        max_batch_size=1,
    )
    ```

    No batching, but allows different-model requests to interleave, avoiding pipeline reload overhead.

## Async job API

`POST /v1/jobs` pushes a request onto the queue and returns a `job_id` immediately; clients poll `GET /v1/jobs/{id}` or subscribe to the `GET /v1/jobs/{id}/events` SSE stream. Best for **video** tasks and anything over ~30s, to avoid long-lived HTTP connections.

Full route map in [HTTP Server](../serving/http_server.md).

## Using the engine directly from Python

You don't have to go through HTTP — instantiate `OmniEngine` yourself:

```python
from omnirt.engine import OmniEngine
from omnirt.requests import text2image

engine = OmniEngine(max_concurrency=2, pipeline_cache_size=4,
                    batch_window_ms=0, max_batch_size=1)
job = engine.submit(text2image(model="sd15", prompt="..."))
result = engine.wait(job.id)
```

## Known limits

!!! warning

    - Batching only applies to **same task + same model + same backend**; mixed requests fall back to serial execution
    - On Ascend keep `max_concurrency=1`: NPU memory fragmentation is not promptly reclaimed under concurrency (see [Ascend Backend](../deployment/ascend.md))
    - `pipeline_cache_size` directly controls resident VRAM; video models (Wan2.2, Hunyuan) each take 20–40 GB, so don't crank it

## Related

- [HTTP Server](../serving/http_server.md) — serving and OpenAI-compatible routes
- [Telemetry](telemetry.md) — engine-level metrics
- [Architecture](../../developer_guide/architecture.md) — `BasePipeline` 5-stage model and backend layer
