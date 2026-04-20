# HTTP Server (FastAPI)

The `server` extra ships a ready-to-run FastAPI application with OmniRT's native `/v1/generate` endpoint plus **OpenAI-compatible** routes (`/v1/images/generations`, `/v1/videos/generations`, `/v1/audio/speech`) that accept existing OpenAI client SDKs.

## Install and launch

```bash
# Install the server extra (FastAPI / uvicorn / pydantic / sse-starlette)
python -m pip install -e '.[runtime,server]'

# Simplest launch
uvicorn 'omnirt.server.app:create_app' --factory --host 0.0.0.0 --port 8000
```

`create_app()` accepts the following parameters (overridable via env vars or a `--factory` wrapper):

| Parameter | Default | Description |
|---|---|---|
| `default_backend` | `"auto"` | Server-wide default backend; a request's `backend` field still overrides |
| `max_concurrency` | `1` | Max concurrent jobs in the engine; keep `1` on Ascend |
| `pipeline_cache_size` | `4` | LRU cache size for loaded model pipelines; lower when VRAM-constrained |
| `batch_window_ms` | `0` | Dynamic-batching aggregation window; `0` disables |
| `max_batch_size` | `1` | Dynamic-batching cap |
| `api_key_file` | `None` | Line-separated API-key file; presence enables `ApiKeyMiddleware` |
| `model_aliases_path` | `None` | YAML / JSON map of OpenAI-style model names → OmniRT registry ids |

## Route map

### Native endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/healthz` | liveness probe |
| `GET` | `/readyz` | readiness probe (engine initialized) |
| `POST` | `/v1/generate` | synchronous generation; returns `GenerateResult` |
| `POST` | `/v1/jobs` | async submission; returns `job_id` |
| `GET` | `/v1/jobs/{job_id}` | job status / result |
| `DELETE` | `/v1/jobs/{job_id}` | cancel a job |
| `GET` | `/v1/jobs/{job_id}/events` | SSE event stream |

### OpenAI-compatible endpoints

| Method | Path | Mapped to |
|---|---|---|
| `POST` | `/v1/images/generations` | `text2image` |
| `POST` | `/v1/images/edits` | `inpaint` / `edit` |
| `POST` | `/v1/videos/generations` | `text2video` |
| `POST` | `/v1/audio/speech` | (external TTS adapter) |

These endpoints accept OpenAI-client payloads; `model_aliases_path` maps names like `gpt-image-1` to OmniRT registry ids like `flux2.dev`.

## Minimal examples

=== "Native /v1/generate"

    ```bash
    curl -sS http://localhost:8000/v1/generate \
      -H 'Content-Type: application/json' \
      -d '{
        "task": "text2image",
        "model": "sd15",
        "inputs": {"prompt": "a lighthouse"},
        "config": {"preset": "fast"}
      }'
    ```

=== "OpenAI-compatible"

    ```python
    from openai import OpenAI
    client = OpenAI(base_url="http://localhost:8000/v1", api_key="sk-omnirt-local")
    resp = client.images.generate(
        model="sd15",
        prompt="a lighthouse in fog",
        size="512x512",
    )
    print(resp.data[0].url)
    ```

=== "Async job"

    ```bash
    # Submit
    curl -sS -X POST http://localhost:8000/v1/jobs \
      -H 'Content-Type: application/json' \
      -d '{"task": "text2video", "model": "wan2.2-t2v-14b",
           "inputs": {"prompt": "..."}, "config": {"num_frames": 81}}'
    # -> {"job_id": "abc123", "status": "queued"}

    # Poll
    curl -sS http://localhost:8000/v1/jobs/abc123

    # Subscribe (SSE)
    curl -sS -N http://localhost:8000/v1/jobs/abc123/events
    ```

## API keys

```bash
# /etc/omnirt/api-keys.txt — one key per line
printf 'sk-omnirt-alice\nsk-omnirt-bob\n' > api-keys.txt

OMNIRT_API_KEY_FILE=api-keys.txt \
uvicorn 'omnirt.server.app:create_app' --factory \
  --host 0.0.0.0 --port 8000
```

Clients send `Authorization: Bearer <key>`.

## Concurrency and batching

- `max_concurrency=N` — at most N requests in the engine at once
- `pipeline_cache_size=K` — keep K pipelines warm (shared weights across requests)
- `batch_window_ms=10, max_batch_size=4` — aggregate same-task same-model requests inside the window; benchmark before enabling — batching is latency-sensitive

**Ascend-specific**: use `max_concurrency=1, pipeline_cache_size=1` to avoid NPU memory fragmentation (see the "memory not released" note in [Ascend Backend](../deployment/ascend.md)).

## Containerization

Docker / k8s templates live in [Docker Deployment](../deployment/docker.md).

## Observability

The server writes stage timings, peak memory, and fallback events into structured logs and each `RunReport`. See [Telemetry](../features/telemetry.md).
