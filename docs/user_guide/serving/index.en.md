# Serving

OmniRT offers three entry points, all sharing the same `GenerateRequest` contract:

| Entry point | Best for | Page |
|---|---|---|
| **Python API** | embedding in existing Python apps, notebook experiments | [Python API](python_api.md) |
| **CLI** | scripted batches, one-off `validate` / `generate` | [CLI](cli.md) |
| **HTTP server** | microservice, multi-tenant, OpenAI-compatible API | [HTTP Server](http_server.md) |

!!! tip "Recommended order"
    Start in Python or CLI to validate the contract; then deploy the HTTP server for concurrency, batching, and production policy tuning.
