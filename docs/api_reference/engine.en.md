# `omnirt.engine`

The async engine underlying the HTTP server and batched dispatch. Auto-rendered reference: [→ API Reference / omnirt.engine](../../api_reference/engine/).

Summary:

- **`OmniEngine`** — constructor takes `max_concurrency`, `pipeline_cache_size`, `batch_window_ms`, `max_batch_size`. `submit(request)` returns a `Job`; `wait(job_id)` blocks for its `GenerateResult`.
- **`ResultCache`** — LRU cache of `GenerateResult` objects keyed by request signature (enabled by the HTTP server).
- **`get_default_engine()`** — lazily constructs a process-wide `OmniEngine` singleton.

For concurrency / batching tuning, see [Dispatch & Queue](../user_guide/features/dispatch_queue.md).
