# Features

Beyond the `GenerateRequest` contract, OmniRT provides several cross-cutting capabilities:

| Feature | Purpose | Page |
|---|---|---|
| **Presets** | `fast` / `balanced` / `quality` / `low-vram` tags that bundle steps / dtype / guidance | [Presets](presets.md) |
| **Request validation** | Catch contract errors before using real hardware | [Validation](validation.md) |
| **Service schema** | Field-level reference for `GenerateRequest` / `GenerateResult` / `RunReport` | [Service Schema](service_schema.md) |
| **Dispatch queue** | Async engine, concurrency, batching, workers | [Dispatch & Queue](dispatch_queue.md) |
| **Telemetry** | Structured logs for stage timings, peak memory, fallbacks | [Telemetry](telemetry.md) |
