# `omnirt.core.types`

Core dataclasses: the `GenerateRequest` family, `GenerateResult`, and the observable `RunReport`. The auto-rendered reference lives on the default-locale page: [→ API Reference / omnirt.core.types](../../api_reference/types/).

Key types:

- **`GenerateRequest`** — the unified request contract. Fields: `task`, `model`, `backend`, `inputs` (task-specific), `config` (preset, numerical settings), `adapters`.
- **`GenerateResult`** — returned by `generate()`. Fields: `artifacts` (list of `Artifact`), `report` (a `RunReport`), `stream_events` (optional).
- **`RunReport`** — observability payload. Fields: `timings`, `memory`, `backend_timeline`, `config_resolved`, `latent_stats`, `error`.
- **Task-specific request types** — `TextToImageRequest`, `TextToVideoRequest`, `ImageToImageRequest`, `ImageToVideoRequest`, `AudioToVideoRequest`, `InpaintRequest`, `EditRequest`.
- **Capabilities and backend objects** — `Capabilities`, `BackendAttempt`, `BackendTimelineEntry`, `BackendName`, `AdapterRef`.

For field-level details on the wire format, see [Service Schema](../user_guide/features/service_schema.md).
