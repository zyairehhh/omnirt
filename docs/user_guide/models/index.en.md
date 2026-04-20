# Models

OmniRT's models are owned by the **registry**: every model uses `@register_model` to declare its supported tasks, accepted adapters, minimum VRAM, and recommended preset. This section is organized around three tables:

| Page | Purpose |
|---|---|
| [Supported Models](supported_models.md) | auto-generated full registry (equivalent to `omnirt models`) |
| [Support Status](support_status.md) | manually curated real-hardware smoke and partial-support notes |
| [Roadmap](roadmap.md) | high-priority unsupported models with status |

!!! tip "Query models from the CLI"
    `omnirt models` lists everything; `omnirt models <id>` dumps a model's `ModelCapabilities` (tasks, adapters, VRAM, recommended preset).
