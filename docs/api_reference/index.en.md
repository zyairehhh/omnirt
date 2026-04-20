# API Reference

This section is the **auto-generated** Python reference, built by `mkdocstrings` directly from the docstrings in `src/omnirt/`. For task-oriented examples see the [Python API guide](../user_guide/serving/python_api.md).

| Module | Contents |
|---|---|
| [`omnirt`](top_level.md) | Top-level `generate` / `validate` / `pipeline` / `list_available_models` / `describe_model` |
| [`omnirt.requests`](requests.md) | Typed request builders (`text2image`, `image2image`, `text2video`, `image2video`, `audio2video`, `inpaint`, `edit`) |
| [`omnirt.core.types`](types.md) | Core dataclasses: `GenerateRequest` / `GenerateResult` / `RunReport` |
| [`omnirt.core.registry`](registry.md) | Model registration and discovery |
| [`omnirt.core.presets`](presets.md) | Preset resolution (`resolve_preset` / `available_presets`) |
| [`omnirt.core.validation`](validation.md) | Request-validation entrypoints |
| [`omnirt.engine`](engine.md) | `OmniEngine` async dispatch |

!!! info "How this section is built"
    Content is pulled from source docstrings. Keep them in sync when you change public APIs.
