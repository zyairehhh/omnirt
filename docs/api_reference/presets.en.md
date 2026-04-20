# `omnirt.core.presets`

Preset system: how the four tags (`fast` / `balanced` / `quality` / `low-vram`) merge across task and model dimensions. Auto-rendered reference: [→ API Reference / omnirt.core.presets](../../api_reference/presets/).

Summary:

- **`available_presets() -> tuple[str, ...]`** — returns the recognized preset names.
- **`resolve_preset(*, task: str, model: str, preset: str) -> dict`** — merges base / task / model preset layers and returns the effective config dict.

For the list of preset values per task/model and usage examples, see [Presets](../user_guide/features/presets.md).
