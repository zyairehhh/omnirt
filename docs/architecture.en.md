# OmniRT Architecture

`omnirt` is organized around a component-oriented runtime that keeps the public interface stable while allowing backend-specific execution strategies.

## Layers

1. **User interface layer** — [src/omnirt/api.py](https://github.com/datascale-ai/omnirt/blob/main/src/omnirt/api.py) + [src/omnirt/cli/main.py](https://github.com/datascale-ai/omnirt/blob/main/src/omnirt/cli/main.py)
   `omnirt.generate(...)`, `omnirt.validate(...)`, `omnirt.pipeline(...)`, and the `omnirt` CLI all normalize inputs into `GenerateRequest` before handing them down the stack.
2. **Pipeline layer** — [src/omnirt/core/base_pipeline.py](https://github.com/datascale-ai/omnirt/blob/main/src/omnirt/core/base_pipeline.py)
   `BasePipeline` provides the five-stage skeleton: `prepare_conditions`, `prepare_latents`, `denoise_loop`, `decode`, `export`. Every stage is timed by the telemetry layer.
3. **Component / model layer** — [src/omnirt/models/](https://github.com/datascale-ai/omnirt/tree/main/src/omnirt/models)
   Each model family (SDXL, SVD, Flux2, Wan2.2, Qwen-Image, CogVideoX, HunyuanVideo, and more) implements a `BasePipeline` subclass and attaches to the registry via `@register_model`. The live list is at [_generated/models.md](_generated/models.md).
4. **Scheduler layer** — [src/omnirt/schedulers/](https://github.com/datascale-ai/omnirt/tree/main/src/omnirt/schedulers)
   Thin wrappers for common schedulers (`euler-discrete`, `euler-ancestral`, `ddim`, `dpm-solver`, `dpm-solver-karras`) are dispatched by `SCHEDULER_REGISTRY` + `build_scheduler(config)`. Pipelines never import Diffusers scheduler classes directly.
5. **Backend layer** — [src/omnirt/backends/](https://github.com/datascale-ai/omnirt/tree/main/src/omnirt/backends)
   `BackendRuntime.wrap_module(...)` attempts `compile`, then `kernel_override`, then eager. Each attempt is recorded in `RunReport.backend_timeline`. Current runtimes: `CudaBackend`, `AscendBackend`, `CpuStubBackend`.
6. **Telemetry layer** — [src/omnirt/telemetry/](https://github.com/datascale-ai/omnirt/tree/main/src/omnirt/telemetry)
   `log.py` emits structured logs; `report.py` builds the `RunReport` (stage timings, peak memory, backend-fallback chain, terminal latent statistics).
7. **Support infrastructure** — [src/omnirt/core/](https://github.com/datascale-ai/omnirt/tree/main/src/omnirt/core)
   Registry, `safetensors`-only weight loading, adapter loading, presets, validation, and parity helpers all live here.

## Public contracts

- `GenerateRequest` carries `task`, `model`, `backend`, task-specific `inputs`, generation `config`, and optional adapters.
- `GenerateResult` contains exported artifacts and a `RunReport`.
- `RunReport` records stage timings, resolved config, peak memory, backend fallback attempts, terminal-latent statistics (for cross-backend parity), and any surfaced error.
- Field-level reference: [service-schema.en.md](service-schema.en.md).

## Registry and aliases

A single pipeline class can expose multiple registry ids by stacking `@register_model(...)` decorators — this is exactly how `flux2.dev` and `flux2-dev` share an implementation. Aliases declare themselves via `ModelCapabilities.alias_of`, and `omnirt models --format markdown` renders canonical ids separately from alias rows. See [model-onboarding.en.md](model-onboarding.en.md) for how to add either.

## Presets

All pipelines share one preset vocabulary (`fast` / `balanced` / `quality` / `low-vram`), merged into `config` during `prepare_conditions`. What each preset changes per task/model is documented in [presets.en.md](presets.en.md), sourced from [src/omnirt/core/presets.py](https://github.com/datascale-ai/omnirt/blob/main/src/omnirt/core/presets.py).

## Artifact export

- `text2image` writes PNG artifacts via Pillow.
- `text2video` / `image2video` / `audio2video` mux MP4 via `imageio-ffmpeg` in [src/omnirt/core/media.py](https://github.com/datascale-ai/omnirt/blob/main/src/omnirt/core/media.py).

## Test tiers

- `tests/unit/` covers contracts, registry, CLI, and pipeline behavior using fake runtimes and fake Diffusers objects.
- `tests/parity/` covers latent statistics plus image / video metric helpers.
- `tests/integration/` contains CUDA/Ascend smoke tests and error-path coverage; hardware-dependent cases skip automatically when prerequisites are missing.
