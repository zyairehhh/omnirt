# OmniRT Architecture

`omnirt` is organized around a component-oriented runtime that keeps the public interface stable while allowing backend-specific execution strategies.

## Layers

1. User interface layer
   `omnirt.generate(...)` and the `omnirt` CLI both normalize inputs into `GenerateRequest`.
2. Pipeline layer
   `BasePipeline` provides the five-stage skeleton: `prepare_conditions`, `prepare_latents`, `denoise_loop`, `decode`, and `export`.
3. Component layer
   SDXL and SVD are assembled from Diffusers-style building blocks such as text encoders, image encoders, UNets, VAEs, and schedulers.
4. Backend layer
   `BackendRuntime.wrap_module(...)` attempts `compile`, then `kernel_override`, then eager fallback. Each attempt is recorded in `RunReport.backend_timeline`.
5. Support layer
   Registry, safetensors-only loading, adapter loading, structured logging, and parity helpers live here.

## Public contracts

- `GenerateRequest` carries `task`, `model`, `backend`, task-specific `inputs`, generation `config`, and optional adapters.
- `GenerateResult` contains exported artifacts and a `RunReport`.
- `RunReport` records stage timings, resolved config, peak memory, backend fallback attempts, and any surfaced error.

## Model paths

- `sdxl-base-1.0` maps to `SDXLPipeline`
- `svd-xt` maps to `SVDPipeline`

Both pipelines cache the loaded Diffusers pipeline inside the object instance, wrap key modules once, then reuse the wrapped pipeline across runs.

## Export model

- `text2image` exports PNG artifacts
- `image2video` exports MP4 artifacts through `imageio-ffmpeg`

## Testing model

- `tests/unit/` covers contracts and pipeline behavior with fakes
- `tests/parity/` covers metric and threshold helpers
- `tests/integration/` contains smoke and error-path coverage, with hardware-dependent cases skipped automatically when prerequisites are missing
