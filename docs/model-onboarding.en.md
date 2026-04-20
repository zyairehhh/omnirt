# Model Onboarding

This guide is the shortest path for onboarding a new Diffusers-style model into `omnirt`.

## 1. Add a pipeline

Create `src/omnirt/models/<model_name>/pipeline.py` and register the class:

```python
@register_model(
    id="my-model",
    task="text2image",
    default_backend="auto",
    resource_hint={"min_vram_gb": 12, "dtype": "fp16"},
)
class MyPipeline(BasePipeline):
    ...
```

Implement the five `BasePipeline` stages:

1. `prepare_conditions`
2. `prepare_latents`
3. `denoise_loop`
4. `decode`
5. `export`

## 2. Reuse the backend contract

Wrap backend-sensitive submodules through `self.runtime.wrap_module(...)`. This gives the new model compile, override, and eager fallback behavior without adding model-specific backend code.

## 3. Use supported formats

- weights: `safetensors`
- adapters: validate at pipeline initialization time, then apply once when the runtime pipeline is materialized; single-file adapters may come from local files or Hugging Face
- artifacts: emit `Artifact` records with concrete file paths

## 4. Add tests

- unit tests with fake runtimes and fake Diffusers objects
- smoke integration tests under `tests/integration/`
- parity metric coverage if the model participates in cross-backend validation

## 5. Document assumptions

Capture:

- model source or registry id
- minimum memory hint
- supported task schema
- backend caveats
- exported artifact format

Following the SDXL and SVD pipelines in `src/omnirt/models/` is the intended reference path.
