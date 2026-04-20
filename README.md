# OmniRT

Open image & video generation runtime with CUDA and Ascend backends.

## Current status

The codebase now covers the full v0.1 repository surface described in `PLAN.md`:

- installable Python package layout under `src/`
- `omnirt` CLI with config-file and direct-flag invocation
- `GenerateRequest` / `GenerateResult` / `RunReport`
- backend, pipeline, registry, telemetry, weight-loader, and adapter abstractions
- SDXL `text2image` and SVD `image2video` pipeline implementations targeting Diffusers
- PNG and MP4 artifact export
- unit, parity, integration, and error-path tests
- basic GitHub Actions CI plus optional CUDA / Ascend self-hosted smoke jobs

Real CUDA and Ascend end-to-end generation still requires the corresponding hardware, runtime libraries, and model weights.

## Quickstart

```bash
python3 -m pip install -e .[dev]
python3 -m omnirt --help
pytest
```

For SDXL and SVD runtime support, install the runtime extras too:

```bash
python3 -m pip install -e '.[runtime,dev]'
```

## Example request

```yaml
task: text2image
model: sdxl-base-1.0
backend: auto
inputs:
  prompt: "a cinematic sci-fi city at sunrise"
config:
  num_inference_steps: 30
  guidance_scale: 7.5
  seed: 42
```

Run it with:

```bash
omnirt generate --config request.yaml --json
```

Or use direct CLI flags:

```bash
omnirt generate \
  --task text2image \
  --model sdxl-base-1.0 \
  --prompt "a cinematic sci-fi city at sunrise" \
  --backend cuda \
  --num-inference-steps 30 \
  --guidance-scale 7.5 \
  --seed 42
```

An `image2video` example:

```bash
omnirt generate \
  --task image2video \
  --model svd-xt \
  --image input.png \
  --backend cuda \
  --num-frames 25 \
  --fps 7 \
  --frame-bucket 127 \
  --decode-chunk-size 8 \
  --num-inference-steps 25
```

## Validation

- `pytest tests/unit tests/parity` exercises the local contract and metric layer
- `pytest tests/integration/test_error_paths.py` checks low-memory and bad-weight failures
- the CUDA / Ascend integration tests automatically skip unless hardware, runtime packages, and model-source environment variables are present

The implementation target and remaining hardware validation details are tracked in [PLAN.md](./PLAN.md).

## Roadmaps

- Model onboarding: [docs/model-onboarding.md](./docs/model-onboarding.md)
- Model support roadmap: [docs/model-support-roadmap.md](./docs/model-support-roadmap.md)
- Architecture notes: [docs/architecture.md](./docs/architecture.md)
