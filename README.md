# OmniRT

[中文](./README.zh-CN.md) | [Documentation](https://datascale-ai.github.io/omnirt/) | [English Docs](https://datascale-ai.github.io/omnirt/en/)

Open runtime for image, video, and audio-driven avatar generation across CUDA and Ascend backends.

OmniRT gives model families a unified CLI, Python API, validation flow, artifact export contract, and backend abstraction, so you can move across model families without relearning the entire runtime surface.

## Highlights

- Unified request/result contract built around `GenerateRequest`, `GenerateResult`, and `RunReport`
- Public CLI centered on `omnirt generate`, `omnirt validate`, and `omnirt models`
- Python helpers via `requests.*` and a convenience `pipeline(...)` wrapper
- CUDA, Ascend, and `cpu-stub` backend modes
- PNG and MP4 artifact export
- Model sources can use local directories or Hugging Face repo ids
- LoRA safetensors can be loaded from local files or Hugging Face single-file refs
- Deployment-friendly workflow for local model directories and restricted networks

## Public Task Surfaces

| Task | Description | Typical outputs |
|---|---|---|
| `text2image` | prompt-driven image generation | PNG |
| `image2image` | image-guided image generation | PNG |
| `text2video` | prompt-driven video generation | MP4 |
| `image2video` | first-frame-guided video generation | MP4 |
| `audio2video` | audio-driven talking avatar generation | MP4 |

The CLI is the fastest way to inspect the exact live registry:

```bash
omnirt models
omnirt models flux2.dev
```

## Supported Model Families

Representative families currently wired into the registry:

- Stable Diffusion: `sd15`, `sd21`, `sdxl-base-1.0`, `sdxl-refiner-1.0`, `sdxl-turbo`, `sd3-medium`, `sd3.5-large`, `sd3.5-large-turbo`
- Flux: `flux-dev`, `flux-depth`, `flux-schnell`, `flux-canny`, `flux-fill`, `flux-kontext`, `flux2.dev`, `flux2-dev`
- Generalist image: `chronoedit`, `kolors`, `glm-image`, `hunyuan-image-2.1`, `omnigen`, `qwen-image`, `qwen-image-edit`, `qwen-image-edit-plus`, `qwen-image-layered`, `sana-1.6b`, `ovis-image`, `hidream-i1`, `pixart-sigma`, `bria-3.2`, `lumina-t2x`
- Video: `svd`, `svd-xt`, `animate-diff-sdxl`, `mochi`, `cogvideox-2b`, `cogvideox-5b`, `kandinsky5-t2v`, `kandinsky5-i2v`, `wan2.1-*`, `wan2.2-*`, `hunyuan-video`, `hunyuan-video-1.5-*`, `helios-*`, `sana-video`, `ltx-video`, `ltx2-i2v`, `skyreels-v2`
- Talking avatar: `soulx-flashtalk-14b` on Ascend

Current public interfaces are stable enough to build against for generation, validation, model discovery, and artifact export. `image2image` is now a documented public task surface, with `sdxl-base-1.0`, `sdxl-refiner-1.0`, `sd15`, and `sd21` as the recommended starting points. `inpaint`, `edit`, and `video2video` are still evolving.

## Quick Start

Install the package for local development:

```bash
python -m pip install -e '.[dev]'
python -m omnirt --help
pytest
```

Install runtime dependencies if you want to execute real model pipelines:

```bash
python -m pip install -e '.[runtime,dev]'
```

Install docs dependencies if you want to preview or work on the documentation site:

```bash
python -m pip install -e '.[docs]'
```

## First Request

Validate a request before execution:

```bash
omnirt validate \
  --task text2image \
  --model qwen-image \
  --prompt "a poster with a bold title" \
  --backend cpu-stub
```

Run a simple generation:

```bash
omnirt generate \
  --task text2image \
  --model sd15 \
  --prompt "a lighthouse in fog" \
  --backend cuda \
  --preset fast
```

Run a minimal `image2image` request:

```bash
omnirt generate \
  --task image2image \
  --model sdxl-base-1.0 \
  --image input.png \
  --prompt "cinematic concept art" \
  --backend cuda
```

YAML request example:

```yaml
task: text2image
model: flux2.dev
backend: auto
inputs:
  prompt: "a cinematic sci-fi city at sunrise"
config:
  preset: balanced
  width: 1024
  height: 1024
```

Run it with:

```bash
omnirt generate --config request.yaml --json
```

`model_path` can point to either:

- a local Diffusers directory
- a Hugging Face repo id such as `stabilityai/stable-diffusion-xl-base-1.0`

For single-file LoRA weights, use a local `.safetensors` file or an explicit Hugging Face ref such as:

```text
hf://owner/repo/path/to/adapter.safetensors
hf://owner/repo/path/to/adapter.safetensors?revision=main
```

## Python API

```python
from omnirt import generate, requests, validate

req = requests.text2image(
    model="flux2.dev",
    prompt="a cinematic sci-fi city at sunrise",
    width=1024,
    height=1024,
    preset="balanced",
)

validation = validate(req, backend="cpu-stub")
result = generate(req, backend="cuda")
```

Pipeline-style convenience wrapper:

```python
import omnirt

pipe = omnirt.pipeline("sd15", backend="cpu-stub")
validation = pipe.validate(prompt="a lighthouse in fog", preset="fast")
```

The same public API also covers `image2image`:

```python
img2img = requests.image2image(
    model="sdxl-base-1.0",
    image="input.png",
    prompt="cinematic concept art",
    strength=0.8,
)
```

## Validation And Testing

- `pytest tests/unit tests/parity` covers the local contract and metrics layers
- `pytest tests/integration/test_error_paths.py` covers low-memory and bad-weight failures
- CUDA and Ascend smoke tests skip automatically unless the required hardware, runtime packages, and local model directories are available

Real end-to-end generation still depends on the target hardware stack, runtime libraries, and model weights.

## Project Status

- Real hardware smoke coverage is currently confirmed for `sdxl-base-1.0` and `svd-xt` on both CUDA and Ascend
- `image2image` is publicly supported; `sdxl-refiner-1.0` already has CUDA and Ascend smoke entrypoints but still needs repository-tracked verified local model directories
- Additional integrated editing models such as `flux-fill`, `flux-kontext`, and `qwen-image-edit*` already have smoke test entrypoints but still need verified local model directories and hardware validation
- The broader support roadmap is documented in [docs/model-support-roadmap.md](./docs/model-support-roadmap.md) and the live integration snapshot in [docs/support-status.md](./docs/support-status.md)

## Documentation

- Docs site: <https://datascale-ai.github.io/omnirt/>
- English docs: <https://datascale-ai.github.io/omnirt/en/>
- Model onboarding: [docs/model-onboarding.md](./docs/model-onboarding.md)
- Support status: [docs/support-status.md](./docs/support-status.md)
- Model support roadmap: [docs/model-support-roadmap.md](./docs/model-support-roadmap.md)
- Gap roadmap vs vLLM-Omni: [docs/vllm-omni-gap-roadmap.md](./docs/vllm-omni-gap-roadmap.md)
- China deployment: [docs/china-deployment.md](./docs/china-deployment.md)
- Architecture notes: [docs/architecture.md](./docs/architecture.md)
- Service schema: [docs/service-schema.md](./docs/service-schema.md)
- Interface proposal: [docs/interface-improvement-proposal.md](./docs/interface-improvement-proposal.md)

## Utilities

- Prepare offline model snapshots: [scripts/prepare_model_snapshot.py](./scripts/prepare_model_snapshot.py)
- Clone Modelers repositories for offline use: [scripts/prepare_modelers_snapshot.py](./scripts/prepare_modelers_snapshot.py)
- Prepare ModelScope repositories and selected large files for offline use: [scripts/prepare_modelscope_snapshot.py](./scripts/prepare_modelscope_snapshot.py)
- Validate local model directory layout: [scripts/check_model_layout.py](./scripts/check_model_layout.py)
- Sync model directories to servers: [scripts/sync_model_dir.sh](./scripts/sync_model_dir.sh)
