# OmniRT

[中文](./README.zh-CN.md)

Open image and video generation runtime with CUDA and Ascend backends.

## Overview

OmniRT is a Diffusers-backed runtime layer that gives image and video models a unified CLI, Python API, validation flow, artifact export contract, and backend abstraction.

Current public task surfaces:

- `text2image`
- `text2video`
- `image2video`
- `audio2video`

Current public interface highlights:

- `omnirt generate`, `omnirt validate`, `omnirt models`
- typed request helpers and a convenience `pipeline(...)` API in Python
- normalized `GenerateRequest` / `GenerateResult` / `RunReport`
- PNG and MP4 artifact export
- model registry metadata, presets, validation, and backend selection

## Supported Models

Representative supported model families today:

- Stable Diffusion: `sd15`, `sd21`, `sdxl-base-1.0`, `sdxl-turbo`, `sd3-medium`, `sd3.5-large`, `sd3.5-large-turbo`
- Flux: `flux-dev`, `flux-schnell`, `flux2.dev`, `flux2-dev`
- Generalist image: `glm-image`, `hunyuan-image-2.1`, `omnigen`, `qwen-image`, `sana-1.6b`, `ovis-image`, `hidream-i1`
- Video: `svd`, `svd-xt`, `cogvideox-2b`, `cogvideox-5b`, `kandinsky5-t2v`, `kandinsky5-i2v`, `wan2.1-*`, `wan2.2-*`, `hunyuan-video`, `hunyuan-video-1.5-*`, `helios-*`, `sana-video`, `ltx-video`, `ltx2-i2v`
- Talking avatar: `soulx-flashtalk-14b` on Ascend, backed by the external `SoulX-FlashTalk-Ascend` checkout

Use the CLI to inspect the exact live registry:

```bash
omnirt models
omnirt models flux2.dev
```

## Quickstart

```bash
python3 -m pip install -e .[dev]
python3 -m omnirt --help
pytest
```

For runtime model execution support, install runtime extras too:

```bash
python3 -m pip install -e '.[runtime,dev]'
```

Real CUDA and Ascend end-to-end generation still requires the corresponding hardware, runtime libraries, and model weights.

## CLI

YAML request:

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

Run it:

```bash
omnirt generate --config request.yaml --json
```

Direct flags:

```bash
omnirt generate \
  --task text2image \
  --model sd15 \
  --prompt "a lighthouse in fog" \
  --backend cuda \
  --preset fast
```

Validate without executing:

```bash
omnirt validate \
  --task text2image \
  --model qwen-image \
  --prompt "一张带有中文标题的电影海报" \
  --backend cpu-stub

omnirt generate \
  --task text2video \
  --model wan2.2-t2v-14b \
  --prompt "a glass whale gliding over a moonlit harbor" \
  --preset fast \
  --dry-run
```

Video examples:

```bash
omnirt generate \
  --task image2video \
  --model svd-xt \
  --image input.png \
  --backend cuda \
  --num-frames 25 \
  --fps 7 \
  --frame-bucket 127 \
  --decode-chunk-size 8

omnirt generate \
  --task text2video \
  --model cogvideox-2b \
  --prompt "a wooden toy ship gliding over a plush blue carpet" \
  --backend cuda \
  --num-frames 81 \
  --fps 16

omnirt generate \
  --task audio2video \
  --model soulx-flashtalk-14b \
  --image speaker.png \
  --audio voice.wav \
  --prompt "A person is talking." \
  --backend ascend \
  --repo-path /home/<user>/SoulX-FlashTalk
```

Available presets:

- `fast`
- `balanced`
- `quality`
- `low-vram`

## Python API

Typed request helpers:

```python
from omnirt import requests, validate, generate

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

## Current Scope

What is already public and stable enough to build against:

- unified image, video, and audio-driven avatar generation requests
- validation and model discovery
- model-family registry metadata
- code and CLI entrypoints

What is still intentionally not exposed as a full public task surface yet:

- `image2image`
- `inpaint`
- `edit`
- `video2video`

Some underlying model families are already scaffolded for future expansion, but these task surfaces are not yet first-class OmniRT APIs.

## Validation and Testing

- `pytest tests/unit tests/parity` exercises the local contract and metrics layer
- `pytest tests/integration/test_error_paths.py` checks low-memory and bad-weight failures
- CUDA and Ascend integration tests automatically skip unless hardware, runtime packages, and local model directories are available

The implementation target and remaining hardware validation details are tracked in [PLAN.md](./PLAN.md).

## Docs

- Model onboarding: [docs/model-onboarding.md](./docs/model-onboarding.md)
- Model support roadmap: [docs/model-support-roadmap.md](./docs/model-support-roadmap.md)
- China deployment: [docs/china-deployment.md](./docs/china-deployment.md)
- Architecture notes: [docs/architecture.md](./docs/architecture.md)
- Service schema: [docs/service-schema.md](./docs/service-schema.md)
- Interface proposal: [docs/interface-improvement-proposal.md](./docs/interface-improvement-proposal.md)

## Utilities

- Prepare offline model snapshots: [scripts/prepare_model_snapshot.py](/Users/<user>/Desktop/code/opensource/omnirt/scripts/prepare_model_snapshot.py)
- Clone Modelers repositories for offline use: [scripts/prepare_modelers_snapshot.py](/Users/<user>/Desktop/code/opensource/omnirt/scripts/prepare_modelers_snapshot.py)
- Validate local model directory layout: [scripts/check_model_layout.py](/Users/<user>/Desktop/code/opensource/omnirt/scripts/check_model_layout.py)
- Sync model directories to servers: [scripts/sync_model_dir.sh](/Users/<user>/Desktop/code/opensource/omnirt/scripts/sync_model_dir.sh)
