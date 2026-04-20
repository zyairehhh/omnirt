# OmniRT

OmniRT provides a unified CLI, Python API, validation flow, artifact export contract, and backend abstraction for image, video, and audio-driven avatar models, so different model families can share one runtime interface.

- Built around `GenerateRequest`, `GenerateResult`, and `RunReport`
- One runtime surface across `cuda`, `ascend`, and `cpu-stub` backends
- Supports local model directories, HF repo ids, and single-file weight imports

<div class="intro-actions">
  <a class="md-button md-button--primary" href="getting-started/">Getting Started</a>
  <a class="md-button" href="cli/">CLI Docs</a>
  <a class="md-button" href="python-api/">Python API</a>
</div>

## Current public baseline

| Item | Current status |
|---|---|
| Stable task surfaces | `text2image`, `image2image`, `text2video`, `image2video`, `audio2video` |
| Hardware backends | `cuda`, `ascend` |
| Core normalized objects | `GenerateRequest`, `GenerateResult`, `RunReport` |
| Standard artifact export | `PNG`, `MP4` |

The recommended path is to start with model discovery and request validation, then move into real CUDA or Ascend execution.

## Why OmniRT

<div class="grid cards compact-cards" markdown>

- __Unified contract__

  `GenerateRequest`, `GenerateResult`, and `RunReport` provide one consistent interface across model families.

- __Backend-aware runtime__

  The same request can be validated or executed on `cuda`, `ascend`, or `cpu-stub`, with backend differences absorbed by the runtime layer.

- __Deployment friendly__

  OmniRT is built around local model directories, offline snapshot preparation, error-path validation, and hardware smoke tests.

</div>

## Public task surfaces today

| Task | Description | Representative models | Output |
|---|---|---|---|
| `text2image` | prompt-driven image generation | `sd15`, `sdxl-base-1.0`, `flux2.dev`, `qwen-image` | PNG |
| `image2image` | image-guided image generation | `sd15`, `sd21`, `sdxl-base-1.0`, `sdxl-refiner-1.0` | PNG |
| `text2video` | prompt-driven video generation | `wan2.2-t2v-14b`, `cogvideox-2b`, `hunyuan-video` | MP4 |
| `image2video` | first-frame-guided video generation | `svd`, `svd-xt`, `wan2.2-i2v-14b`, `ltx2-i2v` | MP4 |
| `audio2video` | audio-driven avatar generation | `soulx-flashtalk-14b` | MP4 |

## Model map

The authoritative list of registered models is generated from the live registry and lives at [_generated/models.md](_generated/models.md). Run `omnirt models` locally for the same view.

`soulx-flashtalk-14b` and `image2image` are public today. `inpaint`, `edit`, and `video2video` already have substantial runtime plumbing but are still evolving as public task surfaces.

## Stable boundary

<div class="split-panels">
  <section>
    <h3>Stable today</h3>
    <ul>
      <li>model discovery and request validation</li>
      <li>unified CLI and Python API</li>
      <li>formal public support for `image2image`</li>
      <li>local model directory and offline deployment workflow</li>
      <li>CUDA / Ascend backend abstraction</li>
    </ul>
  </section>
  <section>
    <h3>Still evolving</h3>
    <ul>
      <li>broader public polish for `inpaint`, `edit`, and `video2video`</li>
      <li>stronger model self-discovery</li>
      <li>more granular model parameter help</li>
      <li>a broader cross-backend hardware validation matrix</li>
    </ul>
  </section>
</div>

## Start here

<div class="grid cards compact-cards" markdown>

- __Getting Started__

  Go from installation and `omnirt models` to validation and a first generation request.

  [Open guide](getting-started.md)

- __CLI__

  Learn the three core commands and the split between `inputs` and `config`.

  [Read CLI docs](cli.md)

- __Python API__

  Use typed request helpers, `generate(...)`, `validate(...)`, and the `pipeline(...)` convenience wrapper.

  [Read Python API docs](python-api.md)

- __Architecture and deployment__

  Review the runtime layers, service schema, China deployment workflow, and Ascend backend notes.

  [Read architecture](architecture.md)

- __Support status__

  See which models are integrated, which have real hardware smoke coverage, and which high-priority targets are still pending.

  [Open support status](support-status.md)

- __Compare With vLLM-Omni__

  Review the current gaps against `vLLM-Omni` across serving, distribution, throughput, and omni-stage orchestration, plus a practical roadmap.

  [Open the gap roadmap](vllm-omni-gap-roadmap.md)

</div>
