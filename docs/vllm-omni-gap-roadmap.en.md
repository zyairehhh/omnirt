# OmniRT Gap Roadmap Relative to vLLM-Omni

This document summarizes the major capability gaps between `OmniRT` and `vLLM-Omni`, then turns that comparison into a practical roadmap for this repository.

The comparison is based on public materials available as of `2026-04-20` and the current repository state. It does not imply the two projects must converge into the exact same product.

## Short conclusion

Today, `OmniRT` behaves more like a unified offline generation runtime. Its strengths are:

- a normalized request contract
- CUDA / Ascend dual-backend support
- local-model and offline deployment workflows
- fast integration of Diffusers-style image and video families

By contrast, `vLLM-Omni` is positioned more like a production-grade omni-modality inference and serving engine. Its strengths are:

- online serving
- high-throughput scheduling
- distributed execution
- heterogeneous multi-stage orchestration
- fuller omni-modality input and output support

## Current comparison

| Capability area | OmniRT today | vLLM-Omni direction | Gap |
|---|---|---|---|
| CLI / Python API | present | present | small |
| Offline local execution | present | present | small |
| Diffusers image / video integration | present and broad | present | moderate |
| OpenAI-compatible serving | not built in | explicitly supported | major |
| Online HTTP server | not built in | core serving feature | major |
| Async scheduling / request queue | no general implementation | strong engine direction | major |
| Dynamic batching / throughput optimization | minimal | core value proposition | major |
| General distributed inference | only model-specific partial support | official distributed direction | major |
| Multi-stage orchestration | no general stage layer | multi-stage / disaggregation is central | major |
| Unified omni request flow | still generation-first | omni-modality-first | major |
| Real-time streaming output | minimal | important direction | major |
| Benchmark / profiling | basic run report only | actively expanded | medium to large |
| Quantization / cache acceleration | scattered, model-specific | actively expanded | medium to large |
| Platform coverage | `cuda` / `ascend` / `cpu-stub` | `CUDA / ROCm / NPU / XPU` | moderate |

## The 5 biggest gaps

### 1. No serving layer

The repository still centers on synchronous `generate(...)` execution and CLI commands. There is no built-in HTTP server and no OpenAI-compatible API surface.

That means OmniRT is currently better suited for:

- local execution
- smoke tests
- direct Python embedding

and less suited for:

- acting as an online inference gateway
- multi-tenant request handling
- unified auth, queueing, and rate limiting

### 2. No general scheduler layer

The execution model is still essentially:

`request -> resolve backend -> instantiate pipeline -> run`

That keeps the runtime understandable, but it also means there is no general:

- async job queue
- worker pool
- dynamic batching
- priority handling
- mixed workload scheduling

### 3. No general distributed execution framework

The general image and video pipelines are still built around one process, one runtime instance, and one direct device placement path.

There is one important exception: `soulx-flashtalk-14b` already supports `torchrun` for an external script workflow. But that is model-specific glue, not a general OmniRT distributed runtime.

### 4. No heterogeneous multi-stage orchestration

`vLLM-Omni` explicitly emphasizes:

- modality encoders
- an LLM core
- modality generators
- pipelined stage execution

OmniRT is still closer to “one pipeline owns the full request lifecycle”, and does not yet have a reusable abstraction for:

- encoder stages
- prefill / reasoning stages
- generator stages
- stage-level scheduling and placement

### 5. No production-grade performance layer

OmniRT already has some solid foundations:

- run reports
- backend timelines
- peak memory reporting
- parity metrics

But it still lacks:

- a benchmark CLI
- a profiling workflow
- a general quantization configuration layer
- diffusion cache acceleration
- cross-device placement strategies
- a streaming output contract

## Recommended roadmap

### Phase 1: turn OmniRT into a serviceable single-node engine

Goal: move from “offline runtime” to “deployable service”.

Recommended deliverables:

1. add `omnirt serve`
2. add a basic HTTP API
3. add a minimal OpenAI-compatible subset
4. add an async job queue plus job ids
5. add status, cancel, and health endpoints

Exit criteria:

- `text2image`, `image2video`, and `audio2video` can run as online services
- basic concurrency is possible without wrapping the Python API in custom scripts

### Phase 2: add single-node multi-GPU execution and throughput scheduling

Goal: move from “it runs” toward “it behaves like an inference engine”.

Recommended deliverables:

1. add a general launcher abstraction, not just for `FlashTalk`
2. wrap `accelerate launch` and `torchrun`
3. add a single-node multi-GPU worker pool
4. add request batching and prompt grouping
5. add `device_map` and component placement across GPUs
6. add a benchmark CLI and throughput metrics

Exit criteria:

- at least one Diffusers image family supports single-node multi-GPU execution
- at least one video family supports throughput-oriented batched execution

### Phase 3: add multi-stage pipeline orchestration

Goal: move closer to a true omni runtime.

Recommended deliverables:

1. introduce a stage abstraction
2. split encoder and generator stages into independent execution units
3. allow one request to chain multiple stages
4. support stage-level resource and device allocation
5. support intermediate-result caching and reuse

Exit criteria:

- a request can explicitly traverse multiple stages
- audio, image, video, and text-related modules can be composed instead of being fully bound to one pipeline class

### Phase 4: add production-grade distributed execution and performance engineering

Goal: approach the operating range of a real omni inference engine.

Recommended deliverables:

1. multi-node worker / controller architecture
2. distributed queueing and scheduling
3. streaming output protocols
4. benchmark / profiling / telemetry dashboards
5. general quantization and cache optimization
6. broader platform coverage such as `rocm` and `xpu`

Exit criteria:

- multi-node deployment becomes an officially supported path
- observability, throughput, and resource scheduling no longer depend on ad-hoc external scripts

## The 8 best next investments

If the goal is return on engineering effort, the best next steps are:

1. `omnirt serve`
2. a minimal HTTP / OpenAI-compatible surface
3. an async job queue
4. a general single-node multi-GPU launcher abstraction
5. `device_map` and multi-GPU placement
6. a benchmark CLI
7. a stage abstraction
8. a controller / worker prototype

## What not to overbuild first

These are valuable, but not good first moves:

- jumping straight to full multi-node distributed execution
- trying to cover every hardware platform at the start
- leading with complex cache optimization
- forcing early alignment with every internal vLLM abstraction

A more stable order is:

- first add serving
- then add single-node throughput features
- then add stage orchestration
- finally add distributed execution

## Recommended repository additions

If the roadmap above is adopted, these top-level modules would be a good fit:

- `src/omnirt/server/`
- `src/omnirt/engine/`
- `src/omnirt/dispatch/`
- `src/omnirt/stages/`
- `src/omnirt/bench/`

The existing `models/` and `backends/` layout should remain mostly stable. The next wave of work belongs primarily in the scheduling and serving layers above the current pipelines.

## References

- vLLM-Omni GitHub:
  <https://github.com/vllm-project/vllm-omni>
- vLLM-Omni launch blog:
  <https://vllm.ai/blog/vllm-omni>
- vLLM-Omni API docs:
  <https://docs.vllm.ai/projects/vllm-omni/en/stable/api/vllm_omni/>
