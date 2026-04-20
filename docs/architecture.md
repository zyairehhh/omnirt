# OmniRT 架构说明

`omnirt` 采用组件化运行时结构，在保持公开接口稳定的同时，允许不同后端使用各自的执行策略。

## 分层

1. **用户接口层** — [src/omnirt/api.py](https://github.com/datascale-ai/omnirt/blob/main/src/omnirt/api.py) + [src/omnirt/cli/main.py](https://github.com/datascale-ai/omnirt/blob/main/src/omnirt/cli/main.py)
   `omnirt.generate(...)`、`omnirt.validate(...)`、`omnirt.pipeline(...)` 和 `omnirt` CLI 都会把输入归一化成 `GenerateRequest`，再下发给 pipeline 层。
2. **Pipeline 层** — [src/omnirt/core/base_pipeline.py](https://github.com/datascale-ai/omnirt/blob/main/src/omnirt/core/base_pipeline.py)
   `BasePipeline` 提供五阶段骨架：`prepare_conditions`、`prepare_latents`、`denoise_loop`、`decode`、`export`。每阶段都会被 telemetry 层打点计时。
3. **组件层** — [src/omnirt/models/](https://github.com/datascale-ai/omnirt/tree/main/src/omnirt/models)
   具体模型家族（SDXL、SVD、Flux2、Wan2.2、Qwen-Image、CogVideoX、HunyuanVideo 等）各自实现一个 `BasePipeline` 子类，通过 `@register_model` 装饰器挂进 registry。完整清单见 [_generated/models.md](_generated/models.md)。
4. **Scheduler 层** — [src/omnirt/schedulers/](https://github.com/datascale-ai/omnirt/tree/main/src/omnirt/schedulers)
   薄封装常用调度器（`euler-discrete`、`euler-ancestral`、`ddim`、`dpm-solver`、`dpm-solver-karras`），由 `SCHEDULER_REGISTRY` 和 `build_scheduler(config)` 统一分发，pipeline 不直接依赖 Diffusers 具体调度器类。
5. **后端层** — [src/omnirt/backends/](https://github.com/datascale-ai/omnirt/tree/main/src/omnirt/backends)
   `BackendRuntime.wrap_module(...)` 依次尝试 `compile`、`kernel_override`，最后回退到 eager。每一步都被记录到 `RunReport.backend_timeline`。当前实现：`CudaBackend`、`AscendBackend`、`CpuStubBackend`。
6. **Telemetry 层** — [src/omnirt/telemetry/](https://github.com/datascale-ai/omnirt/tree/main/src/omnirt/telemetry)
   `log.py` 负责结构化日志；`report.py` 负责 `RunReport` 构建（阶段耗时、峰值显存、后端回退、latent 统计）。
7. **支撑设施** — [src/omnirt/core/](https://github.com/datascale-ai/omnirt/tree/main/src/omnirt/core)
   registry、仅 `safetensors` 权重加载、adapter 加载、presets、validation、parity helper 都位于这一层。

## 公开契约

- `GenerateRequest` 负责携带 `task`、`model`、`backend`、任务相关 `inputs`、执行 `config` 以及可选 adapters。
- `GenerateResult` 包含导出产物和一份 `RunReport`。
- `RunReport` 记录阶段耗时、解析后的配置、峰值内存、后端回退尝试、终端 latent 统计（用于跨后端 parity）以及最终暴露出的错误。
- 详细字段见 [service-schema.md](service-schema.md)。

## Registry 与 alias

同一个 pipeline 类可以通过多次 `@register_model(...)` 暴露多个 id，这就是 `flux2.dev` / `flux2-dev` 这类别名的由来；在 `ModelCapabilities.alias_of` 上标记，`omnirt models --format markdown` 会把规范 id 与别名分开渲染。接入细节见 [model-onboarding.md](model-onboarding.md)。

## Preset

所有 pipeline 共享同一组 preset (`fast` / `balanced` / `quality` / `low-vram`)，在 `prepare_conditions` 阶段被合并进 `config`；具体 preset 对不同 task / model 产生的变化见 [presets.md](presets.md)，源头在 [src/omnirt/core/presets.py](https://github.com/datascale-ai/omnirt/blob/main/src/omnirt/core/presets.py) 的 `_BASE_PRESETS` / `_TASK_PRESETS` / `_MODEL_PRESETS`。

## 产物导出

- `text2image` 导出 PNG 产物（Pillow 写盘）。
- `text2video` / `image2video` / `audio2video` 通过 `imageio-ffmpeg` 封装 MP4（见 [src/omnirt/core/media.py](https://github.com/datascale-ai/omnirt/blob/main/src/omnirt/core/media.py)）。

## 测试分层

- `tests/unit/` 覆盖契约、registry、CLI、pipeline 行为（用 fake runtime + fake Diffusers）。
- `tests/parity/` 覆盖 latent 统计与图像/视频指标 helper。
- `tests/integration/` 提供 CUDA / Ascend smoke 与错误路径覆盖；依赖硬件的 case 在缺少前置条件时会自动跳过。
