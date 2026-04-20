# OmniRT 架构说明

`omnirt` 采用组件化运行时结构，在保持公开接口稳定的同时，允许不同后端使用各自的执行策略。

## 分层

1. 用户接口层
   `omnirt.generate(...)` 和 `omnirt` CLI 都会先把输入归一化成 `GenerateRequest`。
2. Pipeline 层
   `BasePipeline` 提供五阶段骨架：`prepare_conditions`、`prepare_latents`、`denoise_loop`、`decode`、`export`。
3. 组件层
   SDXL 和 SVD 等模型由 Diffusers 风格组件拼装而成，例如文本编码器、图像编码器、UNet、VAE 和 scheduler。
4. 后端层
   `BackendRuntime.wrap_module(...)` 会依次尝试 `compile`、`kernel_override`，最后回退到 eager。每一步都会记录到 `RunReport.backend_timeline`。
5. 支撑层
   registry、仅 `safetensors` 权重加载、adapter 加载、结构化日志和 parity helper 都位于这一层。

## 公开契约

- `GenerateRequest` 负责携带 `task`、`model`、`backend`、任务相关 `inputs`、执行 `config` 以及可选 adapters。
- `GenerateResult` 包含导出产物和一份 `RunReport`。
- `RunReport` 记录阶段耗时、解析后的配置、峰值内存、后端回退尝试，以及最终暴露出的错误。

## 模型路径

- `sdxl-base-1.0` maps to `SDXLPipeline`
- `svd-xt` maps to `SVDPipeline`

这两类 pipeline 都会在对象实例里缓存已加载的 Diffusers pipeline，对关键模块只封装一次，并在多次运行之间复用。

## 导出模型

- `text2image` 导出 PNG 产物
- `image2video` 通过 `imageio-ffmpeg` 导出 MP4 产物

## 测试模型

- `tests/unit/` 覆盖契约和基于 fake 的 pipeline 行为
- `tests/parity/` 覆盖指标和阈值 helper
- `tests/integration/` 提供 smoke 和错误路径覆盖；依赖硬件的 case 在缺少前置条件时会自动跳过
