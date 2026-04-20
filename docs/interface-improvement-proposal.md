# OmniRT 接口改进提案

本文档从用户体验角度给出一份务实的 OmniRT 接口演进路线图。

当前接口已经足够支撑内部工程使用和早期采用者，但对于更广泛的开源用户来说，还不够完整，也不够容易被发现与理解。

## 当前评估

当前做得好的地方：

- one obvious top-level entrypoint: `omnirt generate`
- one unified request contract: `GenerateRequest`
- one unified result contract: `GenerateResult` plus `RunReport`
- backend, artifact export, and telemetry are already normalized across models

对用户来说依然薄弱的地方：

- users must already know which `model` ids exist
- users cannot easily discover which parameters a model accepts
- `inputs` versus `config` is not self-explanatory
- schema validation is mostly runtime-only
- task coverage is narrower than the roadmap
- CLI is execution-oriented, not workflow-oriented

## 设计目标

1. 保持 OmniRT 在不同模型家族之间的公开接口稳定。
2. 内部继续基于 Diffusers，但不要把原始 Diffusers API 直接强加给用户。
3. 让 CLI 和 Python API 具备自发现能力。
4. 尽量用显式校验替代运行时惊喜。
5. 让未来的 `image2image`、`inpaint`、`edit`、`video2video` 能自然纳入同一套契约体系。

## 非目标

- 不替换 Diffusers 作为内部模型执行层。
- 不把每个模型上游的原始参数面完整暴露给用户。
- 不追求在最外层 API 上与 Diffusers 完全兼容。

## P0

这些是价值最高、应最先落地的改进。

### 1. 模型发现命令

新增：

- `omnirt models`
- `omnirt models <model-id>`

期望行为：

- 列出支持的模型 id
- 展示任务、默认后端、最低显存提示和当前成熟度
- 展示模型级支持参数及默认值
- 展示该模型的示例命令

为什么重要：

- 它解决了当前最大的可用性缺口
- 它让 CLI 本身变得更可理解

### 2. 一等请求校验能力

新增：

- `omnirt validate --config request.yaml`
- `omnirt generate --dry-run`

期望行为：

- 校验任务与模型是否兼容
- 校验必填输入
- 校验支持的 config key
- 在不执行生成的情况下打印解析后的默认值

为什么重要：

- 用户可以在长耗时或高成本执行前发现错误
- 这对视频模型和远程模型目录场景尤其重要

### 3. 明确参数归属规则

需要文档化并强制执行一个简单规则：

- `inputs`: semantic generation inputs such as `prompt`, `negative_prompt`, `image`, `mask`, `control_image`
- `config`: execution settings such as `num_inference_steps`, `guidance_scale`, `height`, `width`, `dtype`, `seed`, `output_dir`

为什么重要：

- 当前拆分方式本身合理，但并不直观
- 固定规则会让请求结构更容易讲解、校验和扩展

### 4. 更好的执行摘要

成功执行后，CLI 输出应进一步增强：

- artifact path summary
- resolved model path
- resolved backend
- key generation settings
- a compact human-readable mode in addition to JSON

为什么重要：

- 当前输出在结构上是可用的，但不够适合快速扫读

### 5. 更强的面向用户错误提示

错误信息至少应包含：

- what was wrong
- what was expected
- a valid example
- suggested nearby models or tasks where possible

示例：

与其只说模型不支持 `text2image`，不如同时提示它实际支持的任务以及替代命令。

## P1

这些改进会让接口显著更完整。

### 1. 按任务区分的 typed schema

在内部引入按任务区分的 typed request 变体，例如：

- `TextToImageRequest`
- `ImageToVideoRequest`
- `TextToVideoRequest`

These can still serialize into the current `GenerateRequest` envelope if desired.

为什么重要：

- IDE 提示更好
- 校验更清晰
- 对允许字段的歧义更少

### 2. 模型能力元数据

扩展 registry，让每个模型都暴露以下能力元数据：

- supported tasks
- required inputs
- optional inputs
- supported config keys
- scheduler support
- adapter support
- output artifact type

这些元数据应同时驱动校验逻辑和模型帮助命令。

### 3. Preset

增加命名 preset，例如：

- `fast`
- `balanced`
- `quality`
- `low-vram`

为什么重要：

- 大多数用户并不希望每次都手动调 `guidance_scale`、`steps` 和 `dtype`

### 4. 补齐缺失的任务面

接口应逐步扩展到：

- `image2image`
- `inpaint`
- `edit`
- `video2video`

这些能力不应以拼接式方式硬塞进来，而应自然融入“任务 + 模型”的同一套契约。

### 5. 更好的 Python 易用性

增加类似这样的 helper constructor：

```python
from omnirt import requests

req = requests.text2image(
    model="flux2.dev",
    prompt="a cinematic city at sunrise",
    width=1024,
    height=1024,
    guidance_scale=2.5,
)
```

为什么重要：

- 当前 dataclass API 虽然可用，但不够顺手

## P2

这些改进有价值，但不紧急。

### 1. 可选的 Diffusers 风格便捷封装

可以提供类似这样的便捷层：

```python
pipe = omnirt.pipeline("flux2.dev")
image = pipe(prompt="hello", num_inference_steps=30)
```

它应是可选语法糖，而不是主 API。

### 2. OpenAPI 或服务协议

如果 OmniRT 预计会作为服务后端使用，就需要定义稳定的服务协议和版本演进方案。

### 3. 交互式 CLI 指导

长期来看，可以增加如下引导式帮助：

- 缺失必填输入时给出建议
- 按任务推荐模型
- 根据可用显存推荐 preset

## 推荐契约方向

推荐方向是：

- 保留顶层 `GenerateRequest` 包结构
- 强化其校验能力
- 增加模型能力元数据
- 增加按任务拆分的 typed helper
- 让模型内部实现继续自由适配 Diffusers

这样既能保住 OmniRT 的运行时定位，也能修复最显著的可用性缺口。

## 建议实施顺序

1. 为 registry 增加模型元数据。
2. 实现 `omnirt models` 和 `omnirt models <id>`。
3. 实现 `validate` 和 `--dry-run`。
4. 收紧现有任务的 schema 校验。
5. 改进 CLI 成功输出和错误输出。
6. 在 Python 侧增加 typed task helper。
7. 把公开任务面扩展到编辑和转换工作流。

## 验收标准

当一个新用户能够做到以下几点时，就可以认为接口有了实质提升：

1. 不读源码也能发现支持的模型
2. 不靠试错就能知道某个模型接受哪些参数
3. 在执行前先校验请求
4. 读 CLI 输出时能立刻理解发生了什么
5. 从一个模型家族切到另一个模型家族时，不需要重学整套请求结构

## 推荐的第一期实现切片

如果只有一次较短迭代，最值得先做的切片是：

- 补齐 registry 元数据
- 落地 `omnirt models`
- 落地 `omnirt validate`
- 给现有 CLI 输出增加更强的默认解释力

- registry capability metadata
- `omnirt models`
- `omnirt validate`
- clearer errors

This would deliver the largest usability improvement for the least surface-area change.
