# OmniRT 对标 vLLM-Omni 的能力缺口 Roadmap

本文档整理 `OmniRT` 相对 `vLLM-Omni` 的关键能力差距，并给出一个更适合当前仓库演进节奏的补齐路线。

这里的比较基于 `2026-04-20` 可查到的公开资料和当前仓库实现现状，不代表两者必须收敛为同一种产品形态。

## 一句话结论

当前 `OmniRT` 更像统一的离线生成运行时，强项在于：

- 统一的请求协议
- CUDA / Ascend 双后端
- 本地模型目录与离线部署
- 图像 / 视频 Diffusers 家族快速接入

相对地，`vLLM-Omni` 更像生产级全模态推理与服务引擎，强项在于：

- 在线 serving
- 高吞吐调度
- 分布式执行
- 多阶段异构流水线
- 更完整的全模态输入输出

## 现状对比

| 能力维度 | OmniRT 当前状态 | vLLM-Omni 官方方向 | 差距判断 |
|---|---|---|---|
| CLI / Python API | 已有统一接口 | 已有 | 差距不大 |
| 本地离线运行 | 已有 | 已有 | 差距不大 |
| 图像 / 视频 Diffusers 接入 | 已有，覆盖较广 | 已有 | 差距中等 |
| OpenAI-compatible serving | 未内建 | 官方明确支持 | 明显缺口 |
| 在线 HTTP 服务 | 未内建 | 核心能力之一 | 明显缺口 |
| 异步调度 / 请求队列 | 没有通用实现 | 有明确服务引擎方向 | 明显缺口 |
| 动态批处理 / 吞吐优化 | 基本没有 | 核心优势之一 | 明显缺口 |
| 通用分布式推理 | 仅个别模型局部支持 | 官方强调 distributed execution | 明显缺口 |
| 多阶段解耦流水线 | 没有通用 stage 编排层 | 官方以 multi-stage / disaggregation 为核心 | 明显缺口 |
| 全模态统一请求 | 仍以生成任务为主 | 面向 omni-modality | 明显缺口 |
| 实时流式输出 | 基本没有 | 重要方向 | 明显缺口 |
| Benchmark / profiling | 只有基础 run report | 官方持续加强 | 中到大缺口 |
| 量化 / cache acceleration | 零散、模型专用 | 官方持续加强 | 中到大缺口 |
| 平台覆盖 | `cuda` / `ascend` / `cpu-stub` | `CUDA / ROCm / NPU / XPU` | 中等缺口 |

## 当前最核心的 5 个差距

### 1. 缺少服务层

当前仓库主入口仍是同步 `generate(...)` 调用和 CLI 执行，没有内建 HTTP server，也没有 OpenAI-compatible 接口。

这意味着：

- 更适合本地执行和 smoke
- 不适合直接作为在线推理网关
- 没有统一在线鉴权、并发、排队与限流边界

### 2. 缺少通用调度层

当前执行模型仍然是：

`request -> resolve backend -> instantiate pipeline -> run`

这让代码结构比较清晰，但也意味着还没有：

- 异步任务队列
- 并发 worker
- 动态 batching
- 任务优先级
- 长短请求混部调度

### 3. 缺少通用分布式执行框架

当前通用图像 / 视频 pipeline 还是单进程、单 runtime、单设备迁移模型的方式。

例外是 `soulx-flashtalk-14b` 已支持 `torchrun` 启动外部脚本，但那属于单模型特化，不是 `OmniRT` 的通用分布式运行时。

### 4. 缺少多阶段异构编排

`vLLM-Omni` 明确强调：

- modality encoders
- LLM core
- modality generators
- pipelined stage execution

而 `OmniRT` 当前更像“单个 pipeline 负责完整推理过程”，还没有统一的：

- encoder stage
- prefill / reasoning stage
- generator stage
- stage 级资源调度

### 5. 缺少生产级性能工程

当前仓库已经有：

- run report
- backend timeline
- peak memory
- parity metrics

但还缺：

- benchmark CLI
- profiling 工作流
- 通用量化配置
- 扩散 cache 加速
- 多卡放置策略
- streaming 输出协议

## 建议路线

### Phase 1: 先把 OmniRT 做成可服务的单机引擎

目标：从“离线运行时”升级到“可部署服务”。

建议交付：

1. 新增 `omnirt serve`
2. 提供基础 HTTP API
3. 提供 OpenAI-compatible 最小子集
4. 增加异步任务队列和 job id
5. 补充状态查询、取消、健康检查接口

完成标志：

- 可以把 `text2image` / `image2video` / `audio2video` 作为在线服务运行
- 支持基础并发，不需要每次由外层脚本直接调 Python API

### Phase 2: 做单机多卡与高吞吐调度

目标：把运行时从“能跑”推进到“更像推理引擎”。

建议交付：

1. 通用 `launcher` 抽象，不只服务于 `FlashTalk`
2. `accelerate launch` / `torchrun` 的统一封装
3. 单机多卡 worker 池
4. 请求 batching 与 prompt grouping
5. `device_map` 和模型组件跨卡放置
6. benchmark CLI 与吞吐指标

完成标志：

- 至少一类 Diffusers 图像模型支持单机多卡
- 至少一类视频模型支持吞吐导向的批处理执行

### Phase 3: 做多阶段 pipeline 编排

目标：向真正的 omni runtime 靠拢。

建议交付：

1. 引入 stage 抽象
2. 将 encoder / generator 拆成独立执行单元
3. 允许单请求串接多个 stage
4. 支持 stage 级资源和设备分配
5. 支持中间结果缓存和复用

完成标志：

- 一个请求可以显式经过多个 stage
- 音频、图像、视频、文本相关模块可以组合执行，而不是强绑定在单个 pipeline 类里

### Phase 4: 做生产级分布式与性能优化

目标：进入与 vLLM-Omni 更接近的生产引擎区间。

建议交付：

1. 多机 worker / controller 架构
2. 分布式队列和调度
3. 流式输出协议
4. benchmark / profiling / telemetry 面板
5. 通用量化与缓存优化
6. 更完善的平台扩展层，如 `rocm` / `xpu`

完成标志：

- 多机部署成为明确支持路径
- 可观测性、吞吐和资源调度不再依赖外部脚本拼装

## 当前最值得先做的 8 项

如果只按投资回报看，我建议按下面顺序推进：

1. `omnirt serve`
2. HTTP / OpenAI-compatible 最小接口
3. 异步 job queue
4. 单机多卡 launcher 抽象
5. `device_map` 与多卡放置
6. benchmark CLI
7. stage 抽象
8. 多机 controller / worker 雏形

## 不建议一开始就重做的部分

以下事情价值高，但不适合作为第一步：

- 直接追求完整多机分布式
- 一开始就覆盖所有平台后端
- 提前做复杂缓存优化
- 过早对齐所有 vLLM 内部抽象

更稳妥的策略是：

- 先补服务层
- 再补单机吞吐
- 再补 stage 编排
- 最后补分布式

## 对仓库结构的直接建议

如果开始推进上面的路线，建议新增这些目录或模块：

- `src/omnirt/server/`
- `src/omnirt/engine/`
- `src/omnirt/dispatch/`
- `src/omnirt/stages/`
- `src/omnirt/bench/`

建议保留现有 `models/` 和 `backends/` 结构不动，把新增能力更多放在“上层调度和服务层”，避免一次性大拆底层 pipeline。

## 参考资料

- vLLM-Omni GitHub:
  <https://github.com/vllm-project/vllm-omni>
- vLLM-Omni 发布博客:
  <https://vllm.ai/blog/vllm-omni>
- vLLM-Omni API 文档:
  <https://docs.vllm.ai/projects/vllm-omni/en/stable/api/vllm_omni/>

