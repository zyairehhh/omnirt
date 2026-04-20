# Ascend 后端说明

Ascend 后端与 CUDA 共享同一套外部契约，但在编译路径上会更保守一些。

## 执行模型

- 后端名称：`ascend`
- 设备名称：`npu`
- 编译尝试：可用时调用 `torch_npu.npu.graph_mode()`
- 回退行为：如果 graph-mode 初始化失败或模块无法编译，运行时会记录失败并保留 eager 模块

## 当前约束

- 请求可以通过 `--backend ascend` 显式指定 Ascend
- CUDA 与 Ascend 使用同一套 `GenerateRequest` schema
- capability reporting 会暴露 dtype 选项和 compile 可用性
- 后端回退尝试会保留在 `RunReport.backend_timeline` 中

## 校验流程

仓库已经提供 Ascend smoke tests，但只有在满足以下条件时才会运行：

- `torch_npu` is installed
- Diffusers runtime dependencies are installed
- model sources are provided through `OMNIRT_SDXL_MODEL_SOURCE` and `OMNIRT_SVD_MODEL_SOURCE`
- the tests execute on an Ascend-capable host

如果这些前置条件不满足，测试会直接 skip，而不是产生噪声式失败。
