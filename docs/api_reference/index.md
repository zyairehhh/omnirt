# API Reference

本节是 OmniRT Python 接口的**自动生成参考**，由 `mkdocstrings` 直接从 `src/omnirt/` 的 docstring 派生。需要任务导向的示例时，请回到 [Python API 使用指南](../user_guide/serving/python_api.md)。

| 模块 | 内容 |
|---|---|
| [`omnirt`](top_level.md) | 顶层函数 `generate` / `validate` / `pipeline` / `list_available_models` / `describe_model` |
| [`omnirt.requests`](requests.md) | Typed request builders（`text2image`、`image2image`、`text2video`、`image2video`、`audio2video`、`inpaint`、`edit`） |
| [`omnirt.core.types`](types.md) | `GenerateRequest` / `GenerateResult` / `RunReport` 等核心数据类 |
| [`omnirt.core.registry`](registry.md) | 模型注册与发现 |
| [`omnirt.core.presets`](presets.md) | Preset 合并 (`resolve_preset` / `available_presets`) |
| [`omnirt.core.validation`](validation.md) | 请求校验入口 |
| [`omnirt.engine`](engine.md) | `OmniEngine` 异步分发 |

!!! info "生成机制"
    本节依赖源代码中的 docstring。改动公开 API 时，别忘了同步更新 docstring。
