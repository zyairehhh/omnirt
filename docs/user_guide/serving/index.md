# 运行入口

OmniRT 提供三种入口，共享同一份 `GenerateRequest` 契约：

| 入口 | 适合 | 页面 |
|---|---|---|
| **Python API** | 嵌入已有 Python 应用、notebook 实验 | [Python API](python_api.md) |
| **CLI** | 脚本化批处理、一次性校验 / 生成 | [CLI](cli.md) |
| **HTTP 服务** | 微服务、多租户、OpenAI 兼容 API | [HTTP 服务](http_server.md) |

!!! tip "建议顺序"
    先在 Python 或 CLI 下跑通 `validate` + `generate` 确认契约，再上 HTTP 服务做并发 / batching / 服务协议调优。
