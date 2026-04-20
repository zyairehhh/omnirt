# 派发与队列

OmniRT 的 `engine` 层在多请求场景下负责排队、并发控制与动态 batching。其骨架由 `omnirt.dispatch` 提供：

| 组件 | 角色 | 代码位置 |
|---|---|---|
| `JobQueue` | 线程安全的作业队列（FIFO + 优先级） | [src/omnirt/dispatch/queue.py](https://github.com/datascale-ai/omnirt/blob/main/src/omnirt/dispatch/queue.py) |
| `Worker` | 从队列拉作业、调用 pipeline、写结果 | [src/omnirt/dispatch/worker.py](https://github.com/datascale-ai/omnirt/blob/main/src/omnirt/dispatch/worker.py) |
| `RequestBatcher` | 在时间窗内合并同任务同模型的请求 | [src/omnirt/dispatch/batcher.py](https://github.com/datascale-ai/omnirt/blob/main/src/omnirt/dispatch/batcher.py) |
| `policies` | 基于 task / backend 的调度策略 | [src/omnirt/dispatch/policies.py](https://github.com/datascale-ai/omnirt/blob/main/src/omnirt/dispatch/policies.py) |

## 从 FastAPI 服务看到的视图

HTTP 服务里，这些组件的配置通过 `create_app()` 的参数暴露（详见 [HTTP 服务](../serving/http_server.md)）：

| 参数 | 含义 |
|---|---|
| `max_concurrency` | 同时在 engine 里执行的请求数上限 |
| `pipeline_cache_size` | LRU 缓存保留的已加载 pipeline 数 |
| `batch_window_ms` | 合批等待窗口（ms）；`0` 关闭合批 |
| `max_batch_size` | 合批大小上限 |

## 典型配置

=== "单卡低延迟"

    ```python
    create_app(
        max_concurrency=1,
        pipeline_cache_size=1,
        batch_window_ms=0,
        max_batch_size=1,
    )
    ```

    优先让单请求尽快跑完。适合**在线交互**或**显存吃紧**的 Ascend 机器。

=== "单卡高吞吐（同一模型）"

    ```python
    create_app(
        max_concurrency=4,
        pipeline_cache_size=1,
        batch_window_ms=20,
        max_batch_size=4,
    )
    ```

    合批内短暂等待，换 2–3× 的吞吐。仅对能合批的任务（例如 `text2image` 同模型）有效。

=== "多模型混合"

    ```python
    create_app(
        max_concurrency=2,
        pipeline_cache_size=4,   # 同时保留 4 个 pipeline
        batch_window_ms=0,
        max_batch_size=1,
    )
    ```

    不合批，但允许不同模型的请求交错执行，减少 pipeline 重载开销。

## 异步 job API

`POST /v1/jobs` 把请求推入队列立即返回 `job_id`，客户端按需轮询 `GET /v1/jobs/{id}` 或订阅 `GET /v1/jobs/{id}/events` 的 SSE 流。适合**视频生成**这类 30s+ 的任务，避免 HTTP 长连接占用服务资源。

完整路由见 [HTTP 服务](../serving/http_server.md)。

## Python 层直接使用

不走 HTTP 也可以直接创建 `OmniEngine`：

```python
from omnirt.engine import OmniEngine
from omnirt.requests import text2image

engine = OmniEngine(max_concurrency=2, pipeline_cache_size=4,
                    batch_window_ms=0, max_batch_size=1)
job = engine.submit(text2image(model="sd15", prompt="..."))
result = engine.wait(job.id)
```

## 已知边界

!!! warning

    - 合批目前只对**同 task + 同 model + 同 backend**的请求生效；混合请求自动退化到逐条执行
    - Ascend 上强烈建议 `max_concurrency=1`：NPU 显存碎片在多并发下不会及时回收（见 [Ascend 后端](../deployment/ascend.md)）
    - `pipeline_cache_size` 直接决定常驻显存；视频模型（Wan2.2、Hunyuan）每个占 20–40 GB，别盲目调大

## 相关

- [HTTP 服务](../serving/http_server.md) — 服务化入口与 OpenAI 兼容路由
- [遥测](telemetry.md) — engine 级指标
- [架构说明](../../developer_guide/architecture.md) — `BasePipeline` 五阶段与后端层关系
