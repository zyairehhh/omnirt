# 遥测

OmniRT 的 `middleware.telemetry` 把每一次 generate 的运行轨迹结构化地记录到两个地方：

- **`RunReport`**（随 `GenerateResult` 返回）— 里面有阶段计时、解析后的 config、峰值内存、`backend_timeline`（compile / kernel_override / fallback）、最终 latent 统计、错误
- **结构化日志**（stdout 或你配置的 sink）— 事件流形式，可接任何日志收集器

## `RunReport` 字段速查

| 字段 | 类型 | 说明 |
|---|---|---|
| `timings` | `dict[stage_name, seconds]` | `prepare_conditions` / `prepare_latents` / `denoise_loop` / `decode` / `export` 五阶段耗时 |
| `memory` | `dict[str, int]` | 峰值显存（`peak_bytes`）、按阶段记录的显存采样 |
| `backend_timeline` | `list[BackendEvent]` | 每一次 compile / kernel_override / fallback 的结果 |
| `config_resolved` | `dict` | 合入 preset 后的最终 config，便于复现 |
| `latent_stats` | `dict` | 最终 latent 的统计量（跨后端 parity 依据） |
| `error` | `ErrorInfo?` | 若失败，字段化的错误信息 |

完整定义见 [src/omnirt/telemetry/report.py](https://github.com/datascale-ai/omnirt/blob/main/src/omnirt/telemetry/report.py)。

## 最简示例

```python
from omnirt import generate
from omnirt.requests import text2image

result = generate(text2image(model="sd15", prompt="a lighthouse"))
report = result.report
print(report.timings)                  # {'prepare_conditions': 0.41, 'denoise_loop': 2.83, ...}
print(report.memory["peak_bytes"])     # 8_765_432_123
for event in report.backend_timeline:
    print(event.stage, event.kind, event.ok, event.reason)
```

## 流式事件

在 engine / 服务层，telemetry 会把同样的阶段事件通过 `attach_stream_events` 推到 `GenerateResult.stream_events` 里；SSE 的 `/v1/jobs/{id}/events` 消费的就是这一流（见 [派发与队列](dispatch_queue.md)）。

## 接到外部观测栈

当前 OmniRT 的遥测是**进程内结构化字段**，没有内置的 Prometheus / OTLP exporter，但数据形式足够用你自己的 adapter 一次性转换：

=== "Prometheus 样例"

    ```python
    from prometheus_client import Histogram
    STAGE_HIST = Histogram("omnirt_stage_seconds", "pipeline stage timing",
                           ["task", "model", "stage"])

    def on_run_complete(req, result):
        for stage, secs in result.report.timings.items():
            STAGE_HIST.labels(task=req.task, model=req.model, stage=stage).observe(secs)
    ```

=== "OTLP 样例"

    ```python
    from opentelemetry import metrics
    meter = metrics.get_meter("omnirt")
    stage_hist = meter.create_histogram("omnirt.stage.seconds")

    def on_run_complete(req, result):
        for stage, secs in result.report.timings.items():
            stage_hist.record(secs, attributes={
                "task": req.task, "model": req.model, "stage": stage})
    ```

## 调试 backend fallback

`RunReport.backend_timeline` 是排查"为什么 Ascend 上这个模型跑得慢"的第一落点：

```python
for ev in result.report.backend_timeline:
    if not ev.ok:
        print(f"[{ev.stage}] {ev.kind} failed: {ev.reason}")
```

每一条条目包含阶段名、动作（`compile` / `kernel_override` / `fallback`）、是否成功、失败原因。详见 [架构说明](../../developer_guide/architecture.md) 的"后端层"章节。

## 相关

- [Python API](../serving/python_api.md) — 从 `GenerateResult` 读取 `report`
- [HTTP 服务](../serving/http_server.md) — SSE 事件订阅
- [派发与队列](dispatch_queue.md) — 跨请求的 engine 级指标
