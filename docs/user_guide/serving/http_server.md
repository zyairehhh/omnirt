# HTTP 服务（FastAPI）

OmniRT 的 `server` extra 提供一个开箱即用的 FastAPI 服务，既有原生 `/v1/generate` 端点，也有一组 **OpenAI 兼容**路由（`/v1/images/generations`、`/v1/videos/generations`、`/v1/audio/speech`），便于把已有 OpenAI 客户端无缝切过来。

## 安装与启动

```bash
# 安装 server extra（带来 FastAPI / uvicorn / pydantic / sse-starlette）
python -m pip install -e '.[runtime,server]'

# 最简启动
uvicorn 'omnirt.server.app:create_app' --factory --host 0.0.0.0 --port 8000
```

`create_app()` 接受以下关键参数（可通过环境变量或 `--factory` 自定义包装函数覆盖）：

| 参数 | 默认 | 说明 |
|---|---|---|
| `default_backend` | `"auto"` | 服务默认后端；请求里 `backend` 字段仍可覆盖 |
| `max_concurrency` | `1` | engine 层最大并发；Ascend 上建议保持 `1` |
| `pipeline_cache_size` | `4` | LRU 缓存的模型 pipeline 数；显存敏感时调小 |
| `batch_window_ms` | `0` | 动态 batching 聚合窗口；`0` 关闭 |
| `max_batch_size` | `1` | 动态 batching 最大合批大小 |
| `api_key_file` | `None` | 行分隔的 API key 文件；设了就启用 `ApiKeyMiddleware` |
| `model_aliases_path` | `None` | 模型别名表（YAML/JSON），用于把 OpenAI 风格模型名映射到 OmniRT registry id |

## 路由总览

### 原生端点

| Method | Path | 说明 |
|---|---|---|
| `GET` | `/healthz` | 存活探针 |
| `GET` | `/readyz` | 就绪探针（engine 初始化完成） |
| `POST` | `/v1/generate` | 提交一次同步生成，返回 `GenerateResult` |
| `POST` | `/v1/jobs` | 异步提交，返回 `job_id` |
| `GET` | `/v1/jobs/{job_id}` | 查询 job 状态 / 结果 |
| `DELETE` | `/v1/jobs/{job_id}` | 取消 job |
| `GET` | `/v1/jobs/{job_id}/events` | SSE 流式事件 |

### OpenAI 兼容端点

| Method | Path | 映射到 |
|---|---|---|
| `POST` | `/v1/images/generations` | `text2image` |
| `POST` | `/v1/images/edits` | `inpaint` / `edit` |
| `POST` | `/v1/videos/generations` | `text2video` |
| `POST` | `/v1/audio/speech` | （接入外部 TTS） |

这些端点接受 OpenAI 客户端的 payload，通过 `model_aliases_path` 把模型名（例如 `gpt-image-1`）映射到 OmniRT registry id（例如 `flux2.dev`）。

## 最小请求示例

=== "原生 /v1/generate"

    ```bash
    curl -sS http://localhost:8000/v1/generate \
      -H 'Content-Type: application/json' \
      -d '{
        "task": "text2image",
        "model": "sd15",
        "inputs": {"prompt": "a lighthouse"},
        "config": {"preset": "fast"}
      }'
    ```

=== "OpenAI 兼容"

    ```python
    from openai import OpenAI
    client = OpenAI(base_url="http://localhost:8000/v1", api_key="sk-omnirt-local")
    resp = client.images.generate(
        model="sd15",
        prompt="a lighthouse in fog",
        size="512x512",
    )
    print(resp.data[0].url)
    ```

=== "异步 job"

    ```bash
    # 提交
    curl -sS -X POST http://localhost:8000/v1/jobs \
      -H 'Content-Type: application/json' \
      -d '{"task": "text2video", "model": "wan2.2-t2v-14b",
           "inputs": {"prompt": "..."}, "config": {"num_frames": 81}}'
    # -> {"job_id": "abc123", "status": "queued"}

    # 查询
    curl -sS http://localhost:8000/v1/jobs/abc123

    # 订阅事件（SSE）
    curl -sS -N http://localhost:8000/v1/jobs/abc123/events
    ```

## API Key

```bash
# /etc/omnirt/api-keys.txt —— 每行一个 key
printf 'sk-omnirt-alice\nsk-omnirt-bob\n' > api-keys.txt

# 启动时传入
OMNIRT_API_KEY_FILE=api-keys.txt \
uvicorn 'omnirt.server.app:create_app' --factory \
  --host 0.0.0.0 --port 8000
```

客户端在请求头里带 `Authorization: Bearer <key>`。

## 并发与 batching

- `max_concurrency=N` —— 允许最多 N 个请求同时在 engine 里
- `pipeline_cache_size=K` —— 同时保留 K 个已加载的 pipeline（跨请求复用权重）
- `batch_window_ms=10, max_batch_size=4` —— 聚合窗口内的同任务同模型请求会合批；启用前务必压测，合批对时延敏感

**Ascend 特别建议**：`max_concurrency=1, pipeline_cache_size=1`，避免 NPU 显存碎片化（见 [Ascend 后端](../deployment/ascend.md) 的"显存不释放"条目）。

## 容器化

Docker 镜像模板、k8s 部署片段见 [Docker 部署](../deployment/docker.md)。

## 观测

服务层会把每个请求的阶段计时、峰值显存、后端回退写入结构化日志与 `RunReport`，详见 [遥测](../features/telemetry.md)。
