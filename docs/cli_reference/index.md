# CLI Reference

`omnirt` CLI 的完整子命令与参数参考。任务导向的示例请见 [CLI 使用指南](../user_guide/serving/cli.md)。

## 调用方式

```bash
omnirt <subcommand> [options]
# 等价于
python -m omnirt <subcommand> [options]
```

可用子命令：

| 子命令 | 用途 |
|---|---|
| [`generate`](#generate) | 执行一次生成请求，导出产物 |
| [`validate`](#validate) | 校验一次请求（不碰硬件） |
| [`models`](#models) | 查询 registry |
| [`bench`](#bench) | 跑一次 benchmark 场景 |

全部子命令都接受相同的 **请求参数组**（`--task` / `--model` / `--backend` / `--prompt` / …）以及一个 **`--config` 选项**从 YAML / JSON 文件读取完整请求。

---

## `generate`

执行一次生成并把产物写到磁盘。

### 请求参数

| 参数 | 类型 | 说明 |
|---|---|---|
| `--task` | `text2image` / `image2image` / `text2video` / `image2video` / `audio2video` / `inpaint` / `edit` | 任务面（与 `--config` 之一必填） |
| `--model` | str | Registry id（例如 `sd15`、`flux2.dev`、`wan2.2-t2v-14b`） |
| `--backend` | `auto` / `cuda` / `ascend` / `cpu-stub` | 后端覆盖；缺省 `auto` 做自动选择 |
| `--prompt` | str | 文本 prompt |
| `--negative-prompt` | str | 负向 prompt |
| `--image` | path | 图像输入（`image2image` / `image2video` / `audio2video` / `inpaint` / `edit`） |
| `--mask` | path | `inpaint` 专用 mask |
| `--audio` | path | `audio2video` 专用音频 |
| `--num-frames` | int | 视频任务帧数 |
| `--fps` | int | 视频任务帧率 |
| `--frame-bucket` / `--motion-bucket-id` | int | SVD 运动幅度 |
| `--decode-chunk-size` | int | 视频解码分块 |
| `--noise-aug-strength` | float | SVD 输入噪声 |
| `--num-inference-steps` | int | 去噪步数 |
| `--guidance-scale` | float | CFG 强度 |
| `--preset` | `fast` / `balanced` / `quality` / `low-vram` | 整组参数预设 |
| `--scheduler` | str | 调度器覆盖（模型支持时） |
| `--seed` | int | 随机种子 |
| `--strength` | float | `image2image` / `edit` 的改写强度 |
| `--width` / `--height` | int | 输出尺寸 |
| `--dtype` | `fp16` / `bf16` / `fp32` | 计算精度 |
| `--num-images-per-prompt` | int | `text2image` 批量生成数量 |
| `--max-sequence-length` | int | Flux2 prompt token 上限 |
| `--output-dir` | path | 产物输出目录 |
| `--model-path` | path | 覆盖默认模型源 |
| `--repo-path` | path | script-backed 模型的外部仓库 checkout（如 FlashTalk） |

### 输入/配置文件

`--config <path>` 读取 YAML / JSON；文件字段与 `GenerateRequest` 一一对应：

```yaml
task: text2image
model: flux2.dev
backend: auto
inputs:
  prompt: "a cinematic sci-fi city at sunrise"
config:
  preset: balanced
  width: 1024
  height: 1024
```

命令行选项优先级**高于**文件字段。

### 输出格式

- 默认：人类可读 summary + 产物路径
- `--json`：把 `GenerateResult` 序列化为 JSON（含 `RunReport`），适合脚本消费

---

## `validate`

与 `generate` 共用请求参数组与 `--config` 选项，但**不执行推理**；只检查请求契约、registry、尺寸 / 参数约束。

用例：`omnirt validate --task text2image --model sd15 --prompt "…" --backend cpu-stub`

`--json` 输出一个 `ValidationResult` 对象，`ok=false` 时 `errors` 列出所有字段级问题。

详见 [请求校验](../user_guide/features/validation.md)。

---

## `models`

查询 registry。

| 用法 | 作用 |
|---|---|
| `omnirt models` | 列出全部注册模型 |
| `omnirt models <id>` | 查看某个模型的 `ModelCapabilities` |
| `omnirt models --task <task>` | 只列出支持指定任务的模型 |
| `omnirt models --format markdown` | 把清单输出成 Markdown 表（与自动生成的 [模型清单](../user_guide/models/supported_models.md) 同源） |

---

## `bench`

跑一次 benchmark 场景。支持 `--scenario` / `--repeat` 等参数；详见 [`src/omnirt/bench`](https://github.com/datascale-ai/omnirt/tree/main/src/omnirt/bench)。

---

## 环境变量

| 变量 | 作用 |
|---|---|
| `OMNIRT_LOG_LEVEL` | 日志级别（`DEBUG` / `INFO` / `WARNING` / `ERROR`），默认 `INFO` |
| `OMNIRT_DISABLE_COMPILE` | 置 `1` 跳过 `torch.compile` 与 `torch_npu.graph_mode` |
| `CUDA_VISIBLE_DEVICES` | CUDA 设备可见性 |
| `ASCEND_RT_VISIBLE_DEVICES` | Ascend 设备可见性（类比 CUDA） |
| `OMNIRT_API_KEY_FILE` | HTTP 服务的 API key 文件路径 |
| `OMNIRT_SDXL_MODEL_SOURCE` / `OMNIRT_SVD_MODEL_SOURCE` / … | Smoke 测试时的模型路径覆盖 |
| `HF_ENDPOINT` | HuggingFace 镜像（国内通常 `https://hf-mirror.com`） |

完整环境变量视角见 [国内部署](../user_guide/deployment/china_mirrors.md) 与 [Ascend 后端](../user_guide/deployment/ascend.md)。

## 相关

- [CLI 使用指南](../user_guide/serving/cli.md) — 任务导向的示例与 YAML 请求格式
- [Python API 参考](../api_reference/top_level.md) — 顶层 API 的自动参考
- [HTTP 服务](../user_guide/serving/http_server.md) — FastAPI 路由与启动参数
