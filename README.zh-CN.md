# OmniRT

[English](./README.md) | [文档站点](https://datascale-ai.github.io/omnirt/) | [英文文档](https://datascale-ai.github.io/omnirt/en/)

面向 CUDA 与 Ascend 后端的开源图像、视频与音频驱动数字人生成运行时。

OmniRT 为不同模型家族提供统一的 CLI、Python API、请求校验流程、产物导出协议和后端抽象，让你在切换模型时不需要重学整套运行时接口。

## 亮点

- 以 `GenerateRequest`、`GenerateResult`、`RunReport` 为核心的统一请求 / 返回契约
- 面向用户的 CLI：`omnirt generate`、`omnirt validate`、`omnirt models`
- Python 侧同时提供 `requests.*` helper 和 `pipeline(...)` 便捷封装
- 支持 `cuda`、`ascend` 和 `cpu-stub` 三类后端模式
- 标准 PNG / MP4 产物导出
- 模型来源既支持本地目录，也支持 Hugging Face repo id
- LoRA safetensors 既支持本地文件，也支持 Hugging Face 单文件引用
- 适合本地模型目录、离线快照和受限网络环境的部署方式

## 当前公开任务面

| 任务面 | 说明 | 典型输出 |
|---|---|---|
| `text2image` | 文本驱动图像生成 | PNG |
| `image2image` | 图像引导图像生成 | PNG |
| `text2video` | 文本驱动视频生成 | MP4 |
| `image2video` | 首帧引导视频生成 | MP4 |
| `audio2video` | 音频驱动说话头像生成 | MP4 |

通过 CLI 可以最快查看当前实时 registry：

```bash
omnirt models
omnirt models flux2.dev
```

## 已支持模型家族

当前已经接入 registry 的代表性模型家族：

- Stable Diffusion：`sd15`、`sd21`、`sdxl-base-1.0`、`sdxl-refiner-1.0`、`sdxl-turbo`、`sd3-medium`、`sd3.5-large`、`sd3.5-large-turbo`
- Flux：`flux-dev`、`flux-schnell`、`flux-fill`、`flux-kontext`、`flux2.dev`、`flux2-dev`
- 通用图像：`glm-image`、`hunyuan-image-2.1`、`omnigen`、`qwen-image`、`qwen-image-edit`、`qwen-image-edit-plus`、`sana-1.6b`、`ovis-image`、`hidream-i1`
- 视频：`svd`、`svd-xt`、`cogvideox-2b`、`cogvideox-5b`、`kandinsky5-t2v`、`kandinsky5-i2v`、`wan2.1-*`、`wan2.2-*`、`hunyuan-video`、`hunyuan-video-1.5-*`、`helios-*`、`sana-video`、`ltx-video`、`ltx2-i2v`
- 数字人 / 说话头像：`soulx-flashtalk-14b`，当前在 Ascend 上运行

当前公开接口已经足以支撑生成、校验、模型发现和产物导出。`image2image` 已作为正式公开任务面提供，当前推荐优先使用 `sdxl-base-1.0`、`sdxl-refiner-1.0`、`sd15` 和 `sd21`。`inpaint`、`edit`、`video2video` 仍在继续演进。

## 快速开始

安装本地开发依赖：

```bash
python -m pip install -e '.[dev]'
python -m omnirt --help
pytest
```

如果要真正执行模型推理，再安装 runtime 依赖：

```bash
python -m pip install -e '.[runtime,dev]'
```

如果要预览或维护文档站，再安装 docs 依赖：

```bash
python -m pip install -e '.[docs]'
```

## 第一条请求

建议先校验再执行：

```bash
omnirt validate \
  --task text2image \
  --model qwen-image \
  --prompt "一张带有醒目标题的海报" \
  --backend cpu-stub
```

执行一条最小生成请求：

```bash
omnirt generate \
  --task text2image \
  --model sd15 \
  --prompt "a lighthouse in fog" \
  --backend cuda \
  --preset fast
```

执行一条最小 `image2image` 请求：

```bash
omnirt generate \
  --task image2image \
  --model sdxl-base-1.0 \
  --image input.png \
  --prompt "cinematic concept art" \
  --backend cuda
```

YAML 请求示例：

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

执行方式：

```bash
omnirt generate --config request.yaml --json
```

`model_path` 现在可以同时指向：

- 本地 Diffusers 模型目录
- Hugging Face repo id，例如 `stabilityai/stable-diffusion-xl-base-1.0`

对于单文件 LoRA 权重，可以使用本地 `.safetensors` 文件，或显式的 Hugging Face 引用：

```text
hf://owner/repo/path/to/adapter.safetensors
hf://owner/repo/path/to/adapter.safetensors?revision=main
```

## Python API

```python
from omnirt import generate, requests, validate

req = requests.text2image(
    model="flux2.dev",
    prompt="a cinematic sci-fi city at sunrise",
    width=1024,
    height=1024,
    preset="balanced",
)

validation = validate(req, backend="cpu-stub")
result = generate(req, backend="cuda")
```

pipeline 风格便捷层：

```python
import omnirt

pipe = omnirt.pipeline("sd15", backend="cpu-stub")
validation = pipe.validate(prompt="a lighthouse in fog", preset="fast")
```

`image2image` 也使用同一套公开入口：

```python
img2img = requests.image2image(
    model="sdxl-base-1.0",
    image="input.png",
    prompt="cinematic concept art",
    strength=0.8,
)
```

## 测试与验证

- `pytest tests/unit tests/parity` 覆盖本地契约层与指标层
- `pytest tests/integration/test_error_paths.py` 覆盖低显存与坏权重错误路径
- CUDA / Ascend smoke tests 会在缺少硬件、运行时包或本地模型目录时自动跳过

真正的端到端生成仍然依赖目标硬件、运行时库和模型权重。

## 当前状态

- `sdxl-base-1.0` 和 `svd-xt` 已经完成 CUDA / Ascend 双后端真机 smoke
- `image2image` 已公开支持；其中 `sdxl-refiner-1.0` 的 CUDA / Ascend smoke 用例已经具备，但仍待补齐已验证的本地模型目录与真机验证结果
- `flux-fill`、`flux-kontext`、`qwen-image-edit*` 等编辑模型已经具备接入和 smoke 用例入口，但仍待补齐已验证的本地模型目录与真机验证结果
- 更完整的路线图见 [docs/model-support-roadmap.md](./docs/model-support-roadmap.md)，当前接入快照见 [docs/support-status.md](./docs/support-status.md)

## 文档

- 文档站点：<https://datascale-ai.github.io/omnirt/>
- 英文文档：<https://datascale-ai.github.io/omnirt/en/>
- 模型接入说明：[docs/model-onboarding.md](./docs/model-onboarding.md)
- 当前支持状态：[docs/support-status.md](./docs/support-status.md)
- 模型支持路线图：[docs/model-support-roadmap.md](./docs/model-support-roadmap.md)
- 中国区部署说明：[docs/china-deployment.md](./docs/china-deployment.md)
- 架构说明：[docs/architecture.md](./docs/architecture.md)
- 服务协议草案：[docs/service-schema.md](./docs/service-schema.md)
- 接口改进提案：[docs/interface-improvement-proposal.md](./docs/interface-improvement-proposal.md)

## 工具脚本

- 准备离线模型快照：[scripts/prepare_model_snapshot.py](./scripts/prepare_model_snapshot.py)
- 拉取 Modelers 仓库快照：[scripts/prepare_modelers_snapshot.py](./scripts/prepare_modelers_snapshot.py)
- 准备 ModelScope 仓库及指定大文件快照：[scripts/prepare_modelscope_snapshot.py](./scripts/prepare_modelscope_snapshot.py)
- 检查本地模型目录布局：[scripts/check_model_layout.py](./scripts/check_model_layout.py)
- 同步模型目录到服务器：[scripts/sync_model_dir.sh](./scripts/sync_model_dir.sh)
