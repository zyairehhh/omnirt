# OmniRT

[English](./README.md)

面向 CUDA 与 Ascend 的图像、视频生成运行时。

## 项目简介

OmniRT 是一个基于 Diffusers 的统一运行时层，为不同模型家族提供一致的 CLI、Python API、校验流程、产物导出协议和后端抽象。

当前公开任务面：

- `text2image`
- `text2video`
- `image2video`

当前公开接口能力：

- `omnirt generate`、`omnirt validate`、`omnirt models`
- Python 侧 typed request helpers 与 `pipeline(...)` 便捷调用
- 统一的 `GenerateRequest` / `GenerateResult` / `RunReport`
- PNG / MP4 产物导出
- 模型注册元数据、preset、请求校验和后端选择

## 当前已支持模型

目前已经接入的代表性模型家族：

- Stable Diffusion：`sd15`、`sd21`、`sdxl-base-1.0`、`sdxl-turbo`、`sd3-medium`、`sd3.5-large`、`sd3.5-large-turbo`
- Flux：`flux-dev`、`flux-schnell`、`flux2.dev`、`flux2-dev`
- 通用图像：`glm-image`、`hunyuan-image-2.1`、`omnigen`、`qwen-image`、`sana-1.6b`、`ovis-image`、`hidream-i1`
- 视频：`svd`、`svd-xt`、`cogvideox-2b`、`cogvideox-5b`、`kandinsky5-t2v`、`kandinsky5-i2v`、`wan2.1-*`、`wan2.2-*`、`hunyuan-video`、`hunyuan-video-1.5-*`、`helios-*`、`sana-video`、`ltx-video`、`ltx2-i2v`

你也可以直接通过 CLI 查看实时 registry：

```bash
omnirt models
omnirt models flux2.dev
```

## 快速开始

```bash
python3 -m pip install -e .[dev]
python3 -m omnirt --help
pytest
```

如果要真正运行模型，还需要安装 runtime 依赖：

```bash
python3 -m pip install -e '.[runtime,dev]'
```

真正的 CUDA / Ascend 端到端生成仍然依赖对应硬件、运行时库和模型权重。

## CLI 用法

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

执行：

```bash
omnirt generate --config request.yaml --json
```

直接传参：

```bash
omnirt generate \
  --task text2image \
  --model sd15 \
  --prompt "a lighthouse in fog" \
  --backend cuda \
  --preset fast
```

先校验、不执行：

```bash
omnirt validate \
  --task text2image \
  --model qwen-image \
  --prompt "一张带有中文标题的电影海报" \
  --backend cpu-stub

omnirt generate \
  --task text2video \
  --model wan2.2-t2v-14b \
  --prompt "a glass whale gliding over a moonlit harbor" \
  --preset fast \
  --dry-run
```

视频示例：

```bash
omnirt generate \
  --task image2video \
  --model svd-xt \
  --image input.png \
  --backend cuda \
  --num-frames 25 \
  --fps 7 \
  --frame-bucket 127 \
  --decode-chunk-size 8

omnirt generate \
  --task text2video \
  --model cogvideox-2b \
  --prompt "a wooden toy ship gliding over a plush blue carpet" \
  --backend cuda \
  --num-frames 81 \
  --fps 16
```

当前 preset：

- `fast`
- `balanced`
- `quality`
- `low-vram`

## Python API

typed request helpers：

```python
from omnirt import requests, validate, generate

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

## 当前边界

已经公开并适合直接集成的能力：

- 统一的图像 / 视频生成请求
- 模型发现与请求校验
- 基于 registry 的模型元数据
- CLI 与 Python 双入口

还没有正式公开成一等 API 的任务面：

- `image2image`
- `inpaint`
- `edit`
- `video2video`

部分底层模型家族已经为后续扩展铺好了 scaffold，但这些任务面目前还不是完整公开接口。

## 测试与验证

- `pytest tests/unit tests/parity` 覆盖本地契约层与指标层
- `pytest tests/integration/test_error_paths.py` 覆盖低显存与坏权重错误路径
- CUDA / Ascend 集成测试会在缺少硬件、运行时包或本地模型目录时自动跳过

整体实现目标和剩余硬件验证事项见 [PLAN.md](./PLAN.md)。

## 文档

- 模型接入说明：[docs/model-onboarding.md](./docs/model-onboarding.md)
- 模型支持路线图：[docs/model-support-roadmap.md](./docs/model-support-roadmap.md)
- 中国区部署说明：[docs/china-deployment.md](./docs/china-deployment.md)
- 架构说明：[docs/architecture.md](./docs/architecture.md)
- 服务协议草案：[docs/service-schema.md](./docs/service-schema.md)
- 接口改进提案：[docs/interface-improvement-proposal.md](./docs/interface-improvement-proposal.md)

## 工具脚本

- 准备离线模型快照：[scripts/prepare_model_snapshot.py](/Users/<user>/Desktop/code/opensource/omnirt/scripts/prepare_model_snapshot.py)
- 拉取 Modelers 仓库快照：[scripts/prepare_modelers_snapshot.py](/Users/<user>/Desktop/code/opensource/omnirt/scripts/prepare_modelers_snapshot.py)
- 检查本地模型目录布局：[scripts/check_model_layout.py](/Users/<user>/Desktop/code/opensource/omnirt/scripts/check_model_layout.py)
- 同步模型目录到服务器：[scripts/sync_model_dir.sh](/Users/<user>/Desktop/code/opensource/omnirt/scripts/sync_model_dir.sh)
