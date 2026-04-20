# 快速开始

这份指南覆盖从全新 checkout 到一次可校验的 OmniRT 请求，以及本地文档站预览的最短路径。

## 安装

通用开发环境：

```bash
python -m pip install -e '.[dev]'
```

如果要真正执行模型推理，再安装 runtime 依赖：

```bash
python -m pip install -e '.[runtime,dev]'
```

如果要维护文档站：

```bash
python -m pip install -e '.[docs]'
```

如果你希望把代码、runtime、文档工具放在同一个环境里：

```bash
python -m pip install -e '.[runtime,dev,docs]'
```

## 查看 CLI

```bash
python -m omnirt --help
omnirt models
omnirt models flux2.dev
```

`omnirt models` 是查看实时模型 registry 的最快方式，不需要先读源码。

## 先校验第一条请求

在真正使用 GPU / NPU 之前，建议先做一次纯校验：

```bash
omnirt validate \
  --task text2image \
  --model qwen-image \
  --prompt "a poster with a bold title" \
  --backend cpu-stub
```

也可以直接校验配置文件：

```bash
omnirt validate --config request.yaml --json
```

## 跑通第一条生成请求

```bash
omnirt generate \
  --task text2image \
  --model sd15 \
  --prompt "a lighthouse in fog" \
  --backend cuda \
  --preset fast
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

## 运行测试

本地快速覆盖：

```bash
pytest tests/unit tests/parity
```

错误路径集成测试：

```bash
pytest tests/integration/test_error_paths.py
```

依赖真实硬件的 CUDA / Ascend smoke tests 已接入 CI；本地若缺少对应 runtime 包或模型目录，会自动跳过。

## 预览文档站

```bash
mkdocs serve
```

使用严格模式构建静态站点：

```bash
mkdocs build --strict
```

GitHub Pages 发布说明见 [文档发布](publishing-docs.md)。
