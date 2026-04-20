# CLI 文档

OmniRT 当前公开三个顶层 CLI 命令：

- `omnirt generate`
- `omnirt validate`
- `omnirt models`

它们分别覆盖执行、预检校验和模型发现三类能力。

当前已经正式公开的任务面包括 `text2image`、`image2image`、`text2video`、`image2video` 和 `audio2video`。其中 `image2image` 当前推荐优先使用 `sdxl-base-1.0`、`sdxl-refiner-1.0`、`sd15`、`sd21`。

## 请求结构

CLI 与 `GenerateRequest` 使用同一套请求包结构：

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

一个简单的判断规则：

- `inputs` 放语义输入，例如 `prompt`、`negative_prompt`、`image`、`mask`、`audio`、`num_frames`、`fps`
- `config` 放执行配置，例如 `preset`、`scheduler`、`num_inference_steps`、`guidance_scale`、`dtype`、`seed`、`height`、`width`、`output_dir`

关于权重来源：

- `model_path` 可以是本地 Diffusers 模型目录，也可以直接是 Hugging Face repo id
- LoRA adapter 路径可以是本地 `.safetensors` 文件，也可以写成 `hf://owner/repo/path/to/file.safetensors`

## `omnirt generate`

直接用命令行参数执行：

```bash
omnirt generate \
  --task text2image \
  --model sd15 \
  --prompt "a lighthouse in fog" \
  --backend cuda \
  --preset fast
```

从 YAML 或 JSON 请求文件执行：

```bash
omnirt generate --config request.yaml --json
```

只校验并解析默认值，不真正执行：

```bash
omnirt generate \
  --task text2video \
  --model wan2.2-t2v-14b \
  --prompt "a glass whale gliding over a moonlit harbor" \
  --preset fast \
  --dry-run
```

## `omnirt validate`

如果你想在昂贵执行之前先拿到 schema 反馈、模型任务兼容性检查和默认值解析结果，可以使用校验命令：

```bash
omnirt validate \
  --task text2image \
  --model qwen-image \
  --prompt "a poster with a bold Chinese headline" \
  --backend cpu-stub
```

校验阶段会带来这些直接收益：

- 对未知模型给出相近建议
- 检查任务与模型是否匹配
- 拒绝不支持的输入或配置字段
- 报告最终解析出的后端

## `omnirt models`

列出当前 registry：

```bash
omnirt models
```

查看某个模型的详细信息：

```bash
omnirt models flux2.dev
```

模型详情会汇总任务支持情况、默认后端、必填输入、支持的配置项、预设和示例命令。

## 任务示例

文生图：

```bash
omnirt generate \
  --task text2image \
  --model flux2.dev \
  --prompt "a cinematic sci-fi city at sunrise" \
  --preset balanced
```

图生视频：

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
```

图生图：

```bash
omnirt generate \
  --task image2image \
  --model sdxl-base-1.0 \
  --image input.png \
  --prompt "cinematic concept art" \
  --backend cuda \
  --strength 0.8
```

文生视频：

```bash
omnirt generate \
  --task text2video \
  --model cogvideox-2b \
  --prompt "a wooden toy ship gliding over a plush blue carpet" \
  --backend cuda \
  --num-frames 81 \
  --fps 16
```

音频驱动视频：

```bash
omnirt generate \
  --task audio2video \
  --model soulx-flashtalk-14b \
  --image speaker.png \
  --audio voice.wav \
  --prompt "A person is talking." \
  --backend ascend \
  --repo-path /path/to/SoulX-FlashTalk
```

## Preset

当前公开的预设：

- `fast`
- `balanced`
- `quality`
- `low-vram`

建议先从 preset 出发，再按需覆写单个配置项。
