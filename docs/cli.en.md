# CLI

OmniRT exposes three top-level CLI commands:

- `omnirt generate`
- `omnirt validate`
- `omnirt models`

Together they cover execution, preflight validation, and model discovery.

The currently documented public task surfaces are `text2image`, `image2image`, `text2video`, `image2video`, and `audio2video`. For `image2image`, start with `sdxl-base-1.0`, `sdxl-refiner-1.0`, `sd15`, or `sd21`.

## Request shape

The CLI mirrors the same envelope used by `GenerateRequest`:

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

Rule of thumb:

- `inputs` contains semantic generation inputs such as `prompt`, `negative_prompt`, `image`, `mask`, `audio`, `num_frames`, and `fps`
- `config` contains execution settings such as `preset`, `scheduler`, `num_inference_steps`, `guidance_scale`, `dtype`, `seed`, `height`, `width`, and `output_dir`

About weight sources:

- `model_path` can be either a local Diffusers directory or a Hugging Face repo id
- LoRA adapter paths can point to a local `.safetensors` file or an explicit Hugging Face ref like `hf://owner/repo/path/to/file.safetensors`

## `omnirt generate`

Run a request from flags:

```bash
omnirt generate \
  --task text2image \
  --model sd15 \
  --prompt "a lighthouse in fog" \
  --backend cuda \
  --preset fast
```

Run a request from YAML or JSON:

```bash
omnirt generate --config request.yaml --json
```

Validate and resolve defaults without execution:

```bash
omnirt generate \
  --task text2video \
  --model wan2.2-t2v-14b \
  --prompt "a glass whale gliding over a moonlit harbor" \
  --preset fast \
  --dry-run
```

## `omnirt validate`

Use validation when you want schema feedback, model-task compatibility checks, and resolved defaults before an expensive run:

```bash
omnirt validate \
  --task text2image \
  --model qwen-image \
  --prompt "a poster with a bold Chinese headline" \
  --backend cpu-stub
```

Useful validation behavior includes:

- unknown model suggestions
- task and model compatibility checks
- rejected unsupported inputs or config keys
- resolved backend reporting

## `omnirt models`

List the registry:

```bash
omnirt models
```

Inspect one model in detail:

```bash
omnirt models flux2.dev
```

The model detail output summarizes task support, backend defaults, required inputs, supported config keys, presets, and an example command.

## Task examples

Text to image:

```bash
omnirt generate \
  --task text2image \
  --model flux2.dev \
  --prompt "a cinematic sci-fi city at sunrise" \
  --preset balanced
```

Image to video:

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

Image to image:

```bash
omnirt generate \
  --task image2image \
  --model sdxl-base-1.0 \
  --image input.png \
  --prompt "cinematic concept art" \
  --backend cuda \
  --strength 0.8
```

Text to video:

```bash
omnirt generate \
  --task text2video \
  --model cogvideox-2b \
  --prompt "a wooden toy ship gliding over a plush blue carpet" \
  --backend cuda \
  --num-frames 81 \
  --fps 16
```

Audio to video:

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

## Presets

Current named presets:

- `fast`
- `balanced`
- `quality`
- `low-vram`

Use presets to start from sensible defaults before overriding individual config values.
