# CLI Reference

Complete subcommand and parameter reference for the `omnirt` CLI. For task-oriented examples see [CLI Guide](../user_guide/serving/cli.md).

## Invocation

```bash
omnirt <subcommand> [options]
# Equivalent to
python -m omnirt <subcommand> [options]
```

Subcommands:

| Subcommand | Purpose |
|---|---|
| [`generate`](#generate) | Run a generation request and export artifacts |
| [`validate`](#validate) | Validate a request without touching hardware |
| [`models`](#models) | Query the registry |
| [`bench`](#bench) | Run a benchmark scenario |

All subcommands accept the same **request parameters** (`--task` / `--model` / `--backend` / `--prompt` / …) and a **`--config` option** to load a complete request from YAML / JSON.

---

## `generate`

Execute one generation and write the artifacts to disk.

### Request parameters

| Parameter | Type | Description |
|---|---|---|
| `--task` | `text2image` / `image2image` / `text2video` / `image2video` / `audio2video` / `inpaint` / `edit` | Task surface (one of `--task` or `--config` required) |
| `--model` | str | Registry id (e.g. `sd15`, `flux2.dev`, `wan2.2-t2v-14b`) |
| `--backend` | `auto` / `cuda` / `ascend` / `cpu-stub` | Override backend selection; default `auto` |
| `--prompt` | str | Text prompt |
| `--negative-prompt` | str | Negative prompt |
| `--image` | path | Image input (`image2image` / `image2video` / `audio2video` / `inpaint` / `edit`) |
| `--mask` | path | Inpaint mask |
| `--audio` | path | Audio input for `audio2video` |
| `--num-frames` | int | Frame count for video tasks |
| `--fps` | int | Frame rate for video tasks |
| `--frame-bucket` / `--motion-bucket-id` | int | SVD motion intensity |
| `--decode-chunk-size` | int | Video decode chunking |
| `--noise-aug-strength` | float | SVD input-noise augmentation |
| `--num-inference-steps` | int | Denoising steps |
| `--guidance-scale` | float | CFG strength |
| `--preset` | `fast` / `balanced` / `quality` / `low-vram` | Bundled parameter preset |
| `--scheduler` | str | Scheduler override (when the model supports alternates) |
| `--seed` | int | Random seed |
| `--strength` | float | Rewrite strength for `image2image` / `edit` |
| `--width` / `--height` | int | Output size |
| `--dtype` | `fp16` / `bf16` / `fp32` | Compute dtype |
| `--num-images-per-prompt` | int | Batch size for `text2image` |
| `--max-sequence-length` | int | Prompt-token cap for Flux2 |
| `--output-dir` | path | Artifact output directory |
| `--model-path` | path | Override the default model source |
| `--repo-path` | path | External repo checkout for script-backed models (e.g. FlashTalk) |

### Config file

`--config <path>` loads YAML / JSON whose fields map to `GenerateRequest`:

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

Command-line options **override** file fields.

### Output format

- Default: human-readable summary plus artifact paths
- `--json`: serialize `GenerateResult` (including `RunReport`) as JSON — ideal for scripts

---

## `validate`

Shares the request parameters and `--config` option with `generate`, but **does not run inference** — it only checks the request contract, registry, and dimensional / parameter constraints.

Example: `omnirt validate --task text2image --model sd15 --prompt "…" --backend cpu-stub`

`--json` emits a `ValidationResult`; when `ok=false`, `errors` lists every field-level issue.

See [Validation](../user_guide/features/validation.md).

---

## `models`

Query the registry.

| Usage | Purpose |
|---|---|
| `omnirt models` | List every registered model |
| `omnirt models <id>` | Dump `ModelCapabilities` for one model |
| `omnirt models --task <task>` | Filter by supported task |
| `omnirt models --format markdown` | Emit the list as Markdown (same source as the auto-generated [Supported Models](../user_guide/models/supported_models.md)) |

---

## `bench`

Run a benchmark scenario. Supports `--scenario` / `--repeat` and related flags; see [`src/omnirt/bench`](https://github.com/datascale-ai/omnirt/tree/main/src/omnirt/bench).

---

## Environment variables

| Variable | Purpose |
|---|---|
| `OMNIRT_LOG_LEVEL` | Log level (`DEBUG` / `INFO` / `WARNING` / `ERROR`), default `INFO` |
| `OMNIRT_DISABLE_COMPILE` | Set to `1` to skip `torch.compile` and `torch_npu.graph_mode` |
| `CUDA_VISIBLE_DEVICES` | CUDA device visibility |
| `ASCEND_RT_VISIBLE_DEVICES` | Ascend device visibility (analog of CUDA) |
| `OMNIRT_API_KEY_FILE` | API-key file for the HTTP server |
| `OMNIRT_SDXL_MODEL_SOURCE` / `OMNIRT_SVD_MODEL_SOURCE` / … | Model-path overrides for smoke tests |
| `HF_ENDPOINT` | HuggingFace mirror (typically `https://hf-mirror.com` in China) |

More environment context in [Domestic Deployment](../user_guide/deployment/china_mirrors.md) and [Ascend Backend](../user_guide/deployment/ascend.md).

## Related

- [CLI Guide](../user_guide/serving/cli.md) — task-oriented examples and YAML request format
- [Python API Reference](../api_reference/top_level.md) — auto-generated top-level API
- [HTTP Server](../user_guide/serving/http_server.md) — FastAPI routes and startup options
