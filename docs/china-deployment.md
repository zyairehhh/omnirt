# China Deployment Guide

This guide documents the recommended OmniRT workflow for environments where public model downloads may be slow, blocked, or unreliable.

The key principle is simple:

`OmniRT production and smoke tests should use local model directories, not runtime downloads.`

## Why this matters

In many mainland China environments:

- `huggingface.co` may resolve in DNS but still time out on HTTPS
- large model downloads are unstable or extremely slow
- GPU and NPU servers often have weaker outbound internet access than development laptops
- temporary runtime downloads make CI and hardware smoke tests flaky

Because of that, `model_path` should be treated as the primary deployment mode.

## Recommended deployment pattern

1. Download model snapshots on a machine with stable external access.
2. Store those snapshots in a shared internal location.
3. Sync them to GPU or Ascend servers ahead of time.
4. Point OmniRT to local directories through `config.model_path` or environment variables.

Recommended directory layout:

```text
/data/models/omnirt/
  sdxl-base-1.0/
  svd/
  svd-xt/
  flux2-dev/
  wan2.2-t2v-14b/
  wan2.2-i2v-14b/
  qwen-image/
```

## Integration test policy

OmniRT integration smoke tests intentionally require local model directories.

- `OMNIRT_SDXL_MODEL_SOURCE` must point to an existing local directory
- `OMNIRT_SVD_MODEL_SOURCE` must point to the base SVD local directory
- `OMNIRT_SVD_XT_MODEL_SOURCE` must point to the SVD-XT local directory
- if the directory is missing, the test skips
- the tests do not treat a Hugging Face repo id as a valid smoke-test source

This keeps hardware validation deterministic in restricted networks.

## Preparing local snapshots

On a machine with outbound access, use:

```bash
python scripts/prepare_model_snapshot.py \
  --repo-id stabilityai/stable-diffusion-xl-base-1.0 \
  --output-dir /data/models/omnirt/sdxl-base-1.0
```

Example for SVD:

```bash
python scripts/prepare_model_snapshot.py \
  --repo-id stabilityai/stable-video-diffusion-img2vid \
  --output-dir /data/models/omnirt/svd
```

Example for SVD-XT:

```bash
python scripts/prepare_model_snapshot.py \
  --repo-id stabilityai/stable-video-diffusion-img2vid-xt \
  --output-dir /data/models/omnirt/svd-xt
```

If a mirror is available, set the endpoint before running:

```bash
export HF_ENDPOINT=https://<your-mirror-host>
```

If you prefer downloading directly from Modelers, clone the repository into a local directory:

```bash
python scripts/prepare_modelers_snapshot.py \
  --repo-id MindSpore-Lab/SDXL_Base1_0 \
  --output-dir /data/models/omnirt/SDXL_Base1_0
```

This follows Modelers' documented Git download flow:

`git clone https://modelers.cn/<username>/<model_name>.git`

After downloading, validate the layout before syncing:

```bash
python scripts/check_model_layout.py \
  --task sdxl \
  --model-dir /data/models/omnirt/sdxl-base-1.0
```

For SVD:

```bash
python scripts/check_model_layout.py \
  --task svd \
  --model-dir /data/models/omnirt/svd-xt
```

This is especially useful for non-Hugging-Face sources such as mirrored archives or Modelers downloads.

## Syncing to target servers

Use the provided sync helper:

```bash
bash scripts/sync_model_dir.sh \
  /data/models/omnirt/sdxl-base-1.0 \
  user@<cuda-host>:/data/models/omnirt/sdxl-base-1.0
```

You can do the same for Ascend:

```bash
bash scripts/sync_model_dir.sh \
  /data/models/omnirt/svd-xt \
  user@<ascend-host>:/home/<user>/models/omnirt/svd-xt
```

## Running OmniRT with local models

CLI example:

```bash
omnirt generate \
  --task text2image \
  --model sdxl-base-1.0 \
  --prompt "a cinematic sci-fi city at sunrise" \
  --backend cuda \
  --model-path /data/models/omnirt/sdxl-base-1.0
```

Environment-variable example for smoke tests:

```bash
export OMNIRT_SDXL_MODEL_SOURCE=/data/models/omnirt/sdxl-base-1.0
export OMNIRT_SVD_MODEL_SOURCE=/data/models/omnirt/svd
export OMNIRT_SVD_XT_MODEL_SOURCE=/data/models/omnirt/svd-xt
pytest tests/integration/test_sdxl_cuda.py tests/integration/test_svd_base_cuda.py tests/integration/test_svd_cuda.py
```

## Python dependency strategy

For restricted networks, prefer:

- internal PyPI mirrors
- prebuilt wheel caches
- environment reuse on target servers

Typical example:

```bash
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -e '.[runtime,dev]'
```

For Ascend servers, avoid large ad-hoc environment rebuilds when a working `torch_npu` environment already exists.

## Server-specific guidance

### CUDA hosts

- preinstall `torch`, `diffusers`, `transformers`, `safetensors`, `accelerate`
- validate with local model directories only
- avoid relying on runtime Hugging Face downloads

### Ascend hosts

- source `set_env.sh` before running OmniRT
- reuse the known-good `torch_npu` virtualenv when possible
- install only the minimal missing packages if the environment is otherwise healthy

## What to avoid

- using Hugging Face repo ids directly in hardware smoke tests
- assuming public internet access from GPU or NPU servers
- downloading model weights during CI
- rebuilding full Python environments on every verification run

## Suggested operational workflow

1. Maintain a connected machine for model snapshot preparation.
2. Mirror or snapshot approved model versions into a local model store.
3. Sync model directories into each hardware environment.
4. Run OmniRT smoke tests against those local directories.
5. Only update model snapshots intentionally, not implicitly during runtime.
