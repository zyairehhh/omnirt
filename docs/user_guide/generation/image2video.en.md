# Image to Video

Generate an MP4 conditioned on a single first frame. Typical use cases: add camera motion to product shots, animate illustrations.

## Minimal example

=== "Python"

    ```python
    from omnirt import generate
    from omnirt.requests import image2video

    result = generate(image2video(
        model="svd-xt",
        image="inputs/first_frame.png",
        num_frames=25,
        fps=7,
        preset="balanced",
    ))
    ```

=== "CLI"

    ```bash
    omnirt generate \
      --task image2video \
      --model svd-xt \
      --image inputs/first_frame.png \
      --num-frames 25 --fps 7 --preset balanced
    ```

=== "HTTP"

    ```bash
    curl -sS http://localhost:8000/v1/generate \
      -H 'Content-Type: application/json' \
      -d '{
        "task": "image2video",
        "model": "svd-xt",
        "inputs": {"image": "inputs/first_frame.png"},
        "config": {"num_frames": 25, "fps": 7, "preset": "balanced"}
      }'
    ```

## Key parameters

| Parameter | Type | Default | Notes |
|---|---|---|---|
| `image` | `str` | **required** | first-frame path |
| `prompt` | `str?` | `None` | motion hint (supported by Wan2.2 i2v and LTX2) |
| `num_frames` | `int?` | model default | SVD typically `14` or `25`; Wan2.2 `81` |
| `fps` | `int?` | model default | output frame rate |
| `motion_bucket_id` / `frame_bucket` | `int?` | model default | SVD-only: motion intensity |
| `noise_aug_strength` | `float?` | model default | SVD-only: input-noise perturbation |
| `decode_chunk_size` | `int?` | model default | lower when VRAM is tight |
| `preset` | `fast`/`balanced`/`quality`/`low-vram` | `balanced` | see [Presets](../features/presets.md) |
| `seed` | `int?` | random | reproducibility |

## Supported models

- **Stable Video Diffusion family**: `svd` (14 frames), `svd-xt` (25 frames)
- **Wan2.2 i2v**: `wan2.2-i2v-14b`
- **LTX-Video 2 i2v**: `ltx2-i2v`

Full list: `omnirt models --task image2video`.

## Common recipes

- **SVD short clip**: `model=svd-xt, num_frames=25, fps=7, motion_bucket_id=127`
- **Wan2.2 long shot**: `model=wan2.2-i2v-14b, num_frames=81, fps=16, prompt="camera panning left"`

## Troubleshooting

!!! warning

    - **First frame not aligned** — SVD requires `1024×576` or `576×1024`; Wan2.2 i2v resizes to supported buckets automatically
    - **Too much / too little motion** — tune `motion_bucket_id` (SVD) or describe motion in the `prompt` (Wan2.2)
    - **Decode OOM** — lower `decode_chunk_size` (e.g. SVD default → `4`)
