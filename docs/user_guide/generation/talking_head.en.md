# Talking Head (audio2video)

Given a face portrait plus an audio clip, produce an MP4 where lips and head motion are aligned to the speech. OmniRT supports this task via `soulx-flashtalk-14b`.

## Minimal example

=== "Python"

    ```python
    from omnirt import generate
    from omnirt.requests import audio2video

    result = generate(audio2video(
        model="soulx-flashtalk-14b",
        image="inputs/portrait.png",
        audio="inputs/speech.wav",
        preset="balanced",
    ))
    ```

=== "CLI"

    ```bash
    omnirt generate \
      --task audio2video \
      --model soulx-flashtalk-14b \
      --image inputs/portrait.png \
      --audio inputs/speech.wav \
      --preset balanced
    ```

=== "HTTP"

    ```bash
    curl -sS http://localhost:8000/v1/generate \
      -H 'Content-Type: application/json' \
      -d '{
        "task": "audio2video",
        "model": "soulx-flashtalk-14b",
        "inputs": {
          "image": "inputs/portrait.png",
          "audio": "inputs/speech.wav"
        },
        "config": {"preset": "balanced"}
      }'
    ```

## Key parameters

| Parameter | Type | Default | Notes |
|---|---|---|---|
| `image` | `str` | **required** | path to the face portrait |
| `audio` | `str` | **required** | audio path (prefer `.wav`; any ffmpeg-decodable format works) |
| `prompt` | `str?` | `None` | optional hint (expression / emotion) |
| `preset` | `fast`/`balanced`/`quality`/`low-vram` | `balanced` | see [Presets](../features/presets.md) |
| `fps` | `int?` | model default | output frame rate |
| `repo_path` | `str?` | auto | external repo checkout path (SoulX-FlashTalk is script-backed; first run auto-clones) |

## Supported models

| Model | Inputs | Output | VRAM |
|---|---|---|---|
| `soulx-flashtalk-14b` | portrait + audio | MP4 | ≥ 20 GB |

!!! info "SoulX-FlashTalk is a script-backed model"
    The first run clones an external repository into `~/.cache/omnirt/repos/`. For restricted networks see the "script-backed model mirrors" section of [Domestic Deployment](../deployment/china_mirrors.md).

## Troubleshooting

!!! warning

    - **Audio sample-rate mismatch** — FlashTalk expects 16 kHz mono. Other formats are resampled automatically, but long audio amplifies error.
    - **Poor portrait alignment** — frontal, upper-body, eyes-visible portraits give the best output.
    - **External repo clone fails** — behind the GFW, route through `GHPROXY` or supply an offline `repo_path`.
    - **Ascend much slower than CUDA** — some FlashTalk custom ops aren't optimized on Ascend and fall back to eager (tracked in `RunReport.backend_timeline`).
