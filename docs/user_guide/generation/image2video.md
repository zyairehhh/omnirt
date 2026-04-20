# 图像到视频（image2video）

以一张静态首帧为条件生成一个 MP4。典型场景：给产品图加镜头运动、给插画生成动态镜头。

## 最小示例

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

## 关键参数

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `image` | `str` | **必填** | 首帧路径 |
| `prompt` | `str?` | `None` | 运动方向文字提示（Wan2.2 i2v、LTX2 支持） |
| `num_frames` | `int?` | 模型默认 | SVD 通常 `14` / `25`；Wan2.2 `81` |
| `fps` | `int?` | 模型默认 | 输出帧率 |
| `motion_bucket_id` / `frame_bucket` | `int?` | 模型默认 | SVD 专属：运动幅度控制（越高越动） |
| `noise_aug_strength` | `float?` | 模型默认 | SVD 专属：输入噪声扰动 |
| `decode_chunk_size` | `int?` | 模型默认 | 解码分块大小；显存紧张时调小 |
| `preset` | `fast`/`balanced`/`quality`/`low-vram` | `balanced` | 见 [预设](../features/presets.md) |
| `seed` | `int?` | 随机 | 固定随机性 |

## 支持模型

- **Stable Video Diffusion 家族**：`svd`（14 帧）、`svd-xt`（25 帧）
- **Wan2.2 i2v**：`wan2.2-i2v-14b`
- **LTX-Video 2 i2v**：`ltx2-i2v`

完整清单：`omnirt models --task image2video`。

## 常见组合

- **SVD 短片**：`model=svd-xt, num_frames=25, fps=7, motion_bucket_id=127`
- **Wan2.2 长镜头**：`model=wan2.2-i2v-14b, num_frames=81, fps=16, prompt="camera panning left"`

## 错误与排查

!!! warning

    - **首帧尺寸未对齐** — SVD 要求 `1024×576` 或 `576×1024`；Wan2.2 i2v 会把输入 resize 到支持的桶宽
    - **运动幅度过大 / 过小** — 调 `motion_bucket_id`（SVD）或在 `prompt` 里描述运动（Wan2.2）
    - **解码 OOM** — 调小 `decode_chunk_size`（例如 SVD 上从默认值调到 `4`）
