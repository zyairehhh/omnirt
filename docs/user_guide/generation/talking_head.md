# 数字人（audio2video / talking head）

给一张人脸 portrait + 一段音频，生成口型与头部动作对齐的 MP4。OmniRT 通过 `soulx-flashtalk-14b` 支持这一任务面。

## 最小示例

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

## 关键参数

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `image` | `str` | **必填** | 人脸 portrait 路径 |
| `audio` | `str` | **必填** | 音频路径（推荐 `.wav`；支持 ffmpeg 能解的格式） |
| `prompt` | `str?` | `None` | 可选提示（表情 / 情绪） |
| `preset` | `fast`/`balanced`/`quality`/`low-vram` | `balanced` | 见 [预设](../features/presets.md) |
| `fps` | `int?` | 模型默认 | 输出帧率 |
| `repo_path` | `str?` | auto | 外部仓库 checkout 路径（SoulX-FlashTalk 为 script-backed 模型，首次运行会自动克隆） |

## 支持模型

| 模型 | 输入 | 输出 | 显存 |
|---|---|---|---|
| `soulx-flashtalk-14b` | portrait + audio | MP4 | ≥ 20 GB |

!!! info "SoulX-FlashTalk 是 script-backed 模型"
    首次运行会克隆外部仓库到 `~/.cache/omnirt/repos/`。内网/离线环境请参考 [国内部署](../deployment/china_mirrors.md) 的 "script-backed 模型镜像" 小节。

## 错误与排查

!!! warning

    - **音频采样率不匹配** — FlashTalk 要求 16 kHz 单声道；非此格式会自动 resample，但过长音频会放大误差
    - **portrait 对齐差** — 正面、上半身、眼睛可见效果最好
    - **外部仓库克隆失败** — 在国内网络下走 `GHPROXY` 或离线提供 `repo_path`
    - **Ascend 上速度显著低于 CUDA** — FlashTalk 的部分自定义算子尚未在 Ascend 上优化，会触发 eager 回退（见 `RunReport.backend_timeline`）
