# OmniRT

OmniRT 为图像、视频和音频驱动数字人模型提供统一的 CLI、Python API、请求校验、产物导出协议与后端抽象，让不同模型家族在同一套运行时契约里工作。

- 统一覆盖 `GenerateRequest`、`GenerateResult`、`RunReport` 三个核心对象
- 同一套运行时接口同时面向 `cuda`、`ascend` 与 `cpu-stub`
- 兼容本地模型目录、HF repo id 与单文件权重导入

<div class="intro-actions">
  <a class="md-button md-button--primary" href="getting-started/">快速开始</a>
  <a class="md-button" href="cli/">CLI 文档</a>
  <a class="md-button" href="python-api/">Python API</a>
</div>

## 当前公开基线

| 项目 | 当前状态 |
|---|---|
| 稳定任务面 | `text2image`、`image2image`、`text2video`、`image2video`、`audio2video` |
| 硬件后端 | `cuda`、`ascend` |
| 统一核心对象 | `GenerateRequest`、`GenerateResult`、`RunReport` |
| 标准产物导出 | `PNG`、`MP4` |

默认推荐先做模型发现和请求校验，再接入真实 CUDA / Ascend 硬件执行。

## 为什么是 OmniRT

<div class="grid cards compact-cards" markdown>

- __统一接口契约__

  使用一致的 `GenerateRequest`、`GenerateResult`、`RunReport`，让不同模型家族共享同一套调用方式。

- __后端感知运行时__

  同一份请求可以在 `cuda`、`ascend` 或 `cpu-stub` 下进行校验与执行，后端差异被收敛到运行时层。

- __部署友好__

  默认支持本地模型目录、离线快照准备、错误路径验证和硬件 smoke tests，适合受限网络和混合算力环境。

</div>

## 当前公开任务面

| 任务面 | 说明 | 代表模型 | 输出 |
|---|---|---|---|
| `text2image` | 文本驱动图像生成 | `sd15`, `sdxl-base-1.0`, `flux2.dev`, `qwen-image` | PNG |
| `image2image` | 图像引导图像生成 | `sd15`, `sd21`, `sdxl-base-1.0`, `sdxl-refiner-1.0` | PNG |
| `text2video` | 文本驱动视频生成 | `wan2.2-t2v-14b`, `cogvideox-2b`, `hunyuan-video` | MP4 |
| `image2video` | 首帧引导视频生成 | `svd`, `svd-xt`, `wan2.2-i2v-14b`, `ltx2-i2v` | MP4 |
| `audio2video` | 音频驱动数字人 | `soulx-flashtalk-14b` | MP4 |

## 模型版图

Registry 的完整清单由自动生成页提供：[_generated/models.md](_generated/models.md)，也可本地运行 `omnirt models`。

`soulx-flashtalk-14b` 和 `image2image` 均已公开；`inpaint`、`edit`、`video2video` 底层接线已部分铺设，但仍在向完整公开任务面演进。

## 稳定边界

<div class="split-panels">
  <section>
    <h3>已经稳定公开</h3>
    <ul>
      <li>模型发现与请求校验</li>
      <li>统一 CLI 与 Python API</li>
      <li>`image2image` 正式公开支持</li>
      <li>本地模型目录与离线部署模式</li>
      <li>CUDA / Ascend 后端抽象</li>
    </ul>
  </section>
  <section>
    <h3>仍在演进中的能力</h3>
    <ul>
      <li>`inpaint` / `edit` / `video2video` 的公开能力继续完善</li>
      <li>更强的模型能力自发现</li>
      <li>更细粒度的模型参数帮助</li>
      <li>更完整的跨后端硬件验证矩阵</li>
    </ul>
  </section>
</div>

## 上手路径

<div class="grid cards compact-cards" markdown>

- __快速开始__

  从安装、`omnirt models`、`omnirt validate` 到第一条生成请求的最短路径。

  [进入快速开始](getting-started.md)

- __CLI 使用__

  了解 `generate`、`validate`、`models` 三个核心命令，以及 `inputs` / `config` 的拆分规则。

  [查看 CLI 文档](cli.md)

- __Python 接入__

  使用 typed request helpers、`generate(...)`、`validate(...)` 和 `pipeline(...)` 封装。

  [查看 Python API](python-api.md)

- __架构与部署__

  深入查看运行时分层、服务协议、中国区部署和 Ascend 后端说明。

  [阅读架构文档](architecture.md)

- __支持状态__

  查看哪些模型已经接入、哪些已经完成真机 smoke，以及当前仍待补齐的高优先级目标。

  [查看支持状态](support-status.md)

- __对标 vLLM-Omni__

  查看当前项目和 `vLLM-Omni` 在服务化、分布式、吞吐与全模态编排上的差距，以及建议的补齐路线。

  [查看能力缺口 roadmap](vllm-omni-gap-roadmap.md)

</div>
