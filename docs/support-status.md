# 当前支持状态

本文档记录 `omnirt` 当前已经接入、已做真机 smoke、以及尚未完成的重点模型。

最近更新：`2026-04-20`

## 当前公开任务面

- `text2image`
- `image2image`
- `text2video`
- `image2video`
- `audio2video`

## 已接入模型

### 图像生成

- `sd15`
- `sd21`
- `sdxl-base-1.0`
- `sdxl-refiner-1.0`
- `sdxl-turbo`
- `sd3-medium`
- `sd3.5-large`
- `sd3.5-large-turbo`
- `kolors`
- `flux-dev`
- `flux-depth`
- `flux-schnell`
- `flux-canny`
- `flux-fill`
- `flux-kontext`
- `flux2.dev`
- `flux2-dev`
- `glm-image`
- `hunyuan-image-2.1`
- `omnigen`
- `qwen-image`
- `qwen-image-edit`
- `qwen-image-edit-plus`
- `qwen-image-layered`
- `sana-1.6b`
- `ovis-image`
- `hidream-i1`
- `pixart-sigma`
- `bria-3.2`
- `lumina-t2x`

### 视频生成

- `svd`
- `svd-xt`
- `animate-diff-sdxl`
- `mochi`
- `cogvideox-2b`
- `cogvideox-5b`
- `kandinsky5-t2v`
- `kandinsky5-i2v`
- `wan2.1-t2v-14b`
- `wan2.1-i2v-14b`
- `wan2.2-t2v-14b`
- `wan2.2-i2v-14b`
- `hunyuan-video`
- `hunyuan-video-1.5-t2v`
- `hunyuan-video-1.5-i2v`
- `helios-t2v`
- `helios-i2v`
- `sana-video`
- `ltx-video`
- `ltx2-i2v`
- `skyreels-v2`

### 音频驱动视频

- `soulx-flashtalk-14b`

## 已完成真机 smoke

以下模型已经基于本地模型目录完成真实硬件 smoke：

- `sdxl-base-1.0`
  CUDA: `<cuda-host>`
  Ascend: `<ascend-host>`
- `svd-xt`
  CUDA: `<cuda-host>`
  Ascend: `<ascend-host>`

## 已接入但仍待真机 smoke

这一批模型已经完成 registry、请求面和本地单测，但还没有在仓库里沉淀出“已验证”的本地模型目录与双后端 smoke 结果：

- `sdxl-refiner-1.0`
- `chronoedit`
- `flux-depth`
- `flux-canny`
- `flux-fill`
- `flux-kontext`
- `qwen-image-edit`
- `qwen-image-edit-plus`
- `qwen-image-layered`
- `animate-diff-sdxl`
- `kolors`
- `pixart-sigma`
- `bria-3.2`
- `lumina-t2x`
- `mochi`
- `skyreels-v2`

其中一部分对应 smoke 用例已经具备。对于已经公开的 `image2image`，当前最推荐的模型起点是 `sdxl-base-1.0`、`sdxl-refiner-1.0`、`sd15` 和 `sd21`：

- `tests/integration/test_sdxl_refiner_cuda.py`
- `tests/integration/test_sdxl_refiner_ascend.py`
- `tests/integration/test_flux_fill_cuda.py`
- `tests/integration/test_flux_fill_ascend.py`
- `tests/integration/test_image_edit_cuda.py`
- `tests/integration/test_image_edit_ascend.py`

## 部分支持

- `helios`
  当前以 `helios-t2v` / `helios-i2v` 两个 registry key 形式提供。
- `hunyuan-video-1.5`
  当前以 `hunyuan-video-1.5-t2v` / `hunyuan-video-1.5-i2v` 两个 registry key 形式提供。

## 尚未完成的重点目标

- 暂无新的高优先级缺口；当前更主要的是把已接入模型继续做真机 smoke 和国内可落地模型源验证

## 参考文档

- [模型支持路线图](model-support-roadmap.md)
- [中国区部署](china-deployment.md)
