# OmniRT 服务协议

本文档定义了面向服务化集成场景的 OmniRT 公开请求与响应结构。

## 版本约定

- 当前 schema 版本：`0.1.0`
- 在 OmniRT 仍处于 1.0 之前时，breaking change 应提升 minor version
- 新增字段应尽量保持向后兼容

## 请求结构

推荐的服务请求结构与 `GenerateRequest` 保持一致。

```json
{
  "task": "text2image",
  "model": "flux2.dev",
  "backend": "auto",
  "inputs": {
    "prompt": "a cinematic sci-fi city at sunrise"
  },
  "config": {
    "preset": "balanced",
    "width": 1024,
    "height": 1024
  },
  "adapters": null
}
```

## 请求规则

- `task` 用于标识用户可见的任务面，例如 `text2image`、`text2video`、`image2video`
- `model` 必须是 OmniRT 的 registry id，而不是上游 Diffusers 的原始类名
- `inputs` 放语义内容输入，例如 `prompt`、`negative_prompt`、`image`、`num_frames`、`fps`
- `config` 放执行设置，例如 `preset`、`scheduler`、`num_inference_steps`、`guidance_scale`、`height`、`width`、`dtype`、`output_dir`
- `adapters` 放可选的 LoRA 引用

## 校验契约

在真正执行之前，服务层应该暴露与 `omnirt validate` 一致的校验行为：

- 对未知模型给出相近建议
- 拒绝任务与模型不匹配的请求
- 拒绝不支持的输入和配置字段
- 解析模型默认值和命名 preset
- 返回最终解析出的后端

## 响应结构

执行响应与 `GenerateResult` 保持一致。

```json
{
  "outputs": [
    {
      "kind": "image",
      "path": "outputs/flux2.dev-random-0.png",
      "mime": "image/png",
      "width": 1024,
      "height": 1024,
      "num_frames": null
    }
  ],
  "metadata": {
    "run_id": "3f33c54e-f4a9-4d22-a4f5-8de4d0c5f5d4",
    "task": "text2image",
    "model": "flux2.dev",
    "backend": "cuda",
    "timings": {
      "prepare_conditions_ms": 3.1
    },
    "memory": {
      "peak_mb": 4096
    },
    "backend_timeline": [],
    "config_resolved": {
      "width": 1024,
      "height": 1024,
      "num_inference_steps": 40
    },
    "artifacts": [],
    "error": null,
    "latent_stats": null,
    "schema_version": "0.1.0"
  }
}
```

## 稳定性建议

- 客户端应把未知响应字段视为前向兼容的新增内容
- 客户端应通过 `schema_version` 决定解析升级策略
- `artifact.path` 默认表示本地运行时输出路径，除非上层服务额外把它映射成对象存储 URL
