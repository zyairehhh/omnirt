# 模型接入指南

本文档给出把一个 Diffusers 风格新模型接入 `omnirt` 的最短路径。

## 1. 新增 pipeline

创建 `src/omnirt/models/<model_name>/pipeline.py` 并注册对应类：

```python
@register_model(
    id="my-model",
    task="text2image",
    default_backend="auto",
    resource_hint={"min_vram_gb": 12, "dtype": "fp16"},
)
class MyPipeline(BasePipeline):
    ...
```

实现 `BasePipeline` 的五个阶段：

1. `prepare_conditions`
2. `prepare_latents`
3. `denoise_loop`
4. `decode`
5. `export`

## 2. 复用后端契约

通过 `self.runtime.wrap_module(...)` 包装对后端敏感的子模块。这样新模型无需编写额外的模型专属后端代码，也能获得 compile、override 和 eager fallback 行为。

## 3. 使用受支持的格式

- 权重：`safetensors`
- adapters：在 pipeline 初始化时校验，在运行时 pipeline 真正 materialize 后应用一次；单文件 adapter 既可以来自本地，也可以来自 Hugging Face
- 产物：输出带具体文件路径的 `Artifact` 记录

## 4. 增加测试

- 使用 fake runtime 和 fake Diffusers 对象编写单元测试
- 在 `tests/integration/` 下补充 smoke integration tests
- 如果模型参与跨后端校验，再补 parity metric 覆盖

## 5. 记录假设与约束

至少记录：

- 模型来源或 registry id
- 最低显存提示
- 支持的任务 schema
- 后端注意事项
- 导出的产物格式

`src/omnirt/models/` 下的 SDXL 与 SVD pipeline 是当前推荐的参考实现。
