# 模型

OmniRT 的模型由 **registry** 管理：每个模型通过 `@register_model` 装饰器声明自己支持哪些任务、接受哪些 adapter、最小显存需求、推荐 preset。本节按三张表组织：

| 页面 | 作用 |
|---|---|
| [模型清单](supported_models.md) | 自动生成的完整注册表（与 `omnirt models` 等价） |
| [支持状态](support_status.md) | 人工维护的"真机 smoke"与"部分支持"追踪 |
| [路线图](roadmap.md) | 高优先级待接入模型与其状态 |

!!! tip "在命令行里查模型"
    `omnirt models` 列出全部；`omnirt models <id>` 查看某个模型的 `ModelCapabilities` 详情（支持任务、adapter、显存、推荐 preset）。
