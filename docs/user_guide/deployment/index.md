# 部署指南

OmniRT 的部署路径按**硬件后端**和**网络环境**两个维度拆分：

| 场景 | 推荐入口 |
|---|---|
| NVIDIA GPU 生产部署 | [CUDA 部署](cuda.md) |
| 昇腾 Atlas / 910 / 910B | [Ascend 后端](ascend.md) |
| 国内网络 / 内网 / 离线 | [国内部署](china_mirrors.md) |
| 容器化（Docker / k8s） | [Docker 与容器](docker.md) |
| 网关 + worker + Redis / OTLP | [分布式服务](distributed_serving.md) |

!!! tip "先跑通 CPU stub"
    正式部署前，建议先用 `--backend cpu-stub` 走一次 `omnirt validate` 与 `omnirt generate --dry-run`，确认请求契约与模型 registry 无问题，再切到真实硬件。详见 [快速开始](../../getting_started/quickstart.md)。

如果你的部署目标包含异步 job、跨进程共享作业状态、Prometheus 指标采集或远程 worker，请直接继续阅读 [分布式服务](distributed_serving.md)。

## 文档版本

线上文档使用版本化发布：`/latest/` 指向当前 main 文档，正式版本会保留在 `/vX.Y.Z/` 这样的路径下。部署生产环境时，优先查看与你安装的 Python 包或 Docker 镜像 tag 对应的版本文档。
